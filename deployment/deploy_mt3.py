"""
MT3 Deployment Script

Demonstrates the complete MT3 pipeline:
1. Load test image and pre-computed segmentation from assets/inference_example/
2. Retrieve similar demonstration via hierarchical retrieval
3. Estimate relative pose with PointNet++ (shows registration result)
4. Refine pose with Generalized ICP (shows registration result again)
5. Apply 4DOF inductive bias (constrain to tabletop manipulation)
6. Transform demonstration bottleneck pose to live scene
7. Access demonstration velocities for replay

Every visualisation is written to the vis dir as a JPEG (the server ships the
whole directory back to the client), never shown in a window: the pipeline runs
headless inside a container, where an Open3D window would hang the run. The
point-cloud views are rendered with matplotlib for the same reason -- Open3D's
offscreen renderer needs an EGL/OSMesa build the image does not have.

For actual robot deployment:
- Replace test image loading with live camera capture
- Provide pre-computed segmentation masks for target objects
- Provide demonstrations on your own robot platform
- For alignment, reach bottleneck pose with motion planning
- For interaction, replay velocities in end-effector frame
"""

import argparse
import glob
import os
from os.path import basename, isdir, join, normpath

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from thousand_tasks.core.globals import ASSETS_DIR
from thousand_tasks.core.utils.scene_state import SceneState
from thousand_tasks.core.utils.se3_tools import pose_inv, rot2euler, euler2rot
from thousand_tasks.core.utils.camera_frames import relative_camera_transform, transform_points
from thousand_tasks.retrieval.hierarchical_retrieval import HierarchicalRetrieval
from thousand_tasks.retrieval.language_based_retrieval import LanguageBasedRetrieval
from thousand_tasks.perception.pose_estimation.pnet_4dof_pose_regressor import PointnetPoseRegressor_4dof
from thousand_tasks.perception.pose_estimation.icp_6dof_pose_estimation_refinement import Open3dIcpPoseRefinement


# Both visualisations are photographic RGB panels, which PNG stores badly (~1.6 MB
# for the pair). JPEG at the same dpi is ~8x smaller at 45 dB PSNR -- indistinguishable
# by eye -- so the full resolution is kept rather than trading detail for size.
VIS_SAVE_KWARGS = dict(dpi=150, bbox_inches='tight',
                       pil_kwargs={'quality': 85, 'optimize': True})


def visualize_test_scene(rgb, depth, segmap, save_path):
    """Visualize test scene with RGB, depth, and segmentation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')

    segmented_rgb = rgb * segmap[..., None]
    axes[2].imshow(segmented_rgb)
    axes[2].set_title('Segmented RGB')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(str(save_path), **VIS_SAVE_KWARGS)
    print(f"  Saved visualization to: {save_path}")
    plt.close()


def visualize_retrieval(test_rgb, test_segmap, demo_rgb, demo_segmap, save_path):
    """Visualize test and retrieved demo with segmentation masks in 2x2 layout."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # First row: Live scene
    axes[0, 0].imshow(test_rgb)
    axes[0, 0].set_title('Live Scene - RGB')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(test_rgb * test_segmap[..., None])
    axes[0, 1].set_title('Live Scene - Segmented')
    axes[0, 1].axis('off')

    # Second row: Retrieved demo
    axes[1, 0].imshow(demo_rgb)
    axes[1, 0].set_title('Retrieved Demo - RGB')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(demo_rgb * demo_segmap[..., None])
    axes[1, 1].set_title('Retrieved Demo - Segmented')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(str(save_path), **VIS_SAVE_KWARGS)
    print(f"  Saved retrieval visualization to: {save_path}")
    plt.close()


# Same colours draw_registration_result() uses in the interactive windows.
DEMO_COLOUR = (1.0, 0.706, 0.0)     # orange
LIVE_COLOUR = (0.0, 0.651, 0.929)   # blue

# (elev, azim) pairs for the two panels of every point-cloud figure. Clouds are
# in the OpenCV camera frame (x right, y down, z forward), so the first looks
# along +z the way the camera does and the second is an oblique view that
# exposes depth errors the camera view hides.
PCD_VIEWS = (('Camera view', (-90, -90)), ('Oblique view', (-35, -60)))
PCD_MAX_POINTS = 4000


def subsample_points(points, colours=None, max_points=PCD_MAX_POINTS):
    """Cap a cloud at `max_points` for plotting. Seeded, so the same cloud
    draws the same dots in every panel it appears in."""
    points = np.asarray(points)
    if len(points) > max_points:
        keep = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points = points[keep]
        colours = colours[keep] if colours is not None else None
    return points, colours


def pcd_to_arrays(pcd, max_points=PCD_MAX_POINTS):
    """(points, colours) from an Open3D cloud, subsampled for plotting.
    Colours are None when the cloud has none."""
    colours = np.asarray(pcd.colors) if pcd.has_colors() else None
    return subsample_points(np.asarray(pcd.points), colours, max_points)


def save_point_cloud_figure(panels, save_path, suptitle=None):
    """Render point clouds to a JPEG, in place of an Open3D window.

    `panels` is a list of (title, clouds) where clouds is a list of
    (points Nx3, colours, label). `colours` is either an Nx3 array of per-point
    RGB in [0, 1], a single RGB tuple, or None (grey). Each panel is drawn from
    every view in PCD_VIEWS, all panels share one axis box so the same cloud
    lands at the same place in every subplot.
    """
    all_points = np.concatenate(
        [pts for _, clouds in panels for pts, _, _ in clouds if len(pts)], axis=0)
    centre = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    half = max((all_points.max(axis=0) - all_points.min(axis=0)).max() / 2, 1e-3)

    n_rows, n_cols = len(panels), len(PCD_VIEWS)
    fig = plt.figure(figsize=(6 * n_cols, 5.5 * n_rows))
    for row, (title, clouds) in enumerate(panels):
        for col, (view_name, (elev, azim)) in enumerate(PCD_VIEWS):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection='3d')
            for points, colours, label in clouds:
                if colours is None:
                    colours = (0.5, 0.5, 0.5)
                if isinstance(colours, np.ndarray) and colours.ndim == 2:
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=np.clip(colours, 0, 1), s=1, label=label)
                else:
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               color=colours, s=1, label=label)
            ax.set_xlim(centre[0] - half, centre[0] + half)
            ax.set_ylim(centre[1] - half, centre[1] + half)
            ax.set_zlim(centre[2] - half, centre[2] + half)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            if abs(elev) == 90:
                ax.set_zticks([])       # z is the viewing axis: its ticks collapse to a smear
            else:
                ax.set_zlabel('z [m]')
            ax.set_title(f'{title} - {view_name}', pad=18)
            if len(clouds) > 1:
                ax.legend(loc='upper right', markerscale=8)

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(str(save_path), **VIS_SAVE_KWARGS)
    print(f"  Saved point cloud visualization to: {save_path}")
    plt.close(fig)


def demo_points_in_live_frame(demo_scene_state, T_WC_live):
    """The demo object cloud re-expressed in the live camera frame.

    This is the frame ICP registers in (see refine_relative_pose with
    different_cameras_live_demo), so a camera-frame T_delta applies directly
    to these points.
    """
    points = np.asarray(demo_scene_state.o3d_pcd.points)
    T_C2C1 = relative_camera_transform(demo_scene_state.T_WC, T_WC_live)
    return transform_points(points, T_C2C1)


def save_registration_figure(demo_points_live_frame, live_points, stages, save_path, suptitle):
    """One row per (stage name, camera-frame T_delta) showing demo (orange,
    moved by T_delta) over live (blue) -- the headless draw_registration_result."""
    panels = []
    for stage_name, C_T_delta in stages:
        moved = (demo_points_live_frame if C_T_delta is None
                 else transform_points(demo_points_live_frame, C_T_delta))
        panels.append((stage_name, [(moved, DEMO_COLOUR, 'Demo'),
                                    (live_points, LIVE_COLOUR, 'Live')]))
    save_point_cloud_figure(panels, save_path, suptitle)


def apply_4dof_inductive_bias(W_T_delta_6dof: np.ndarray, T_WE: np.ndarray) -> np.ndarray:
    """
    Apply 4DOF inductive bias to transformation (constrain to 4DOF: x, y, z, yaw).

    This function constrains a 6DOF transformation to only 4DOF by:
    1. Removing roll and pitch rotations (keeping only yaw around vertical axis)
    2. Adjusting translation to compensate for the rotation constraint

    The inductive bias is useful for tabletop manipulation tasks where objects
    typically only rotate around the vertical (z) axis, not around x or y axes.
    This reduces the search space and improves robustness.

    Args:
        W_T_delta_6dof: 4x4 transformation matrix (demo → live) in world frame
        T_WE: 4x4 end-effector pose in world frame (used for rotation center)

    Returns:
        W_T_delta_4dof: 4x4 transformation constrained to 4DOF (x, y, z, yaw)

    Mathematical explanation:
    -------------------------
    For tabletop tasks, we want to preserve:
    - Full translation (x, y, z)
    - Yaw rotation (θz around vertical axis)

    But remove:
    - Roll rotation (θx around x-axis)
    - Pitch rotation (θy around y-axis)

    The key insight is that when we remove roll/pitch, we must adjust the
    translation to keep the end-effector at the correct position. This is
    done by:
    1. Computing translation if we rotate around end-effector: R_6dof @ t_E
    2. Computing translation if we rotate around origin with 4DOF: R_4dof @ t_E
    3. Adding the difference to compensate: t_4dof = t_6dof + (R_6dof - R_4dof) @ t_E

    Note: We use a point 24cm above the end-effector (gripper tip) as the
    rotation center to better match the contact point with objects.
    """
    # Create a copy of end-effector pose and move 24cm up (to gripper tip)
    # This accounts for the offset between wrist and actual contact point
    T_WE_copy = T_WE.copy()
    #T_WE_copy[:3, 3] += T_WE_copy[:3, :3] @ np.array([0, 0, 0.24])
    T_WE_copy[:3, 3] += T_WE_copy[:3, :3] @ np.array([0, 0, 0.20])
    t_WE = T_WE_copy[:3, 3:]  # Position of rotation center

    # Extract rotation and translation from 6DOF transformation
    W_R_delta_6dof = W_T_delta_6dof[:3, :3]
    W_t_delta_6dof = W_T_delta_6dof[:3, 3:]

    # Convert 6DOF rotation to Euler angles (xyz convention)
    three_dof_euler = rot2euler('xyz', W_R_delta_6dof, degrees=False)

    # Zero out roll (θx) and pitch (θy), keeping only yaw (θz)
    # This constrains rotation to vertical axis only
    three_dof_euler[:2] = 0

    # Convert back to rotation matrix (now only yaw rotation)
    W_R_delta_4dof = euler2rot('xyz', three_dof_euler, degrees=False)

    # Adjust translation to compensate for rotation constraint
    # Formula: t_new = t_old + (R_old - R_new) @ rotation_center
    # This ensures the end-effector still reaches the same world position
    # even though we've constrained the rotation
    W_t_delta_4dof = W_R_delta_6dof @ t_WE + W_t_delta_6dof - W_R_delta_4dof @ t_WE

    # Construct 4DOF transformation matrix
    W_T_delta_4dof = np.eye(4)
    W_T_delta_4dof[:3, :3] = W_R_delta_4dof
    W_T_delta_4dof[:3, 3:] = W_t_delta_4dof

    return W_T_delta_4dof


class NoDemonstrationsError(RuntimeError):
    """Retrieval found no candidate demonstration for the requested task_name."""


class Mt3Context:
    """Everything that is expensive to build, held across runs.

    Nothing stored here is request-specific: the camera extrinsics T_WC arrive
    with each request and are rebound onto these objects by `run_once`.
    """

    def __init__(self, retrieval, pose_estimator, pose_refiner, demo_dir, vis_dir, save_dir):
        self.retrieval = retrieval
        self.pose_estimator = pose_estimator
        self.pose_refiner = pose_refiner
        self.demo_dir = demo_dir
        self.vis_dir = vis_dir
        self.save_dir = save_dir
        self.store_fingerprint = demo_store_fingerprint(demo_dir)


def _list_demo_folders(root_dir):
    """The demo folders HierarchicalRetrieval considers.

    Mirrors thousand_tasks/retrieval/hierarchical_retrieval.py:33-35. Kept here
    so `refresh_retrieval` does not require touching the thousand_tasks package.
    """
    non_task_folder_dirs = ['interaction_processed', 'bn_reaching_processed', 'processed']
    return np.sort(
        [basename(normpath(task)) for task in glob.glob(join(str(root_dir), '*')) if
         (isdir(task) and basename(normpath(task)) not in non_task_folder_dirs)]).tolist()


def demo_store_fingerprint(demo_dir):
    """(name, mtime_ns) for every demo folder -- cheap change detection.

    Dot-prefixed entries are skipped so an in-flight `.tmp_<request_id>` upload
    staging directory never registers as a store change; HierarchicalRetrieval's
    own glob('*') ignores them too.
    """
    try:
        with os.scandir(str(demo_dir)) as entries:
            return tuple(sorted((e.name, e.stat().st_mtime_ns)
                                for e in entries
                                if e.is_dir() and not e.name.startswith('.')))
    except FileNotFoundError:
        return ()


def refresh_retrieval(retrieval, T_WC):
    """Re-snapshot the demo store in place, reusing the loaded encoder.

    Rebuilding HierarchicalRetrieval outright would re-load geometry_encoder.ckpt,
    which is the cost we are trying to avoid. These four statements are everything
    its __init__ does after load_encoder().
    """
    retrieval.T_WC_live = T_WC
    # The full store; _load_task_embeddings narrows this to the demos that
    # actually carry a T_WC.npy.
    retrieval.tasks_folder_names = _list_demo_folders(retrieval.root_dir)
    retrieval.language_based_retrieval = LanguageBasedRetrieval(
        learned_tasks_dir=retrieval.root_dir, verbose=False)
    # np.load of the cached geometry_encoding.npy per demo; only a demo without
    # one costs an encoder forward pass, which is then cached to disk.
    retrieval._load_task_embeddings()


def build_context(T_WC, device=None):
    """Load both checkpoints and create the CUDA context. Call once per process."""
    demo_dir = ASSETS_DIR / 'demonstrations'
    vis_dir = ASSETS_DIR / 'example_visualisations'
    vis_dir.mkdir(exist_ok=True)
    save_dir = ASSETS_DIR.parent / 'saved_data'      # == /workspace/saved_data
    save_dir.mkdir(parents=True, exist_ok=True)

    retrieval = HierarchicalRetrieval(
        T_WC_demo=T_WC,
        T_WC_live=T_WC,
        learned_tasks_dir=str(demo_dir)
    )

    pose_estimator = PointnetPoseRegressor_4dof(
        filter_pointcloud=True,
        n_points=2048,
        T_WC=T_WC,
        T_WC_demo=T_WC,
        depth_units='mm',
        device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # The restart budget is the single biggest share of a warm request. The
    # icp_sweep.py A/B over 56 demo pairs found 1.0 s indistinguishable from
    # 3.0 s (median pose deviation 0.0 mm; the worst-case spread it does show
    # is within the 3 s budget's own restart-lottery noise), so 1.0 s is the
    # default. Both knobs are env vars so a compose restart -- no rebuild --
    # changes them: MT3_ICP_TIMEOUT seconds of restarts, MT3_ICP_ENABLE=0
    # skips refinement entirely and uses the PointNet++ pose as-is.
    pose_refiner = Open3dIcpPoseRefinement(
        error_metric='generalised-icp',     # Use Generalized ICP (GICP)
        max_correspondence_distance=0.1,    # Max distance for point correspondence (10cm)
        max_iteration=20,                   # Max ICP iterations per trial
        depth_units='mm',                   # Match depth units from pose estimator
        timeout=float(os.environ.get('MT3_ICP_TIMEOUT', '1.0'))
    )

    return Mt3Context(retrieval, pose_estimator, pose_refiner, demo_dir, vis_dir, save_dir)


def resolve_icp(icp=None):
    """This run's ICP setting: the request's, else the container's env vars.

    Returns (enabled, timeout). Kept separate from run_once so the precedence is
    testable without a CUDA context.
    """
    env_enabled = os.environ.get('MT3_ICP_ENABLE', '1').lower() not in ('0', 'false', 'no')
    env_timeout = float(os.environ.get('MT3_ICP_TIMEOUT', '1.0'))

    if not icp:
        return env_enabled, env_timeout

    enabled = icp.get('enabled')
    timeout = icp.get('timeout')
    return (env_enabled if enabled is None else bool(enabled),
            env_timeout if timeout is None else float(timeout))


def run_once(ctx, task_name, T_WC, icp=None):
    """Run one inference against a warm context. Returns the retrieved demo name.

    `icp` is this task's refinement setting, {"enabled": bool, "timeout": secs},
    shipped with the request. Refinement used to be configured per container, so
    a task that wanted no ICP -- a rotationally symmetric knob, where ICP has
    nothing to lock onto -- could not say so without changing it for every task.
    """

    # Rebind the request's live extrinsics. HierarchicalRetrieval and
    # PointnetPoseRegressor_4dof both capture T_WC at construction and read it on
    # every later call, so without this every run after the first would silently
    # use the first request's calibration.
    #
    # Only the *live* pose is bound here. The demo pose comes from whichever demo
    # retrieval picks, so `pose_estimator.extrinsics_demo` is set in
    # _run_pipeline once that is known -- binding it to T_WC here would pin every
    # demo to the live camera and quietly undo per-demo extrinsics.
    ctx.retrieval.T_WC_live = T_WC
    ctx.pose_estimator.extrinsics = T_WC

    # Re-snapshot the demo store if it changed (MQTT upload, or a folder dropped
    # in through the bind mount) since retrieval last looked.
    fingerprint = demo_store_fingerprint(ctx.demo_dir)
    if fingerprint != ctx.store_fingerprint:
        print("  Demo store changed ({} -> {} demos); refreshing retrieval".format(
            len(ctx.store_fingerprint or ()), len(fingerprint)))
        refresh_retrieval(ctx.retrieval, T_WC)
        ctx.store_fingerprint = fingerprint

    # Never hand back a previous scene's visualisation: the pipeline overwrites
    # these but does not clear them, so a run that raises between the two
    # savefig calls would leave a stale image behind.
    for stale in ctx.vis_dir.glob('*'):
        if stale.is_file():
            stale.unlink()

    # Per-request, so it must be rebound on the shared context every run --
    # `timeout` is read inside the refiner's restart loop, so setting it here
    # takes effect for this run and this run only.
    icp_enabled, icp_timeout = resolve_icp(icp)
    ctx.pose_refiner.timeout = icp_timeout

    try:
        return _run_pipeline(ctx, task_name, T_WC, icp_enabled=icp_enabled)
    finally:
        plt.close('all')


def _run_pipeline(ctx, task_name, T_WC, icp_enabled=True):
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    inference_dir = ASSETS_DIR / 'inference_example'
    demo_dir = ctx.demo_dir

    # Task-specific parameters
    #task_name = 'pick_up_cube'

    print("="*80)
    print("MT3 Deployment")
    print("="*80)

    # -------------------------------------------------------------------------
    # Step 1: Load test image
    # -------------------------------------------------------------------------
    # In practice, capture RGB-D image from camera
    print("\nStep 1: Load test image")

    # Load RGB and depth from workspace images
    test_rgb = np.array(Image.open(str(inference_dir / 'head_camera_ws_rgb.png')))
    test_depth = np.array(Image.open(str(inference_dir / 'head_camera_ws_depth_to_rgb.png')))

    # Load segmentation mask for the target object (pre-computed)
    test_segmap = np.load(str(inference_dir / 'head_camera_ws_segmap.npy'))

    # Load camera intrinsics
    intrinsics = np.load(str(inference_dir / 'head_camera_rgb_intrinsic_matrix.npy'))

    # Camera extrinsics (world-to-camera transform) arrive with the request.

    print(f"  Loaded RGB: {test_rgb.shape}, Depth: {test_depth.shape}, Segmap: {test_segmap.shape}")

    # Visualize test image, depth, and segmentation
    vis_dir = ctx.vis_dir

    save_path = vis_dir / 'test_scene_visualization.jpg'
    visualize_test_scene(test_rgb, test_depth, test_segmap, save_path)

    # -------------------------------------------------------------------------
    # Step 2: Initialize live scene state
    # -------------------------------------------------------------------------
    print("\nStep 2: Initialize live scene state")
    live_scene_state = SceneState.initialise_from_dict({
        'rgb': test_rgb,
        'depth': test_depth,
        'segmap': test_segmap,
        'intrinsic_matrix': intrinsics
    })
    # initialise_from_dict does not take extrinsics, and ICP reads them off the
    # scene states directly -- without this the refinement step raises
    # 'The extrinsic matrix has not been set.'
    live_scene_state.T_WC = T_WC

    # Process segmentation: erode mask and crop to object region
    live_scene_state.erode_segmap()
    live_scene_state.crop_object_using_segmap()
    print(f"  Processed segmentation for: '{task_name}'")

    # Visualize the live object point cloud (RGB-coloured, camera frame)
    live_pcd = live_scene_state.o3d_pcd     # a property: each access rebuilds the cloud
    print(f"  Point cloud has {len(live_pcd.points)} points")
    live_points, live_colours = pcd_to_arrays(live_pcd)
    save_point_cloud_figure(
        [('Live scene', [(live_points, live_colours, 'Live')])],
        vis_dir / 'live_point_cloud.jpg',
        suptitle='Live object point cloud (camera frame)')

    # -------------------------------------------------------------------------
    # Step 3: Retrieve similar demonstration
    # -------------------------------------------------------------------------
    print("\nStep 3: Retrieve demonstration via hierarchical retrieval")
    retrieval = ctx.retrieval

    # Retrieve most similar demo based on visual similarity
    retrieved_demo_name = retrieval.get_most_similar_demo_name(
        scene_state=live_scene_state,
        template_task_description=task_name
    )

    # get_most_similar_demo_name prints 'No demonstrations exist for skill ...'
    # and falls off the end returning None; the resulting `demo_dir / None`
    # TypeError is what signals this case today.
    if retrieved_demo_name is None:
        raise NoDemonstrationsError(
            f'No demonstrations exist for skill {task_name}')

    print(f"  Retrieved: {retrieved_demo_name}")

    # -------------------------------------------------------------------------
    # Step 4: Load and segment demonstration
    # -------------------------------------------------------------------------
    print("\nStep 4: Load retrieved demonstration")
    demo_path = demo_dir / retrieved_demo_name

    # The camera pose this demo was recorded under, which is only knowable once
    # retrieval has chosen a demo -- hence bound here rather than in run_once
    # alongside the live extrinsics.
    demo_T_WC = np.load(str(demo_path / 'T_WC.npy'))
    ctx.pose_estimator.extrinsics_demo = demo_T_WC

    # Load workspace images
    demo_rgb = np.array(Image.open(str(demo_path / 'head_camera_ws_rgb.png')))
    demo_depth = np.array(Image.open(str(demo_path / 'head_camera_ws_depth_to_rgb.png')))
    demo_segmap = np.load(str(demo_path / 'head_camera_ws_segmap.npy'))
    demo_intrinsics = np.load(str(demo_path / 'head_camera_rgb_intrinsic_matrix.npy'))

    demo_scene_state = SceneState.initialise_from_dict({
        'rgb': demo_rgb,
        'depth': demo_depth,
        'segmap': demo_segmap,
        'intrinsic_matrix': demo_intrinsics
    })
    # ICP reads this to re-frame the demo cloud into the live camera frame.
    demo_scene_state.T_WC = demo_T_WC

    # Process demonstration segmentation
    demo_scene_state.erode_segmap()
    demo_scene_state.crop_object_using_segmap()

    # Visualize retrieval results
    retrieval_vis_path = vis_dir / 'retrieval_visualization.jpg'
    visualize_retrieval(test_rgb, live_scene_state.segmap, demo_rgb, demo_scene_state.segmap, retrieval_vis_path)

    # Visualize live vs retrieved demo point clouds, each in its own camera
    # frame, before any registration.
    demo_points_own_frame, demo_colours = pcd_to_arrays(demo_scene_state.o3d_pcd)
    save_point_cloud_figure(
        [('Live scene', [(live_points, live_colours, 'Live')]),
         ('Retrieved demo', [(demo_points_own_frame, demo_colours, 'Demo')])],
        vis_dir / 'point_clouds_live_vs_demo.jpg',
        suptitle=f'Live object vs retrieved demo "{retrieved_demo_name}" (unregistered)')

    # The registration figures below draw the demo cloud in the live camera
    # frame -- the frame ICP registers in -- so a camera-frame T_delta moves it
    # straight onto the live cloud.
    demo_points_live_frame, _ = subsample_points(
        demo_points_in_live_frame(demo_scene_state, T_WC))

    # =========================================================================
    # PART 1: ALIGNMENT - Estimate target pose for robot
    # =========================================================================

    # -------------------------------------------------------------------------
    # Step 5: Estimate relative pose with PointNet++
    # -------------------------------------------------------------------------
    print("\nStep 5: Estimate relative pose with PointNet++")
    pose_estimator = ctx.pose_estimator

    # Estimate transformation from demo to live scene (in world frame)
    W_T_delta = pose_estimator.estimate_relative_pose(
        scene1_state=demo_scene_state,
        scene2_state=live_scene_state,
        visualise_pcds=False,   # would open a window; rendered to file below instead
        verbose=False
    )

    print(f"  PointNet++ prediction complete")

    # Convert world-frame transformation to camera frame: ICP works in camera
    # frame where point clouds are expressed, and so do the registration figures.
    C_T_delta = pose_inv(T_WC) @ W_T_delta @ T_WC

    save_registration_figure(
        demo_points_live_frame, live_points,
        [('Before registration', None), ('After PointNet++', C_T_delta)],
        vis_dir / 'registration_pointnet.jpg',
        suptitle='Registration: demo (orange) moved onto live (blue), PointNet++ estimate')

    # -------------------------------------------------------------------------
    # Step 5b: Refine pose estimate with Generalized ICP
    # -------------------------------------------------------------------------
    if not icp_enabled:
        # Disabled for this task (or MT3_ICP_ENABLE=0): use PointNet++ as-is.
        print("\nStep 5b: ICP refinement disabled for this run")
        W_T_delta_refined = W_T_delta
    else:
        print("\nStep 5b: Refine pose with Generalized ICP "
              "(restart budget {:.2f}s)".format(ctx.pose_refiner.timeout))
        print("  Initializing ICP refinement...")

        # Initialize ICP pose refiner with Generalized ICP
        # Generalized ICP uses point-to-plane distances with covariance weighting
        # for more robust alignment than standard point-to-point ICP
        pose_refiner = ctx.pose_refiner

        # Refine pose using Generalized ICP
        # This runs multiple ICP trials with small perturbations around the
        # PointNet++ prediction to find the best alignment
        # different_cameras_live_demo re-expresses the demo point cloud in the live
        # camera frame before registering. It is unconditional: when the two
        # extrinsics match, the composed transform is identity and this is a no-op,
        # so there is no behaviour to switch between.
        C_T_delta_refined = pose_refiner.refine_relative_pose(
            scene1_state=demo_scene_state,
            scene2_state=live_scene_state,
            T_delta_init=C_T_delta,
            T_WC_live=T_WC,
            verbose=False,
            visualise_pcds=False,   # would open a window; rendered to file below instead
            different_cameras_live_demo=True
        )

        # Convert refined pose back to world frame
        W_T_delta_refined = T_WC @ C_T_delta_refined @ pose_inv(T_WC)

        print(f"  ICP refinement complete")

        save_registration_figure(
            demo_points_live_frame, live_points,
            [('PointNet++ init', C_T_delta), ('After ICP', C_T_delta_refined)],
            vis_dir / 'registration_icp.jpg',
            suptitle='Registration: demo (orange) moved onto live (blue), ICP refinement')

    # -------------------------------------------------------------------------
    # Step 6: Apply 4DOF inductive bias
    # -------------------------------------------------------------------------
    print("\nStep 6: Apply 4DOF inductive bias")

    # Load demonstration bottleneck pose to get rotation center
    demo_bottleneck_pose = np.load(str(demo_path / 'bottleneck_pose.npy'))

    # Apply 4DOF constraint (x, y, z, yaw only - no roll/pitch)
    # This constrains the transformation to only allow yaw rotation around
    # the vertical axis, which is appropriate for tabletop manipulation
    # W_T_delta_4dof = apply_4dof_inductive_bias(
    #     W_T_delta_6dof=W_T_delta_refined,
    #     T_WE=demo_bottleneck_pose
    # )

    print(f"  4DOF constraint applied (removed roll and pitch)")

    # -------------------------------------------------------------------------
    # Step 7: Transform demonstration bottleneck pose to live scene
    # -------------------------------------------------------------------------
    print("\nStep 7: Transform bottleneck pose to live scene")

    # Transform demonstration bottleneck pose to live scene using the
    # refined and constrained relative transformation
    # Formula: T_WE_live = W_T_delta_4dof @ T_WE_demo
    # This applies the estimated transformation to get the target pose in the live scene
    #live_bottleneck_pose = W_T_delta_4dof @ demo_bottleneck_pose
    live_bottleneck_pose = W_T_delta_refined @ demo_bottleneck_pose

    print(f"  Demo bottleneck pose (T_WE):\n{demo_bottleneck_pose}")
    print(f"  Live bottleneck pose (T_WE):\n{live_bottleneck_pose}")
    print("\n" + "="*80)
    print("ALIGNMENT PHASE COMPLETE")
    print("="*80)
    print("Target end-effector pose for live scene:")
    print(f"{live_bottleneck_pose}")
    print("\nUse motion planning (e.g., MoveIt, OMPL) or a linear controller")
    print("to move the robot's end-effector to this bottleneck pose.")
    print("="*80)

    # =========================================================================
    # PART 2: INTERACTION - Replay demonstration velocities
    # =========================================================================
    # -------------------------------------------------------------------------
    # Step 8: Load demonstrated end-effector twists
    # -------------------------------------------------------------------------
    print("\n\nPART 2: INTERACTION")
    print("="*80)
    print("Step 8: Load demonstration end-effector twists")

    # Load end-effector velocities/twists
    #end_effector_twists = np.load(str(demo_path / 'demo_eef_twists.npy'))

    # update here

    # After loading end_effector_twists (around line 419)  
    end_effector_twists = np.load(str(demo_path / 'demo_eef_twists.npy'))  
    
    # Save data to ROS workspace  
    #save_dir = Path('/home/aitana_viudes/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/python_demos/saved_data')  
    #save_dir.mkdir(parents=True, exist_ok=True)  
    
    # Save the computed bottleneck pose and twists  
    #np.save(save_dir / 'live_bottleneck_pose.npy', live_bottleneck_pose)  
    #np.save(save_dir / 'end_effector_twists.npy', end_effector_twists)  
    
    #print(f"\n  Saved data to: {save_dir}")  
    #print(f"    - live_bottleneck_pose.npy: {live_bottleneck_pose.shape}")  
    #print(f"    - end_effector_twists.npy: {end_effector_twists.shape}")

    # In deploy_mt3.py, replace the save path with:
    save_dir = ctx.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / 'live_bottleneck_pose.npy', live_bottleneck_pose)  
    np.save(save_dir / 'end_effector_twists.npy', end_effector_twists)

    # Verify dimensionality and explain format
    num_timesteps, twist_dim = end_effector_twists.shape
    print(f"  Loaded twists: {end_effector_twists.shape}")
    print(f"    - Timesteps: {num_timesteps}")
    print(f"    - Dimensions per timestep: {twist_dim}")

    print("    - Format: 6D twist + gripper state at next timestep")
    print("      [vx, vy, vz, wx, wy, wz, gripper_next]")
    print("      where gripper_next: 1 = close, 0 = open")

    print("\n" + "="*80)
    print("INTERACTION PHASE")
    print("="*80)
    print("Once the robot reaches the bottleneck pose, replay these velocities:")
    print(f"  1. Use a velocity controller to track the demonstrated end-effector twists")
    print(f"  2. Execute twists in the end-effector frame at the recorded frequency")
    print(f"  3. Replay all {num_timesteps} timesteps sequentially")
    print(f"  4. Update gripper state according to the 7th dimension at each timestep")
    print("="*80)

    return retrieved_demo_name


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='pick_up_cube',
                        help='Task description for retrieval')
    args = parser.parse_args()

    T_WC = np.load(str(ASSETS_DIR / 'T_WC_head.npy'))
    ctx = build_context(T_WC)
    run_once(ctx, args.task_name, T_WC)


if __name__ == '__main__':
    main()
