import time

import numpy as np
import open3d as o3d

from thousand_tasks.perception.relative_pose_estimation.utils.open3d_icp import Open3dICP
from thousand_tasks.core.base_classes.rgbd_pose_estimation_refinement import RgbdPoseEstimationRefinementBase
from thousand_tasks.core.utils.scene_state import SceneState
from thousand_tasks.core.utils.se3_tools import pose_inv
from thousand_tasks.core.utils.se3_tools import rot2euler, euler2rot
from thousand_tasks.core.utils.camera_frames import relative_camera_transform, transform_points
from thousand_tasks.core.utils.visualisation import draw_registration_result


class Open3dIcpPoseRefinement(RgbdPoseEstimationRefinementBase):

    def __init__(self,
                 error_metric: str = 'point-to-point',
                 max_correspondence_distance: float = 0.05,
                 max_iteration: int = 50,
                 depth_units: str = 'mm',
                 timeout=5
                 ):

        self.timeout = timeout
        self.icp_pose_estimator = Open3dICP(error_metric=error_metric,
                                            max_correspondence_distance=max_correspondence_distance,
                                            max_iteration=max_iteration)

        super().__init__(depth_units=depth_units)

    def refine_relative_pose(self,
                             scene1_state: SceneState,
                             scene2_state: SceneState,
                             T_WC_live: np.ndarray,
                             T_delta_init: np.ndarray,  # expressed in camera frame
                             verbose=True,
                             visualise_pcds=False,
                             different_cameras_live_demo: bool = False,
                             **kwargs) -> np.ndarray:

        self.check_refine_relative_pose_inputs(scene1_state, scene2_state)

        start = time.time()

        pcd1_o3d = scene1_state.o3d_pcd
        pcd2_o3d = scene2_state.o3d_pcd

        assert len(pcd1_o3d.points) > 10, f'PCD 1 has {len(pcd1_o3d.points)} points which is <= 10'
        assert len(pcd2_o3d.points) > 10, f'PCD 2 has {len(pcd2_o3d.points)} points which is <= 10'

        if different_cameras_live_demo:
            # Move the demo cloud into the live camera frame so both clouds are
            # expressed in the same frame before registering. A no-op when the
            # two cameras share a pose.
            T_C2C1 = relative_camera_transform(scene1_state.T_WC, scene2_state.T_WC)
            pcd1_o3d.points = o3d.utility.Vector3dVector(
                transform_points(np.asarray(pcd1_o3d.points), T_C2C1))

        if self.icp_pose_estimator.error_metric == 'point-to-plane':
            pcd1_o3d.estimate_normals()
            pcd2_o3d.estimate_normals()
        elif self.icp_pose_estimator.error_metric == 'generalised-icp':
            pcd1_o3d.estimate_covariances()
            pcd2_o3d.estimate_covariances()

        if verbose:
            counter = 1
        best_fitness = None
        T_delta_best = None
        best_inlier_rmse = None
        best_num_correspondences = 0

        while time.time() - start < self.timeout:
            T_init = Open3dIcpPoseRefinement.sample_icp_initialisation(T_delta_init=T_delta_init,
                                                                       T_WC=T_WC_live,
                                                                       std_t=0.02,
                                                                       max_rot_angle=0)

            # draw_registration_result(pcd1_o3d, pcd2_o3d, T_init)

            T_delta, fitness, inlier_rmse, num_correspondences = self.icp_pose_estimator.estimate_relative_pose(
                o3d_source_pcd=pcd1_o3d,
                o3d_target_pcd=pcd2_o3d,
                T_init=T_init)

            # draw_registration_result(pcd1_o3d, pcd2_o3d, T_delta)

            if best_inlier_rmse is None:
                if fitness == 0:
                    best_fitness = fitness
                    best_inlier_rmse = np.inf
                    T_delta_best = T_delta
                    best_num_correspondences = 0
                else:
                    best_fitness = fitness
                    best_inlier_rmse = inlier_rmse
                    T_delta_best = T_delta
                    best_num_correspondences = num_correspondences
            elif inlier_rmse <= best_inlier_rmse and inlier_rmse != 0:
                best_fitness = fitness
                best_inlier_rmse = inlier_rmse
                T_delta_best = T_delta
                best_num_correspondences = num_correspondences

            if verbose:
                print(
                    f'ICP initialisation {counter:3d} - fitness: {fitness:.3f} / {best_fitness:.3f} - inlier RMSE: {inlier_rmse:.3f} / {best_inlier_rmse:.3f} - num correspondences: {num_correspondences} \ {best_num_correspondences}')
                counter += 1

        if visualise_pcds:
            draw_registration_result(source=pcd1_o3d, target=pcd2_o3d, transformation=T_delta_best)

        if best_fitness == 0:
            raise Exception(f'Opend3D ICP was unable to register the two point clouds')

        return T_delta_best

    @staticmethod
    def sample_icp_initialisation(T_delta_init, T_WC, std_t=0.01, max_rot_angle=2):

        # Map to world frame
        T_delta_init_W = T_WC @ T_delta_init @ pose_inv(T_WC)

        # Perturb rotation
        euler_xyz = rot2euler('xyz', T_delta_init_W[:3, :3], degrees=True)
        euler_xyz = np.asarray([0., 0., euler_xyz[2] + np.random.uniform(low=-max_rot_angle, high=max_rot_angle)])
        T_delta_init_W[:3, :3] = euler2rot('xyz', euler_xyz, degrees=True)

        # Perturb translation
        T_delta_init_W[:3, 3] += std_t * np.random.randn(3)

        # Map to camera frame
        T_delta_init_C = pose_inv(T_WC) @ T_delta_init_W @ T_WC

        return T_delta_init_C

# if __name__ == '__main__':
#     from os.path import join
#
#     from thousand_tasks.core.globals import ASSETS_DIR
#     from thousand_tasks.core.utils.sim2real_project import load_scene_from_msgpack, SCENE_SAVE_FORMAT, \
#         create_scene_states_from_data_msgpack
#     from thousand_tasks.perception.relative_pose_estimation.pnet_4dof_pose_regressor import PointnetPoseRegressor_4dof
#
#     # Loading a scene
#     scene_idx = 5
#     scene_data = load_scene_from_msgpack(join(ASSETS_DIR, 'datasets', 'test', SCENE_SAVE_FORMAT.format(scene_idx)))
#     T_WC = scene_data['T_WC_opencv'].copy()
#
#     scene_state_frame_0, scene_state_frame_1, T_delta = create_scene_states_from_data_msgpack(scene_data)
#
#     pose_estimator = PointnetPoseRegressor_4dof(filter_pointcloud=True,
#                                                 filter_outliers_o3d=False,
#                                                 n_points=2048,
#                                                 T_WC=scene_data['T_WC_opencv'],
#                                                 depth_units='mm')
#     pose_estimation_refiner = Open3dIcpPoseRefinement(error_metric='generalised-icp', timeout=2)
#
#     # o3d_pcd_0 = scene_state_frame_0.o3d_pcd
#     # o3d_pcd_1 = scene_state_frame_1.o3d_pcd
#
#     # # Apply orientation than point cloud centring
#     # T_delta_rot = T_delta.copy()
#     # T_delta_rot[:3, 3] = 0
#     # o3d_pcd_0.transform(T_delta_rot)
#     #
#     # # point cloud centring
#     # T_delta_trans = np.eye(4)
#     # T_delta_trans[:3, 3] = np.asarray(o3d_pcd_1.points).mean(axis=0) - np.asarray(o3d_pcd_0.points).mean(axis=0)
#     # o3d_pcd_0.transform(T_delta_trans)
#
#     T_delta_pred_W = pose_estimator.estimate_relative_pose(scene_state_frame_0,
#                                                            scene_state_frame_1,
#                                                            visualise_pcds=True,
#                                                            verbose=True)
#
#     T_delta_pred = pose_inv(T_WC) @ T_delta_pred_W @ T_WC
#
#     T_delta_pred_refined = pose_estimation_refiner.refine_relative_pose(scene_state_frame_0,
#                                                                         scene_state_frame_1,
#                                                                         T_WC=T_WC,
#                                                                         T_delta_init=T_delta_pred,
#                                                                         verbose=True,
#                                                                         visualise_pcds=True)
