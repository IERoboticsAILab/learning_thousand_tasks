"""Brute-force evaluation harness for the GICP refinement stopping rule.

The deployed refiner (Open3dIcpPoseRefinement.refine_relative_pose) burns a
fixed `timeout=3` seconds in a random-restart loop even though the winning
restart is typically found within the first few usable trials. This harness
measures what each cheaper stopping rule would have cost in *final live
bottleneck pose* accuracy, so the fastest safe setting can be picked.

Architecture: record once, replay many.

  record  (GPU box only)  For every ordered (live, demo) pair of same-family
          demos in the demo store, build both scene states exactly as
          deploy_mt3._run_pipeline does, get the ICP initialisation from
          PointNet++ (or identity with --init identity), then run the
          random-restart GICP loop for a full --ceiling seconds (today's 3 s
          budget), recording EVERY trial: timing, fitness, inlier_rmse,
          n_corr and the full 4x4 result. One .npz trace per pair.

  replay  (pure numpy, runs anywhere)  Scan each recorded trial sequence with
          the refiner's exact selection rule, stop where each candidate rule
          fires (time budget / restart count / usable count / patience /
          first-usable), propagate the winning T to the live bottleneck pose,
          and score it (a) against the full-ceiling baseline pose and
          (b) against ground truth (the live folder is itself a demo, so its
          own bottleneck_pose.npy is the true answer). Prints an aggregate
          table + recommendation and dumps a full CSV.

This file lives in deployment/, which is bind-mounted into the once-mt3
container at /workspace/deployment, so on kharon it is runnable immediately
after checkout/copy -- no image rebuild. Exact commands:

  # record traces (expensive, needs the GPU box's demo store):
  docker exec once-mt3 sh -lc "cd /workspace && MPLBACKEND=Agg \
      PYTHONPATH=/workspace python -u deployment/icp_sweep.py record"

  # replay stopping rules over the saved traces (cheap, numpy only):
  docker exec once-mt3 sh -lc "cd /workspace && MPLBACKEND=Agg \
      PYTHONPATH=/workspace python -u deployment/icp_sweep.py replay"

  # or both in one go:
  docker exec once-mt3 sh -lc "cd /workspace && MPLBACKEND=Agg \
      PYTHONPATH=/workspace python -u deployment/icp_sweep.py all"

Traces land in --out (default saved_data/icp_sweep/, i.e.
/workspace/saved_data/icp_sweep in the container, which is bind-mounted), so
`replay` can also be re-run on any machine with numpy after copying that
directory over.

Self-check of the replay/rule logic (numpy only, runs on a laptop):

  python3 deployment/icp_sweep.py --self-test
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# Files a demo folder must contain to be usable as either side of a pair.
REQUIRED_FILES = (
    'head_camera_ws_rgb.png',
    'head_camera_ws_depth_to_rgb.png',
    'head_camera_ws_segmap.npy',
    'head_camera_rgb_intrinsic_matrix.npy',
    'T_WC.npy',
    'bottleneck_pose.npy',
)

DEFAULT_TIME_GRID = '0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0'
DEFAULT_COUNT_GRID = '5,10,25,50,100,200,400,800'
DEFAULT_USABLE_GRID = '1,2,3,5,10,20'
DEFAULT_PATIENCE_GRID = '1,2,3,5,10'


# ---------------------------------------------------------------------------
# numpy-only SE(3) helpers (replay must not import scipy/numba/open3d)
# ---------------------------------------------------------------------------

def rigid_inverse(T):
    """Inverse of a 4x4 rigid transform."""
    R = T[:3, :3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ np.ascontiguousarray(T[:3, 3])
    return T_inv


def pose_error(T1, T2):
    """numpy replica of se3_tools.calculate_pose_error (se3_tools.py:994).

    Returns (position_error_m, orientation_error_deg). The rotvec norm used
    there equals the rotation angle of the relative rotation.
    """
    T_delta = T1 @ rigid_inverse(T2)
    position_error = float(np.linalg.norm(T_delta[:3, 3]))
    cos_angle = (np.trace(T_delta[:3, :3]) - 1.0) / 2.0
    orientation_error = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    return position_error, orientation_error


def live_pose_from_C_T_delta(C_T_delta, T_WC_live, demo_bottleneck_pose):
    """deploy_mt3._run_pipeline's world-frame propagation of the refined pose."""
    W_T_delta = T_WC_live @ C_T_delta @ rigid_inverse(T_WC_live)
    return W_T_delta @ demo_bottleneck_pose


# ---------------------------------------------------------------------------
# the refiner's exact selection rule + candidate stopping rules (numpy only)
# ---------------------------------------------------------------------------

def select_best(fitness, rmse, n_stop):
    """Replay Open3dIcpPoseRefinement.refine_relative_pose's selection over the
    first n_stop trials. Returns (best_index, failed). The first trial always
    seeds best (rmse treated as inf when fitness==0); afterwards a trial
    replaces it iff `rmse <= best and rmse != 0` (note <=: later ties win).
    failed=True mirrors the deployed `if best_fitness == 0: raise`.
    """
    best_i = 0
    best_rmse = np.inf if fitness[0] == 0 else rmse[0]
    for i in range(1, n_stop):
        if rmse[i] != 0 and rmse[i] <= best_rmse:
            best_i = i
            best_rmse = rmse[i]
    return best_i, fitness[best_i] == 0


def stop_time(t_rel, ms, budget):
    """Trials the deployed loop would run with `timeout=budget`: a trial starts
    iff the elapsed time before it is < budget. t_rel is elapsed-at-completion,
    so a trial's start is t_rel - ms/1000."""
    n = len(t_rel)
    for i in range(n):
        if t_rel[i] - ms[i] / 1000.0 >= budget:
            return max(i, 1)
    return n


def stop_count(n_trials, n):
    return max(1, min(n, n_trials))


def stop_usable(rmse, k):
    """Stop once k trials with rmse != 0 have been seen."""
    seen = 0
    for i in range(len(rmse)):
        if rmse[i] != 0:
            seen += 1
            if seen >= k:
                return i + 1
    return len(rmse)


def stop_patience(rmse, k, eps):
    """Stop after k consecutive usable trials without a relative improvement of
    the best usable rmse by more than eps."""
    best = None
    since = 0
    for i in range(len(rmse)):
        if rmse[i] != 0:
            if best is None or rmse[i] < best * (1.0 - eps):
                best = rmse[i]
                since = 0
            else:
                since += 1
            if since >= k:
                return i + 1
    return len(rmse)


def stop_first_usable(rmse):
    return stop_usable(rmse, 1)


# ---------------------------------------------------------------------------
# pair enumeration
# ---------------------------------------------------------------------------

def family_of(name):
    """Strip a trailing _NNN index: pick_up_mug_003 -> pick_up_mug."""
    return re.sub(r'_\d+$', '', name)


def enumerate_pairs(demo_root, family_filter=None, max_pairs=None):
    """All ordered (live_dir, demo_dir) pairs of distinct demos within each
    task family that has >= 2 complete demo folders."""
    demo_root = Path(demo_root)
    families = {}
    for d in sorted(p for p in demo_root.iterdir() if p.is_dir()):
        if all((d / f).exists() for f in REQUIRED_FILES):
            families.setdefault(family_of(d.name), []).append(d)
    pairs = []
    for fam in sorted(families):
        if family_filter and fam != family_filter:
            continue
        members = families[fam]
        if len(members) < 2:
            continue
        for live in members:
            for demo in members:
                if live is not demo:
                    pairs.append((live, demo))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


# ---------------------------------------------------------------------------
# recording pass (GPU box; imports open3d/torch/PIL lazily)
# ---------------------------------------------------------------------------

def load_scene(d):
    """The deployment scene-state recipe. Returns (scene_state, T_WC)."""
    from PIL import Image
    from thousand_tasks.core.utils.scene_state import SceneState
    d = Path(d)
    T_WC = np.load(str(d / 'T_WC.npy'))
    s = SceneState.initialise_from_dict({
        'rgb': np.array(Image.open(str(d / 'head_camera_ws_rgb.png'))),
        'depth': np.array(Image.open(str(d / 'head_camera_ws_depth_to_rgb.png'))),
        'segmap': np.load(str(d / 'head_camera_ws_segmap.npy')),
        'intrinsic_matrix': np.load(str(d / 'head_camera_rgb_intrinsic_matrix.npy')),
    })
    s.T_WC = T_WC
    s.erode_segmap()
    s.crop_object_using_segmap()
    return s, T_WC


def build_pose_estimator():
    """PointNet++ regressor, constructed as deploy_mt3.build_context does.
    Extrinsics are rebound per pair, so the constructor values are dummies."""
    import torch
    from thousand_tasks.perception.pose_estimation.pnet_4dof_pose_regressor import (
        PointnetPoseRegressor_4dof,
    )
    return PointnetPoseRegressor_4dof(
        filter_pointcloud=True,
        n_points=2048,
        T_WC=np.eye(4),
        T_WC_demo=np.eye(4),
        depth_units='mm',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )


def record_pair(live_dir, demo_dir, pose_estimator, args):
    """Run the restart loop for a full --ceiling seconds and return the trace."""
    import open3d as o3d
    from thousand_tasks.core.utils.se3_tools import pose_inv
    from thousand_tasks.perception.pose_estimation.icp_6dof_pose_estimation_refinement import (
        Open3dIcpPoseRefinement,
    )
    from thousand_tasks.perception.relative_pose_estimation.utils.open3d_icp import Open3dICP

    live_scene, T_WC_live = load_scene(live_dir)
    demo_scene, T_WC_demo = load_scene(demo_dir)

    # ICP initialisation, matching deployment reality.
    if pose_estimator is not None:
        pose_estimator.extrinsics = T_WC_live
        pose_estimator.extrinsics_demo = T_WC_demo
        W_T_delta = pose_estimator.estimate_relative_pose(
            scene1_state=demo_scene, scene2_state=live_scene,
            visualise_pcds=False, verbose=False)
        C_T_delta_init = pose_inv(T_WC_live) @ W_T_delta @ T_WC_live
    else:
        C_T_delta_init = np.eye(4)

    # Clouds, built once (o3d_pcd is an uncached property).
    pcd_demo = demo_scene.o3d_pcd
    pcd_live = live_scene.o3d_pcd
    assert len(pcd_demo.points) > 10, 'demo cloud has <= 10 points'
    assert len(pcd_live.points) > 10, 'live cloud has <= 10 points'

    # Re-express the demo cloud in the live camera frame (the refiner's
    # different_cameras_live_demo=True branch).
    T_C2C1 = pose_inv(T_WC_live) @ T_WC_demo
    pts = np.asarray(pcd_demo.points)
    pts_h = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    pcd_demo.points = o3d.utility.Vector3dVector((pts_h @ T_C2C1.T)[:, :3])

    pcd_demo.estimate_covariances()
    pcd_live.estimate_covariances()

    icp = Open3dICP(error_metric='generalised-icp',
                    max_correspondence_distance=0.1,
                    max_iteration=20)

    np.random.seed(args.seed)
    t_rel, ms, fitness, rmse, n_corr, Ts = [], [], [], [], [], []
    start = time.time()
    while time.time() - start < args.ceiling:
        it0 = time.time()
        T_init = Open3dIcpPoseRefinement.sample_icp_initialisation(
            T_delta_init=C_T_delta_init, T_WC=T_WC_live,
            std_t=args.std_t, max_rot_angle=args.max_rot_angle)
        T, fit, inlier_rmse, corr = icp.estimate_relative_pose(
            o3d_source_pcd=pcd_demo, o3d_target_pcd=pcd_live, T_init=T_init)
        now = time.time()
        t_rel.append(now - start)
        ms.append((now - it0) * 1e3)
        fitness.append(float(fit))
        rmse.append(float(inlier_rmse))
        n_corr.append(int(corr))
        Ts.append(np.asarray(T, dtype=np.float64))

    meta = {
        'live': Path(live_dir).name,
        'demo': Path(demo_dir).name,
        'init': args.init,
        'std_t': args.std_t,
        'max_rot_angle': args.max_rot_angle,
        'seed': args.seed,
        'ceiling': args.ceiling,
    }
    return {
        't_rel': np.asarray(t_rel),
        'ms': np.asarray(ms),
        'fitness': np.asarray(fitness),
        'inlier_rmse': np.asarray(rmse),
        'n_corr': np.asarray(n_corr, dtype=np.int64),
        'T': np.stack(Ts),
        'T_WC_live': T_WC_live,
        'T_WC_demo': T_WC_demo,
        'C_T_delta_init': C_T_delta_init,
        'demo_bottleneck_pose': np.load(str(Path(demo_dir) / 'bottleneck_pose.npy')),
        'live_bottleneck_pose_gt': np.load(str(Path(live_dir) / 'bottleneck_pose.npy')),
        'meta': np.array(json.dumps(meta)),
    }


def cmd_record(args):
    demo_root = args.demo_root
    if demo_root is None:
        from thousand_tasks.core.globals import ASSETS_DIR
        demo_root = ASSETS_DIR / 'demonstrations'
    pairs = enumerate_pairs(demo_root, args.family, args.pairs)
    if not pairs:
        print('No (live, demo) pairs found under {} -- need task families with '
              '>= 2 complete demo folders (incl. T_WC.npy).'.format(demo_root))
        sys.exit(1)
    print('{} ordered pairs to record (ceiling {:.1f} s each)'.format(
        len(pairs), args.ceiling))

    pose_estimator = build_pose_estimator() if args.init == 'pointnet' else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (live_dir, demo_dir) in enumerate(pairs):
        name = '{}__{}.npz'.format(live_dir.name, demo_dir.name)
        print('[{}/{}] {} <- {}'.format(i + 1, len(pairs), live_dir.name,
                                        demo_dir.name), flush=True)
        trace = record_pair(live_dir, demo_dir, pose_estimator, args)
        np.savez_compressed(str(out_dir / name), **trace)
        n = len(trace['t_rel'])
        usable = int(np.count_nonzero(trace['inlier_rmse']))
        print('    {} trials, {} usable ({:.1f}%), {:.1f} ms/trial'.format(
            n, usable, 100.0 * usable / max(n, 1),
            float(np.mean(trace['ms'])) if n else 0.0))
    print('Traces saved to {}'.format(out_dir))


# ---------------------------------------------------------------------------
# replay pass (numpy only)
# ---------------------------------------------------------------------------

def load_traces(out_dir):
    traces = []
    for p in sorted(Path(out_dir).glob('*.npz')):
        z = np.load(str(p), allow_pickle=False)
        tr = {k: z[k] for k in z.files}
        tr['meta'] = json.loads(str(tr['meta']))
        traces.append(tr)
    return traces


def build_rules(args):
    """List of (rule_name, param, fn(trace) -> n_stop)."""
    rules = []
    for t in [float(x) for x in args.time_grid.split(',') if x]:
        rules.append(('time', t,
                      lambda tr, t=t: stop_time(tr['t_rel'], tr['ms'], t)))
    for n in [int(x) for x in args.count_grid.split(',') if x]:
        rules.append(('count', n,
                      lambda tr, n=n: stop_count(len(tr['t_rel']), n)))
    for k in [int(x) for x in args.usable_grid.split(',') if x]:
        rules.append(('usable', k,
                      lambda tr, k=k: stop_usable(tr['inlier_rmse'], k)))
    for k in [int(x) for x in args.patience_grid.split(',') if x]:
        rules.append(('patience', k,
                      lambda tr, k=k: stop_patience(tr['inlier_rmse'], k,
                                                    args.patience_eps)))
    rules.append(('first-usable', 1,
                  lambda tr: stop_first_usable(tr['inlier_rmse'])))
    return rules


def evaluate_trace(trace, n_stop):
    """(live_bottleneck_pose, elapsed_s, failed) at a given stopping point."""
    fitness = trace['fitness']
    rmse = trace['inlier_rmse']
    best_i, failed = select_best(fitness, rmse, n_stop)
    pose = live_pose_from_C_T_delta(trace['T'][best_i], trace['T_WC_live'],
                                    trace['demo_bottleneck_pose'])
    return pose, float(trace['t_rel'][n_stop - 1]), failed


def cmd_replay(args):
    traces = load_traces(args.out)
    if not traces:
        print('No traces found in {} -- run `record` first.'.format(args.out))
        sys.exit(1)
    rules = build_rules(args)
    print('{} traces, {} rule settings'.format(len(traces), len(rules)))

    # Full-ceiling baseline per pair (+ its absolute error vs ground truth, so
    # it is visible whether the 3 s budget even helps absolute accuracy).
    baselines = []
    print()
    print('Baseline (full ceiling) absolute error vs ground truth:')
    print('  {:<24} {:<24} {:>7} {:>9} {:>10} {:>9}'.format(
        'live', 'demo', 'trials', 'elapsed', 'pos_mm', 'rot_deg'))
    for tr in traces:
        n_all = len(tr['t_rel'])
        pose, elapsed, failed = evaluate_trace(tr, n_all)
        gt = tr['live_bottleneck_pose_gt']
        p, r = pose_error(pose, gt)
        baselines.append({'pose': pose, 'elapsed': elapsed, 'failed': failed,
                          'gt_pos_mm': p * 1e3, 'gt_rot_deg': r})
        print('  {:<24} {:<24} {:>7} {:>8.2f}s {:>10.2f} {:>9.2f}{}'.format(
            tr['meta']['live'], tr['meta']['demo'], n_all, elapsed,
            p * 1e3, r, '  [FAILED]' if failed else ''))

    # Sweep the rules.
    rows = []
    agg = {}
    for rule_name, param, fn in rules:
        key = (rule_name, param)
        agg[key] = {'dev_pos': [], 'dev_rot': [], 'gt_pos': [], 'gt_rot': [],
                    'elapsed': [], 'saved': [], 'n_fail': 0}
        for tr, base in zip(traces, baselines):
            n_stop = fn(tr)
            pose, elapsed, failed = evaluate_trace(tr, n_stop)
            saved = base['elapsed'] - elapsed
            if failed:
                agg[key]['n_fail'] += 1
                rows.append([rule_name, param, tr['meta']['live'],
                             tr['meta']['demo'], n_stop, elapsed, saved,
                             '', '', '', '', 1])
                continue
            dp, dr = pose_error(pose, base['pose'])
            gp, gr = pose_error(pose, tr['live_bottleneck_pose_gt'])
            a = agg[key]
            a['dev_pos'].append(dp * 1e3)
            a['dev_rot'].append(dr)
            a['gt_pos'].append(gp * 1e3)
            a['gt_rot'].append(gr)
            a['elapsed'].append(elapsed)
            a['saved'].append(saved)
            rows.append([rule_name, param, tr['meta']['live'],
                         tr['meta']['demo'], n_stop, round(elapsed, 4),
                         round(saved, 4), round(dp * 1e3, 4), round(dr, 4),
                         round(gp * 1e3, 4), round(gr, 4), 0])

    # Report table.
    print()
    print('Per-rule aggregate over {} pairs (errors: median/max deviation vs '
          'full-ceiling baseline; abs = vs ground truth):'.format(len(traces)))
    hdr = ('rule', 'param', 'dev_pos_mm md/max', 'dev_rot_deg md/max',
           'abs_pos_mm md', 'abs_rot_deg md', 'elapsed md', 'saved md', 'fail')
    print('  {:<13} {:>7} {:>18} {:>19} {:>13} {:>14} {:>10} {:>9} {:>5}'.format(*hdr))
    for (rule_name, param), a in agg.items():
        if a['dev_pos']:
            line = ('  {:<13} {:>7} {:>8.2f}/{:>8.2f} {:>9.2f}/{:>8.2f} '
                    '{:>13.2f} {:>14.2f} {:>9.2f}s {:>8.2f}s {:>5}').format(
                rule_name, param,
                float(np.median(a['dev_pos'])), float(np.max(a['dev_pos'])),
                float(np.median(a['dev_rot'])), float(np.max(a['dev_rot'])),
                float(np.median(a['gt_pos'])), float(np.median(a['gt_rot'])),
                float(np.median(a['elapsed'])), float(np.median(a['saved'])),
                a['n_fail'])
        else:
            line = '  {:<13} {:>7} all pairs failed ({} fail)'.format(
                rule_name, param, a['n_fail'])
        print(line)

    # CSV dump.
    out_dir = Path(args.out)
    csv_path = out_dir / 'icp_sweep_results.csv'
    with open(str(csv_path), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rule', 'param', 'live', 'demo', 'n_stop', 'elapsed_s',
                    'time_saved_s', 'dev_pos_mm', 'dev_rot_deg',
                    'abs_pos_mm', 'abs_rot_deg', 'failed'])
        w.writerows(rows)
    print()
    print('Full per-pair results: {}'.format(csv_path))

    # Recommendation: fastest rule with no failures whose WORST-case deviation
    # from the full-ceiling baseline stays inside the tolerances.
    candidates = []
    for key, a in agg.items():
        if a['n_fail'] == 0 and a['dev_pos'] and \
                float(np.max(a['dev_pos'])) < args.tol_mm and \
                float(np.max(a['dev_rot'])) < args.tol_deg:
            candidates.append((float(np.median(a['elapsed'])), key, a))
    print()
    if candidates:
        candidates.sort(key=lambda c: c[0])
        med_el, (rule_name, param), a = candidates[0]
        print('RECOMMENDATION: rule "{}" param {} -- median {:.2f} s '
              '(saves {:.2f} s vs full ceiling), worst-case deviation '
              '{:.2f} mm / {:.2f} deg (tolerances {} mm / {} deg), 0 failures.'
              .format(rule_name, param, med_el, float(np.median(a['saved'])),
                      float(np.max(a['dev_pos'])), float(np.max(a['dev_rot'])),
                      args.tol_mm, args.tol_deg))
    else:
        print('RECOMMENDATION: no rule met the tolerances ({} mm / {} deg) '
              'with zero failures -- keep the full timeout or loosen '
              '--tol-mm/--tol-deg.'.format(args.tol_mm, args.tol_deg))


# ---------------------------------------------------------------------------
# self-test (numpy only)
# ---------------------------------------------------------------------------

def run_self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print('  {:<58} {}'.format(name, 'ok' if cond else 'FAIL'))
        ok = ok and cond

    print('icp_sweep self-test')

    # --- selection rule ---------------------------------------------------
    fitness = np.array([0.0, 0.9, 0.8, 0.0, 0.7])
    rmse = np.array([0.0, 0.5, 0.5, 0.0, 0.4])
    check('select_best: failed-first-trial seeds inf, flagged failed',
          select_best(fitness, rmse, 1) == (0, True))
    check('select_best: first usable replaces inf seed',
          select_best(fitness, rmse, 2) == (1, False))
    check('select_best: equal rmse -> later tie wins (<=)',
          select_best(fitness, rmse, 3) == (2, False))
    check('select_best: rmse==0 trials are never selected',
          select_best(fitness, rmse, 4) == (2, False))
    check('select_best: lower rmse wins over full trace',
          select_best(fitness, rmse, 5) == (4, False))

    # --- stopping rules ---------------------------------------------------
    t_rel = np.array([0.010, 0.020, 0.030, 0.040, 0.050])
    ms = np.full(5, 10.0)  # trial starts at t_rel - 10 ms: 0, 10, 20, 30, 40 ms
    check('stop_time: budget 15 ms admits starts at 0 and 10 ms',
          stop_time(t_rel, ms, 0.015) == 2)
    check('stop_time: budget covering all starts runs everything',
          stop_time(t_rel, ms, 1.0) == 5)
    check('stop_time: tiny budget still runs at least one trial',
          stop_time(t_rel, ms, 1e-9) == 1)
    check('stop_count clamps to trace length',
          stop_count(5, 3) == 3 and stop_count(5, 99) == 5 and stop_count(5, 0) == 1)
    check('stop_usable: k=2 consumes through 2nd usable trial',
          stop_usable(rmse, 2) == 3)
    check('stop_usable: k beyond usable supply consumes everything',
          stop_usable(rmse, 4) == 5)
    check('first-usable stops at trial 2',
          stop_first_usable(rmse) == 2)
    check('stop_patience: k=1 eps=0 stops on the tied (non-improving) trial',
          stop_patience(rmse, 1, 0.0) == 3)
    check('stop_patience: k=2 waits for a 2nd non-improvement',
          stop_patience(np.array([0.5, 0.5, 0.5, 0.3]), 2, 0.0) == 3)
    check('stop_patience: never fires -> consumes everything',
          stop_patience(np.array([0.5, 0.4, 0.3]), 1, 0.0) == 3)

    # --- pose propagation round-trip -------------------------------------
    ang = np.radians(30.0)
    T_WC = np.eye(4)
    T_WC[:3, :3] = np.array([[np.cos(ang), -np.sin(ang), 0],
                             [np.sin(ang), np.cos(ang), 0],
                             [0, 0, 1]])
    T_WC[:3, 3] = [0.3, -0.2, 0.9]
    bottleneck = np.eye(4)
    bottleneck[:3, 3] = [0.5, 0.1, 0.2]
    check('identity C_T_delta -> live pose == demo bottleneck',
          np.allclose(live_pose_from_C_T_delta(np.eye(4), T_WC, bottleneck),
                      bottleneck))
    check('rigid_inverse round-trips', np.allclose(rigid_inverse(T_WC) @ T_WC,
                                                   np.eye(4)))
    # A world-frame delta pushed through the camera frame must come back out.
    W_T_delta = np.eye(4)
    W_T_delta[:3, 3] = [0.05, 0.0, -0.02]
    C_T_delta = rigid_inverse(T_WC) @ W_T_delta @ T_WC
    check('camera-frame delta propagates to W_T_delta @ bottleneck',
          np.allclose(live_pose_from_C_T_delta(C_T_delta, T_WC, bottleneck),
                      W_T_delta @ bottleneck))

    # --- pose error metric ------------------------------------------------
    p, r = pose_error(bottleneck, bottleneck)
    check('pose_error of identical poses is (0, 0)', p < 1e-12 and r < 1e-6)
    T2 = bottleneck.copy()
    T2[:3, 3] += [0.003, 0.004, 0.0]
    p, r = pose_error(T2, bottleneck)
    check('pose_error translation: 3-4-0 mm offset -> 5 mm',
          abs(p - 0.005) < 1e-12 and r < 1e-6)
    T3 = bottleneck.copy()
    T3[:3, :3] = T_WC[:3, :3]  # 30 deg about z
    p, r = pose_error(T3, bottleneck)
    check('pose_error rotation: 30 deg about z -> 30 deg', abs(r - 30.0) < 1e-9)

    # --- end-to-end replay over a synthetic trace -------------------------
    n = 6
    Ts = np.stack([np.eye(4) for _ in range(n)])
    for i in range(n):
        Ts[i][0, 3] = 0.01 * i  # each trial's T is distinguishable
    trace = {
        't_rel': np.arange(1, n + 1) * 0.01,
        'ms': np.full(n, 10.0),
        'fitness': np.array([0.0, 0.9, 0.9, 0.8, 0.9, 0.9]),
        'inlier_rmse': np.array([0.0, 0.5, 0.0, 0.4, 0.6, 0.3]),
        'T': Ts,
        'T_WC_live': T_WC,
        'demo_bottleneck_pose': bottleneck,
        'live_bottleneck_pose_gt': bottleneck,
    }
    pose, elapsed, failed = evaluate_trace(trace, n)  # winner: trial 5
    exp = live_pose_from_C_T_delta(Ts[5], T_WC, bottleneck)
    check('evaluate_trace picks the full-trace winner (trial 5)',
          np.allclose(pose, exp) and not failed and abs(elapsed - 0.06) < 1e-12)
    pose, elapsed, failed = evaluate_trace(trace, stop_usable(trace['inlier_rmse'], 2))
    exp = live_pose_from_C_T_delta(Ts[3], T_WC, bottleneck)
    check('usable-k=2 replay stops at trial 3 and returns its pose',
          np.allclose(pose, exp) and not failed)
    pose, elapsed, failed = evaluate_trace(trace, 1)
    check('stopping on a fitness==0 seed is reported as failed', failed)

    print()
    print('self-test: {}'.format('ALL PASSED' if ok else 'FAILURES'))
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd')

    def record_args(sp):
        sp.add_argument('--demo-root', default=None,
                        help='demo store (default: <ASSETS_DIR>/demonstrations)')
        sp.add_argument('--family', default=None,
                        help='only pairs from this task family')
        sp.add_argument('--pairs', type=int, default=None,
                        help='limit the number of pairs recorded')
        sp.add_argument('--ceiling', type=float, default=3.0,
                        help='seconds of restarts recorded per pair '
                             '(default: %(default)s, today\'s budget)')
        sp.add_argument('--init', choices=('pointnet', 'identity'),
                        default='pointnet',
                        help='ICP initialisation source (default: %(default)s)')
        sp.add_argument('--std-t', type=float, default=0.02,
                        help='translation jitter std in m (default: %(default)s)')
        sp.add_argument('--max-rot-angle', type=float, default=0.0,
                        help='max yaw jitter in deg (default: %(default)s)')
        sp.add_argument('--seed', type=int, default=0,
                        help='numpy RNG seed per pair (default: %(default)s)')

    def replay_args(sp):
        sp.add_argument('--time-grid', default=DEFAULT_TIME_GRID)
        sp.add_argument('--count-grid', default=DEFAULT_COUNT_GRID)
        sp.add_argument('--usable-grid', default=DEFAULT_USABLE_GRID)
        sp.add_argument('--patience-grid', default=DEFAULT_PATIENCE_GRID)
        sp.add_argument('--patience-eps', type=float, default=0.01,
                        help='relative rmse improvement threshold '
                             '(default: %(default)s)')
        sp.add_argument('--tol-mm', type=float, default=2.0,
                        help='max tolerated worst-case position deviation vs '
                             'baseline (default: %(default)s)')
        sp.add_argument('--tol-deg', type=float, default=1.0,
                        help='max tolerated worst-case rotation deviation vs '
                             'baseline (default: %(default)s)')

    for name, help_text, arg_fns in (
            ('record', 'run + record ICP restarts (GPU box)', (record_args,)),
            ('replay', 'sweep stopping rules over traces', (replay_args,)),
            ('all', 'record then replay', (record_args, replay_args))):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument('--out', default='saved_data/icp_sweep',
                        help='trace/result directory (default: %(default)s)')
        for fn in arg_fns:
            fn(sp)
    return p


def main():
    if '--self-test' in sys.argv:
        run_self_test()
    args = build_parser().parse_args()
    if args.cmd == 'record':
        cmd_record(args)
    elif args.cmd == 'replay':
        cmd_replay(args)
    elif args.cmd == 'all':
        cmd_record(args)
        cmd_replay(args)
    else:
        build_parser().print_help()
        sys.exit(2)


if __name__ == '__main__':
    main()
