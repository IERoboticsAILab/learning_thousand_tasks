"""Tests for the depth-outlier filter applied to a segmented object's points.

Runs under pytest, or standalone with `python3 tests/test_depth_filter.py`
(the deployment box has no pytest). numpy is the only dependency: the module is
loaded by file path, because importing it through the package pulls in torch.
"""

import importlib.util
import os

import numpy as np

_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'thousand_tasks', 'perception',
    'relative_pose_estimation', 'pose_estimators', 'direct', 'preprocessor.py')
_spec = importlib.util.spec_from_file_location('preprocessor', _PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
Preprocessor = _module.Preprocessor

SHAPE = (72, 128)


def _scene(n_points, far=()):
    """A mask of `n_points` pixels at 1 m, with the listed indices pushed to 3 m."""
    seg = np.zeros(SHAPE[0] * SHAPE[1], dtype=bool)
    seg[:n_points] = True
    depth = np.zeros(seg.shape, dtype=np.float32)
    depth[:n_points] = 1.0
    depth[list(far)] = 3.0
    return depth.reshape(SHAPE), seg.reshape(SHAPE)


def test_small_mask_does_not_raise():
    # 427 is the real case: a 571 px handle mask after deploy_mt3's erosion.
    # argpartition(kth=500) raised "kth(=500) out of bounds (427)" on it.
    for n_points in (1, 2, 427, 500):
        depth, seg = _scene(n_points)
        keep = Preprocessor().get_filtered_depth_ids(depth, seg)
        assert keep.shape == (n_points,)
        assert keep.all()


def test_small_mask_still_drops_a_far_outlier():
    depth, seg = _scene(427, far=(3,))
    keep = Preprocessor().get_filtered_depth_ids(depth, seg)
    assert not keep[3]
    assert keep.sum() == 426


def test_large_mask_is_filtered_as_before():
    # 5000 points at 1 m with five at 3 m: the 500 largest are 495 near points
    # and the five outliers, whose mean + 4 std sits below 3 m.
    depth, seg = _scene(5000, far=(10, 20, 30, 40, 50))
    keep = Preprocessor().get_filtered_depth_ids(depth, seg)
    assert keep.sum() == 4995
    assert not keep[[10, 20, 30, 40, 50]].any()


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print(f'  ok  {test.__name__}')
    print(f'\n{len(tests)} passed')
