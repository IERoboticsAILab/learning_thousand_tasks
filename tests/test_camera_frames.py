"""Tests for the demo -> live camera re-framing used when the two differ.

Runs under pytest, or standalone with `python3 tests/test_camera_frames.py`
(the deployment box has no pytest). numpy is the only dependency -- that is the
point of keeping camera_frames.py free of the numba/open3d stack.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from thousand_tasks.core.utils.camera_frames import (  # noqa: E402
    relative_camera_transform,
    rigid_inverse,
    transform_points,
)


def _pose(rpy=(0.0, 0.0, 0.0), t=(0.0, 0.0, 0.0)):
    """A 4x4 camera-to-world transform from roll/pitch/yaw and a translation."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = t
    return T


def test_identity_when_cameras_share_a_pose():
    """The always-on path must not disturb the demo cloud when demo == live."""
    T_WC = _pose(rpy=(0.1, -0.2, 0.3), t=(0.4, -0.5, 0.6))

    T_C2C1 = relative_camera_transform(T_WC, T_WC)

    assert np.allclose(T_C2C1, np.eye(4), atol=1e-12)

    points = np.array([[0.1, 0.2, 0.9], [-0.3, 0.05, 1.4], [0.0, 0.0, 0.7]])
    assert np.allclose(transform_points(points, T_C2C1), points, atol=1e-12)


def test_known_offset_maps_points_through_the_world_frame():
    """A point must land at the same world location whichever camera saw it."""
    T_WC_demo = _pose(rpy=(0.0, 0.0, 0.0), t=(1.0, 0.0, 0.5))
    T_WC_live = _pose(rpy=(0.0, 0.0, np.pi / 2), t=(0.0, 1.0, 0.5))

    point_in_demo_cam = np.array([[0.2, -0.1, 1.3]])

    # Ground truth: demo camera frame -> world -> live camera frame, done the
    # long way round, independently of the function under test.
    world = (T_WC_demo @ np.append(point_in_demo_cam[0], 1.0))[:3]
    expected = (rigid_inverse(T_WC_live) @ np.append(world, 1.0))[:3]

    T_C2C1 = relative_camera_transform(T_WC_demo, T_WC_live)
    actual = transform_points(point_in_demo_cam, T_C2C1)[0]

    assert np.allclose(actual, expected, atol=1e-12)


def test_pure_translation_offset_is_the_camera_displacement():
    """Sliding the camera +0.3m in x must shift points -0.3m in camera frame."""
    T_WC_demo = _pose(t=(0.0, 0.0, 0.0))
    T_WC_live = _pose(t=(0.3, 0.0, 0.0))

    points = np.array([[0.0, 0.0, 1.0], [0.5, -0.2, 2.0]])
    moved = transform_points(points, relative_camera_transform(T_WC_demo, T_WC_live))

    assert np.allclose(moved, points - np.array([0.3, 0.0, 0.0]), atol=1e-12)


def test_rigid_inverse_matches_a_numerical_inverse():
    T = _pose(rpy=(0.3, 0.7, -1.1), t=(0.2, -0.4, 2.5))

    assert np.allclose(rigid_inverse(T), np.linalg.inv(T), atol=1e-12)
    assert np.allclose(rigid_inverse(T) @ T, np.eye(4), atol=1e-12)


def test_transform_points_preserves_shape_and_distances():
    """A rigid transform must not deform the cloud it is applied to."""
    T_C2C1 = relative_camera_transform(
        _pose(rpy=(0.2, 0.0, 0.4), t=(0.1, 0.2, 0.3)),
        _pose(rpy=(-0.3, 0.5, 0.0), t=(-0.2, 0.4, 0.1)),
    )
    points = np.random.default_rng(0).normal(size=(50, 3))

    moved = transform_points(points, T_C2C1)

    assert moved.shape == points.shape

    def pairwise(p):
        return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

    assert np.allclose(pairwise(moved), pairwise(points), atol=1e-10)


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS  {}'.format(name))
            except AssertionError as exc:
                failures += 1
                print('FAIL  {}: {}'.format(name, exc))
    print('\n{} passed, {} failed'.format(
        len([n for n in globals() if n.startswith('test_')]) - failures, failures))
    sys.exit(1 if failures else 0)
