"""Re-expressing a point cloud captured by one camera in another camera's frame.

Used when a demonstration and the live scene were recorded from different camera
poses: the demo cloud is rigidly moved into the live camera frame so the two can
be registered against each other.

Deliberately numpy-only. The equivalent rigid inverse in `se3_tools.pose_inv`
pulls in numba and scipy, which drags the whole CUDA/open3d stack into anything
that wants to check this geometry; keeping these few lines self-contained makes
them testable on their own.
"""

import numpy as np


def rigid_inverse(T):
    """Inverse of a 4x4 rigid transform, via transpose rather than a solve."""
    R = T[:3, :3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ np.ascontiguousarray(T[:3, 3])
    return T_inv


def relative_camera_transform(T_WC_from, T_WC_to):
    """Map points in the `from` camera frame into the `to` camera frame.

    Both arguments are camera-to-world transforms, so the composition goes
    `from` camera -> world -> `to` camera. Returns identity when the two
    cameras share a pose.
    """
    return rigid_inverse(T_WC_to) @ T_WC_from


def transform_points(points, T):
    """Apply a 4x4 rigid transform to an (N, 3) array of points."""
    points = np.asarray(points)
    points_h = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    return (points_h @ T.T)[:, :3]
