import logging
from typing import TypedDict

import numpy as np


class SceneData(TypedDict):
    image_0: np.ndarray
    image_1: np.ndarray
    depth_0: np.ndarray
    depth_1: np.ndarray
    seg_0: np.ndarray
    seg_1: np.ndarray
    intrinsics_0: np.ndarray
    intrinsics_1: np.ndarray
    T_WC_0: np.ndarray
    T_WC_1: np.ndarray


class ProcessedData(TypedDict):
    pointcloud_0: np.ndarray
    pointcloud_1: np.ndarray


def pose_inv(pose):
    R = pose[:3, :3]
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = - R.T @ np.ascontiguousarray(pose[:3, 3])
    return T


class Preprocessor():

    def __init__(self,
                 n_points: int = 2048,
                 filter_pointcloud: bool = True,
                 filter_outliers_o3d: bool = False,
                 debug: bool = False) -> None:

        self.debug_mode = debug

        self.n_points = n_points
        self.filter_pointcloud = filter_pointcloud
        self.filter_outliers_o3d = filter_outliers_o3d

        if filter_pointcloud and filter_outliers_o3d:
            logging.warning("[WARNING] You are both filtering the pointcloud with heuristics and with Open3D")
        elif filter_outliers_o3d:
            raise NotImplementedError("[Open3D] Pointcloud outlier filtering is not implemented")
            logging.warning("[WARNING] The Open3D parameters for filtering outliers might have to be tuned")

    def _initial_processing(self, data: SceneData):
        data["seg_0"] = data["seg_0"].astype(bool)
        data["seg_1"] = data["seg_1"].astype(bool)

        data["depth_0"] = data["depth_0"] / 1000
        data["depth_1"] = data["depth_1"] / 1000
        data["depth_0"] = data["depth_0"] * (data["depth_0"] < 5.0) * data["seg_0"]
        data["depth_1"] = data["depth_1"] * (data["depth_1"] < 5.0) * data["seg_1"]

        data["image_0"] = data["image_0"].astype(np.float32)
        data["image_1"] = data["image_1"].astype(np.float32)
        data["depth_0"] = data["depth_0"].astype(np.float32)
        data["depth_1"] = data["depth_1"].astype(np.float32)

    def _get_segmented_keypoints(self, seg: np.ndarray):
        x = np.arange(seg.shape[1], dtype=int)
        y = np.arange(seg.shape[0], dtype=int)
        xx, yy = np.meshgrid(x, y)
        indices = np.concatenate((xx[seg][..., None], yy[seg][..., None]), axis=1)
        return indices

    def get_filtered_depth_ids(self, depth: np.ndarray, seg: np.ndarray):
        n_smallest = 500
        std_scale = 4
        seg_flat_depth = depth.reshape(-1)[seg.reshape(-1)]

        if seg_flat_depth.size <= n_smallest:
            # A mask this small has no "n_smallest largest" to partition out, and
            # argpartition raises on it. Use every point for the statistics.
            largest_values = seg_flat_depth
        else:
            semi_sorted_args = np.argpartition(-seg_flat_depth[:], n_smallest)
            largest_values = seg_flat_depth[semi_sorted_args[:n_smallest]]

        mean, std = np.mean(largest_values), np.std(largest_values)
        filter_args = seg_flat_depth[:] > (mean + std_scale * std)

        return ~filter_args

    def project_pointcloud(self,
                           keypoints: np.ndarray,
                           depth_image: np.ndarray,
                           K0: np.ndarray,
                           depth_units: str = 'mm'):

        depth_values_0 = np.empty((keypoints.shape[0], 1))
        for i in range(keypoints.shape[0]):
            depth_values_0[i, 0] = depth_image[keypoints[i, 1], keypoints[i, 0]]

        if depth_units == 'mm':
            depth_values_0 = depth_values_0 / 1e3

        keypoint_pos_C_0 = depth_values_0 * np.concatenate((keypoints, np.ones((len(keypoints), 1))),
                                                           axis=1) @ np.linalg.inv(K0).T
        return keypoint_pos_C_0

    def _get_pointclouds(self, data: SceneData):
        K = data["intrinsics_0"]
        depth = data["depth_0"]
        keypoints = self._get_segmented_keypoints(data['seg_0'])
        projected_keypoints = self.project_pointcloud(keypoints, depth, K, 'm')
        if self.filter_pointcloud:
            valid_keypoints = projected_keypoints[:, 2] != 0
            data['pc0_keep_id'] = (self.get_filtered_depth_ids(data["depth_0"], data["seg_0"]).astype(
                int) + valid_keypoints.astype(int)) == 2
        projected_keypoints = np.concatenate((projected_keypoints, np.ones((projected_keypoints.shape[0], 1))),
                                             axis=1).transpose(1, 0)
        data['pc0'] = (data["T_WC_0"] @ projected_keypoints).astype(np.float32)
        data['pc0'] = data["pc0"].transpose(1, 0)[:, :3]

        K = data["intrinsics_1"]
        depth = data["depth_1"]
        keypoints = self._get_segmented_keypoints(data['seg_1'])
        projected_keypoints = self.project_pointcloud(keypoints, depth, K, 'm')
        if self.filter_pointcloud:
            valid_keypoints = projected_keypoints[:, 2] != 0
            data['pc1_keep_id'] = (self.get_filtered_depth_ids(data["depth_1"], data["seg_1"]).astype(
                int) + valid_keypoints.astype(int)) == 2
        projected_keypoints = np.concatenate((projected_keypoints, np.ones((projected_keypoints.shape[0], 1))),
                                             axis=1).transpose(1, 0)
        data['pc1'] = (data["T_WC_1"] @ projected_keypoints).astype(np.float32)
        data['pc1'] = data["pc1"].transpose(1, 0)[:, :3]

    def __call__(self, data: SceneData) -> ProcessedData:

        self._initial_processing(data)
        self._get_pointclouds(data)

        data["image_0"] = data["image_0"] / 255 * 2 - 1
        data["image_1"] = data["image_1"] / 255 * 2 - 1

        rgb_data = data["image_0"].transpose(1, 2, 0).reshape(-1, 3)
        rgb_data = rgb_data[data["seg_0"].reshape(-1), :]
        data["pc0"] = np.concatenate((data["pc0"], rgb_data), axis=1)

        rgb_data = data["image_1"].transpose(1, 2, 0).reshape(-1, 3)
        rgb_data = rgb_data[data["seg_1"].reshape(-1), :]
        data["pc1"] = np.concatenate((data["pc1"], rgb_data), axis=1)

        data["pc0"] = data["pc0"][data['pc0_keep_id'], :]
        data["pc1"] = data["pc1"][data['pc1_keep_id'], :]

        sample_args = np.random.randint(low=0, high=data["pc0"].shape[0], size=self.n_points)
        data["pc0"] = data["pc0"][sample_args, :]
        sample_args = np.random.randint(low=0, high=data["pc1"].shape[0], size=self.n_points)
        data["pc1"] = data["pc1"][sample_args, :]

        data["pc0"] = np.concatenate((data["pc0"], data["pc0"][:, :3]), axis=1)
        data["pc1"] = np.concatenate((data["pc1"], data["pc1"][:, :3]), axis=1)

        return {
            'pointcloud_0': data["pc0"].transpose(1, 0).astype(np.float32),
            'pointcloud_1': data["pc1"].transpose(1, 0).astype(np.float32),
        }
