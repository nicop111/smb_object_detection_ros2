#!/usr/bin/env python3
import yaml
import numpy as np
from rclpy.node import Node
from rclpy.logging import get_logger

from object_detection.utils import check_validity_lidar2camera_transformation

AXIS_X = 0
AXIS_Y = 1


class PointProjector:
    def __init__(self, node, config_path):
        self.node = node
        self.logger = node.get_logger() if node else get_logger("point_projector")

        try:
            with open(config_path) as file:
                self.config = yaml.safe_load(file)
                R_camera_lidar = np.float64(self.config["R_camera_lidar"])
                R_correction = np.float64(self.config["R_correction"])
                t_camera_lidar = np.float64(self.config["t_camera_lidar"])
                t_correction = np.float64(self.config["t_correction"])
                self.forward_axis = self.config["forward_axis"]

            if not check_validity_lidar2camera_transformation(
                R_camera_lidar, t_camera_lidar
            ):
                self.logger.error(
                    f"Lidar to Camera transformation matrices are not correctly configured in {config_path}"
                )
                raise RuntimeError("Invalid transformation matrices")

            R_camera_lidar = R_camera_lidar @ R_correction
            t_camera_lidar = t_camera_lidar + t_correction

            self.set_extrinsic_params(R_camera_lidar, t_camera_lidar)
            self.K = None
            self.w = None
            self.h = None
            self.P = None

            self.logger.info("PointProjector initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize PointProjector: {str(e)}")
            raise

    def set_intrinsic_params(self, K, size):
        """Set camera intrinsic parameters"""
        self.K = K
        self.w = size[0]
        self.h = size[1]
        self.P = np.zeros((3, 4))
        self.P[:, :3] = self.K
        self.logger.debug("Intrinsic parameters set")

    def set_extrinsic_params(self, R, t):
        """Set camera extrinsic parameters"""
        self.R = R
        self.t = t
        self.T = np.eye(4)
        self.T[:3, :3] = R
        self.T[:3, 3] = t
        self.logger.debug("Extrinsic parameters set")

    def transform_points(self, points):
        """Transform points from lidar to camera frame"""
        homo_coor = np.ones(points.shape[0])
        XYZ = np.vstack((np.transpose(points), homo_coor))
        self.logger.info(f"Transformed points shape: {XYZ.shape}")
        XYZ = self.T @ XYZ
        XYZ = XYZ / XYZ[3, :]
        return np.transpose(XYZ[:3, :])

    def project_points(self, points):
        """Project points from camera frame to image plane"""
        indices = np.arange(0, len(points))

        # Filter front hemisphere points
        axis_idx = abs(self.forward_axis) - 1
        if self.forward_axis > 0:
            front_hemisphere = points[:, axis_idx] > 0
        else:
            front_hemisphere = points[:, axis_idx] < 0

        front_hemisphere_indices = np.nonzero(front_hemisphere)[0]
        indices = indices[front_hemisphere_indices]
        points = points[front_hemisphere_indices, :]

        # Project points
        homo_coor = np.ones(points.shape[0])
        XYZ = np.vstack((np.transpose(points), homo_coor))
        xy = self.P @ XYZ
        xy = xy / xy[2, None]
        return np.transpose(xy[:2, :]), indices

    # def project_points_on_image(self, points):
    #     """Project 3D points onto image and filter to those within frame"""
    #     points_on_image, indices = self.project_points(points)
    #     points_on_image = np.uint32(np.squeeze(points_on_image))

    #     inside_frame_x = np.logical_and(
    #         (points_on_image[:, AXIS_X] >= 0), (points_on_image[:, AXIS_X] < self.w - 1)
    #     )
    #     inside_frame_y = np.logical_and(
    #         (points_on_image[:, AXIS_Y] >= 0), (points_on_image[:, AXIS_Y] < self.h - 1)
    #     )
    #     inside_frame_indices = np.nonzero(
    #         np.logical_and(inside_frame_x, inside_frame_y)
    #     )[0]

    #     indices = indices[inside_frame_indices]
    #     points_on_image = points_on_image[inside_frame_indices, :]
    #     return points_on_image, indices
    def project_points_on_image(self, points):
        """
        Project 3D points onto image and filter to those within frame

        Args:
            points: Nx3 array of 3D points
        Returns:
            points_on_image: Nx2 array of image coordinates
            indices: indices of valid points in original array
        """
        # Project points and get initial indices
        points_on_image, indices = self.project_points(points)

        # Handle empty input
        if points_on_image.size == 0:
            return np.array([]), np.array([])

        # Remove single-dimensional entries
        points_on_image = np.squeeze(points_on_image)

        # Handle single point case
        if len(points_on_image.shape) == 1:
            points_on_image = points_on_image.reshape(1, 2)

        # Clip values to valid range before converting to uint32
        points_on_image = np.clip(points_on_image, 0, [self.w - 1, self.h - 1])
        points_on_image = points_on_image.astype(np.uint32)

        # Check points within frame bounds
        inside_frame_x = np.logical_and(
            points_on_image[:, AXIS_X] >= 0, points_on_image[:, AXIS_X] < self.w - 1
        )
        inside_frame_y = np.logical_and(
            points_on_image[:, AXIS_Y] >= 0, points_on_image[:, AXIS_Y] < self.h - 1
        )

        # Combine conditions and get valid indices
        inside_frame_indices = np.nonzero(
            np.logical_and(inside_frame_x, inside_frame_y)
        )[0]

        # Filter points and indices
        valid_indices = indices[inside_frame_indices]
        valid_points = points_on_image[inside_frame_indices]

        # Add debug logging
        self.logger.debug(
            f"Projected points shape: {valid_points.shape}, "
            f"Valid indices: {len(valid_indices)}/{len(points)}"
        )

        return valid_points, valid_indices
