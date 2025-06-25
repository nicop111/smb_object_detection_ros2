#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from tf2_msgs.msg import TFMessage
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
from sensor_msgs_py import point_cloud2
import time
import cv2
from os.path import join
from numpy.lib.recfunctions import unstructured_to_structured
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from geometry_msgs.msg import PoseArray, Pose, Quaternion

from object_detection_msgs.msg import (
    PointCloudArray,
    ObjectDetectionInfo,
    ObjectDetectionInfoArray,
)
from std_msgs.msg import Header

from object_detection.objectdetectorONNX import ObjectDetectorONNX
from object_detection.pointprojector import PointProjector
from object_detection.objectlocalizer import ObjectLocalizer
from object_detection.utils import *

# from object_detection.ros_numpy import *

from ament_index_python.packages import get_package_share_directory


def transform_matrix(translation, rotation):
    tx, ty, tz = translation
    qx, qy, qz, qw = rotation
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


class CameraToWorldNode(Node):
    def __init__(self):
        super().__init__('camera_to_world_transform')

        # Parameters
        self.declare_parameter('camera_to_base_translation', [0.0, 0.0, 0.0])
        self.declare_parameter('camera_to_base_rotation', [ 0.5, -0.5, 0.5, -0.5 ])
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('world_frame', 'odom')
        trans = self.get_parameter('camera_to_base_translation').value
        rot = self.get_parameter('camera_to_base_rotation').value
        self.base_frame = self.get_parameter('robot_base_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        self.T_cam2base = transform_matrix(trans, rot)

        # Subscribers
        self.obj_sub = self.create_subscription(
            PoseArray,
            '/object_poses',
            self.object_callback,
            10
        )
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )

        # Publisher
        self.obj_world_pub = self.create_publisher(PoseArray, 'object_world_poses', 10)

        # State
        self.last_obj_array = None
        self.last_base2world = None

    def object_callback(self, msg: PoseArray):
        self.get_logger().debug(f'Received PoseArray with {len(msg.poses)} poses; header.frame_id={msg.header.frame_id}')
        self.last_obj_array = msg
        self.try_transform()

    def tf_callback(self, msg: TFMessage):
        for t in msg.transforms:
            if t.header.frame_id == self.world_frame and t.child_frame_id == self.base_frame:
                tr = t.transform.translation
                rot = t.transform.rotation
                self.last_base2world = ([tr.x, tr.y, tr.z], [rot.x, rot.y, rot.z, rot.w])
                self.get_logger().debug(
                    f'Got TF {self.world_frame} -> {self.base_frame}:'
                    f' translation={self.last_base2world[0]}'
                    f' rotation={self.last_base2world[1]}'
                )
                break
        self.try_transform()

    def try_transform(self):
        if self.last_obj_array is None:
            self.get_logger().warn('No object poses yet; skipping transform')
            return
        if self.last_base2world is None:
            self.get_logger().warn('No base->world TF yet; skipping transform')
            return

        trans, rot = self.last_base2world
        T_base2world = transform_matrix(trans, rot)

        self.get_logger().info(f'Transforming {len(self.last_obj_array.poses)} poses using:'
                                f' T_base2world translation={trans}, rotation={rot}')

        world_array = PoseArray()
        world_array.header.stamp = self.get_clock().now().to_msg()
        world_array.header.frame_id = self.world_frame

        for idx, pose in enumerate(self.last_obj_array.poses):
            p_cam = np.array([pose.position.x, pose.position.y, pose.position.z, 1.0])
            p_world = T_base2world @ self.T_cam2base @ p_cam

            new_pose = Pose()
            new_pose.position.x, new_pose.position.y, new_pose.position.z = p_world[:3]
            new_pose.orientation = pose.orientation
            world_array.poses.append(new_pose)
            self.get_logger().debug(f'Pose {idx}: cam=({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f}) -> '
                                     f'world=({p_world[0]:.2f}, {p_world[1]:.2f}, {p_world[2]:.2f})')

        self.obj_world_pub.publish(world_array)
        self.get_logger().info(f'Published transformed PoseArray with {len(world_array.poses)} poses')

        # Optionally reset state
        # self.last_obj_array = None
        # self.last_base2world = None


def main(args=None):
    rclpy.init(args=args)
    node = CameraToWorldNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()