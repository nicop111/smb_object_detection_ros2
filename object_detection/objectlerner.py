#!/usr/bin/env python3
import os
import sys
import yaml
import numpy as np
import rclpy
from rclpy.node import Node


class BBoxToDistance(Node):
    def __init__(self):
        super().__init__("bb2dist_converter")

        # Get package share directory in ROS2 style
        package_share_directory = self.get_package_share_directory()

        # Set up paths (ROS2 style)
        self.CFG_DIR = os.path.join(package_share_directory, "cfg")
        self.DATA_DIR = os.path.join(self.CFG_DIR, "data")

        # Declare parameter for polynomial degree
        self.declare_parameter("degree", 5)

        self.get_logger().info("BBox to Distance converter initialized")

    def get_package_share_directory(self):
        """Helper to get package share directory"""
        from ament_index_python.packages import get_package_share_path

        return str(get_package_share_path("your_package_name"))

    def bb2dist(self):
        degree = self.get_parameter("degree").value
        bb2dist_dict = {}
        data_dir = os.path.join(self.DATA_DIR, "bb2dist")

        for filename in os.listdir(data_dir):
            obj, _ = os.path.splitext(filename)
            data = np.loadtxt(os.path.join(data_dir, filename))

            # Perform polynomial fitting
            coefficients = np.polyfit(data[:, 0], data[:, 1], degree).tolist()
            bb2dist_dict[obj] = coefficients
            self.get_logger().info(f"Processed {obj} with coefficients: {coefficients}")

        # Save results
        output_path = os.path.join(self.CFG_DIR, "bb2dist.yaml")
        with open(output_path, "w") as file:
            yaml.dump(bb2dist_dict, file)

        self.get_logger().info(f"Saved results to {output_path}")


def main(args=None):
    rclpy.init(args=args)

    # Create node and run processing
    converter = BBoxToDistance()
    converter.bb2dist()

    # Shutdown
    converter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
