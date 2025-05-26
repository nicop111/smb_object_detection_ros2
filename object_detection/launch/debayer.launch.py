from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    EnvironmentVariable,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # Declare all launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            "input_camera_name",
            default_value="/rgb_camera",
            description="Name of the camera, i.e. topic prefix for camera stream and camera info",
        ),
    ]

    # Create a container and add the debayer node to it
    debayer_container = ComposableNodeContainer(
        name="debayer_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="image_proc",
                plugin="image_proc::DebayerNode",
                name="Debayer_node",
                remappings=[
                    (
                        "image_raw",
                        [LaunchConfiguration("input_camera_name"), "/image_raw"],
                    ),
                    (
                        "image_color",
                        [LaunchConfiguration("input_camera_name"), "/image_color"],
                    ),
                    (
                        "image_mono",
                        [LaunchConfiguration("input_camera_name"), "/image_mono"],
                    ),
                ],
            )
        ],
    )

    return LaunchDescription(
        declared_arguments
        + [
            debayer_container,
        ]
    )
