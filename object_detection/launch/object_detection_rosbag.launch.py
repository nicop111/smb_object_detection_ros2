from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
    SetParameter,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare all launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            "rosbag_name",
            default_value="object_detection_bag",
            description="Base name of the rosbag file (without extension)",
        ),
        DeclareLaunchArgument(
            "config_file",
            default_value="/smb_ros2_workspace/src/rosbags/object_detection_bag_ros2/object_detection_bag.db3",
            description="Full path to the rosbag file",
        ),
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution(
                [
                    "/home/shobhit/Downloads",
                    [LaunchConfiguration("rosbag_name"), ".yaml"],
                ]
            ),
            description="Full path to the config YAML file",
        ),
        DeclareLaunchArgument(
            "gpu",
            default_value="off",
            description="Run on GPU? Options: 'local', 'remote', 'off' (default)",
            choices=["local", "remote", "off"],
        ),
        DeclareLaunchArgument(
            "input_camera_name",
            default_value="/rgb_camera",
            description="Name of the camera, i.e. topic prefix for camera stream and camera info",
        ),
        DeclareLaunchArgument(
            "lidar_topic",
            default_value="/rslidar/points",
            description="Topic containing the point cloud from the lidar",
        ),
    ]

    # Set use_sim_time parameter
    sim_time_action = SetParameter(name="use_sim_time", value=True)

    # Topic droppers
    topic_droppers_group = GroupAction(
        [
            Node(
                package="topic_tools",
                executable="drop",
                name="image_dropper",
                arguments=[
                    LaunchConfiguration("input_camera_name") + "/image_raw",
                    "19",
                    "20",
                    LaunchConfiguration("input_camera_name") + "/slow/image_raw",
                ],
            ),
            Node(
                package="topic_tools",
                executable="drop",
                name="camera_info_dropper",
                arguments=[
                    LaunchConfiguration("input_camera_name") + "/camera_info",
                    "19",
                    "20",
                    LaunchConfiguration("input_camera_name") + "/slow/camera_info",
                ],
            ),
        ]
    )

    # Object detection launch
    object_detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("object_detection"),
                    "launch",
                    "object_detection.launch.py",
                ]
            )
        ),
        launch_arguments={
            "gpu": LaunchConfiguration("gpu"),
            "input_camera_name": LaunchConfiguration("input_camera_name"),
            "lidar_topic": LaunchConfiguration("lidar_topic"),
        }.items(),
    )

    # # RViz node
    # rviz_node = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     output="screen",
    #     arguments=[
    #         "-d",
    #         PathJoinSubstitution(
    #             [FindPackageShare("object_detection"), "rviz", "object_detection.rviz"]
    #         ),
    #     ],
    #     parameters=[LaunchConfiguration("config_file")],
    # )

    # Rosbag player
    rosbag_player = Node(
        package="rosbag2",
        executable="play",
        name="player",
        output="screen",
        arguments=["--delay", "3", "--clock", LaunchConfiguration("rosbag_path")],
        required=False,
    )

    return LaunchDescription(
        declared_arguments
        + [
            sim_time_action,
            topic_droppers_group,
            object_detection_launch,
            # rviz_node,
            rosbag_player,
        ]
    )
