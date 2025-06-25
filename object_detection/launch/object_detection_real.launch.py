from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    EnvironmentVariable,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare all launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            "smb_name",
            default_value=EnvironmentVariable("ROBOT_ID", default_value="sim"),
            description="Name of the SMB in the format smb26x (relevant for calibrations)",
        ),
        DeclareLaunchArgument(
            "gpu",
            default_value="local",
            description="Run on GPU? Options: 'local', 'remote' (default), 'off'",
            choices=["local", "remote", "off"],
        ),
        DeclareLaunchArgument(
            "GPU_user",
            default_value=EnvironmentVariable("USER"),
            description="Username to use on the jetson xavier GPU",
        ),
        DeclareLaunchArgument(
            "input_camera_name",
            default_value="/rgb_camera",
            description="Name of the camera, i.e. topic prefix for camera stream and camera info",
        ),
        DeclareLaunchArgument(
            "debayer_image",
            default_value="false",
            description="Debayer the images (supplied in $input_camera_name/image_raw)",
        ),
        DeclareLaunchArgument(
            "lidar_topic",
            default_value="/rslidar/points",
            description="Topic containing the point cloud from the lidar",
        ),
        DeclareLaunchArgument(
            "object_detection_classes",
            default_value="[0,24,25,28,32,39,41,45,46,47,56]",
            description="List of the ids of classes for detection (COCO dataset)",
        ),
        DeclareLaunchArgument(
            "model_dir_path",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("object_detection"),
                    "models",
                ]
            ),
            description="path to the yolo model directory",
        ),
        DeclareLaunchArgument(
            "model", default_value="yolov5l6", description="yolo model name"
        ),
    ]

    # Debayer the image (conditionally included)
    debayer_image_group = GroupAction(
    [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        FindPackageShare("object_detection"),
                        "launch",
                        "debayer.launch.py",
                    ]
                )
            ),
            condition=IfCondition(LaunchConfiguration("debayer_image")),
            launch_arguments=[
                ("input_camera_name", LaunchConfiguration("input_camera_name"))
            ],
        )
    ]
)

    # Object detection node

    object_detection_group = GroupAction(
        [
            Node(
                package="object_detection",
                executable="object_detection_node.py",
                name="object_detector",
                output="screen",
                parameters=[
                    # Input related
                    {
                        "camera_topic": PathJoinSubstitution(
                            [LaunchConfiguration("input_camera_name"), "image_debayered"]
                        )
                    },
                    {
                        "camera_info_topic": PathJoinSubstitution(
                            [LaunchConfiguration("input_camera_name"), "camera_info"]
                        )
                    },
                    {"lidar_topic": LaunchConfiguration("lidar_topic")},
                    # Output related - load from file
                    {"project_object_points_to_image": True},
                    {"project_all_points_to_image": False},
                    {"object_detection_pos_topic": "object_positions"},
                    {"object_detection_output_image_topic": "detections_in_image"},
                    {"object_detection_point_clouds_topic": "detection_point_clouds"},
                    {"object_detection_info_topic": "detection_info"},
                    # Camera Lidar synchronization related
                    {"camera_lidar_sync_queue_size": 10},
                    {"camera_lidar_sync_slop": 0.1},
                    # Point Projector related
                    {
                        "project_config": [
                            TextSubstitution(text="projector_config_"),
                            LaunchConfiguration("smb_name"),
                            TextSubstitution(text=".yaml"),
                        ]
                    },
                    # Object detection related
                    {"model": LaunchConfiguration("model")},
                    {"model_dir_path": LaunchConfiguration("model_dir_path")},
                    {"device": "0" if LaunchConfiguration("gpu") != "off" else "cpu"},
                    {"confident": 0.0},
                    {"iou": 0.1},
                    {"classes": LaunchConfiguration("object_detection_classes")},
                    {"multiple_instance": False},
                    # Object localization related
                    {"model_method": "hdbscan"},
                    {"ground_percentage": 25},
                    {"bb_contract_percentage": 10},
                ],
            )
        ]
    )

    return LaunchDescription(
        declared_arguments + [debayer_image_group, object_detection_group]+ [Node(
                package="object_detection",
                executable="object_transformation_node.py",
                name="object_transform",
                output="screen",
                parameters=[]
            )]
    )