#!/usr/bin/env python3
"""conversation_manager.launch.py

Launch the conversation_manager node with its package configuration.

Nodes started:
    conversation_manager/conversation_manager (node: conversation_manager)
        Retrieval-augmented dialogue action server over the Upanzi knowledge
        base.

Launch arguments:
    (none)

Configuration:
    config/conversation_manager_configuration.yaml — collection name, LLM and
    embedding models, and retrieval settings.

Prerequisites:
    An OpenAI-compatible LLM endpoint must be reachable at llm_base_url, and
    LLM_API_KEY must be exported in the environment (it is deliberately not a
    ROS parameter). The ChromaDB collection is built on first configure from
    the knowledge-base JSON in data/.

Usage:
    ros2 launch conversation_manager conversation_manager.launch.py

The node's ROS interface is documented in conversation_manager_application.py.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 8, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('conversation_manager'),
        'config',
        'conversation_manager_configuration.yaml'
    )

    return LaunchDescription([
        Node(
            package='conversation_manager',
            executable='conversation_manager',
            name='conversation_manager',
            parameters=[config],
            output='screen',
        ),
    ])
