"""
naoqi_driver.launch.py: launch the NAOqi driver, the bridge to the Pepper robot itself.

Launch files included:
    naoqi_driver2/pepper_bringup.launch.py — the driver, plus the gscam2 front
    camera when use_camera is true. That file forwards to
    naoqi_driver.launch.py, which is where publish_wheel_odom_tf is defined;
    it is deliberately NOT re-declared anywhere above, because re-declaring it
    at each layer is what let its default silently drift to true before.

Nodes started:
    None directly; everything comes from the included bringup launch.

What the driver provides (the parts this system uses):
    /cmd_vel                            consumed — the Nav2 collision monitor's
                                        output is what actually moves the robot
    /pepper_odom                        wheel odometry, as a TOPIC
    /joint_states, /tf, robot state     Pepper's body TF tree
    /naoqi_driver/speech_with_feedback  the action SpeechWithFeedback (and so
                                        the ASR -> CM -> TTS pipeline) calls

Launch arguments:
    nao_ip (default: "172.29.111.240")
        Robot address. THE DEFAULT IS THE CMU-Pepper ROUTER ADDRESS -- on the
        PepperNet AP the robot is 10.42.0.204 instead, so pass
        nao_ip:=10.42.0.204 there. Both values are the ones recorded in
        gscam2/launch/pepper_camera_launch.py; confirm against the robot before
        trusting either.
    nao_port (default: "9559")
    user / password (defaults: "nao" / "no_password")
    network_interface (default: "eth0")
        The LOCAL interface whose address the driver hands NAOqi so the robot
        can connect back (ros_env.cpp:30-48). It is looked up by name and the
        node THROWS if it does not exist, listing the interfaces it did find --
        so on a laptop associated over Wi-Fi this must be the wireless
        interface, not eth0. Check with `ip -br addr`.
    qi_listen_url (default: "tcp://0.0.0.0:0")
        Endpoint NAOqi connects back to for audio.
    namespace (default: "")
    use_camera (default: "false")
        Start the gscam2 Pepper front camera alongside the driver. FALSE here,
        unlike naoqi_driver2's own bringup, which defaults it true: this system
        sees through the bottom-mounted RealSense (realsense_bottom.launch.py
        -> overt_attention), and the front camera additionally needs the stream
        started on the robot first (see Prerequisites).
    publish_wheel_odom_tf (default: "false", set in naoqi_driver.launch.py)
        NOT declared here, and should stay false. FAST-LIO owns
        odom -> base_footprint; publishing it from wheel odometry too gives
        that frame two parents and splits the TF tree. The flag is latched in
        JointStateConverter's constructor with no parameter callback, so it
        cannot be corrected at runtime. Wheel odometry remains available as
        /pepper_odom either way.

Configuration:
    None of its own; naoqi_driver2 reads its boot config internally.

Prerequisites:
    The robot powered, awake, and reachable at nao_ip. For use_camera:=true,
    start the stream on the robot first and stop it afterwards:
        ssh nao@<robot-ip> '~/start_camera.sh'
        ssh nao@<robot-ip> '~/stop_camera.sh'

Usage:
    ros2 launch dec_launch naoqi_driver.launch.py
    ros2 launch dec_launch naoqi_driver.launch.py nao_ip:=10.42.0.204
    ros2 launch dec_launch naoqi_driver.launch.py use_camera:=true

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

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    bringup_launch = os.path.join(
        get_package_share_directory('naoqi_driver'),
        'launch', 'pepper_bringup.launch.py')

    return LaunchDescription([

        DeclareLaunchArgument(
            'nao_ip', default_value='172.29.111.240',
            description='Robot IP. 172.29.111.240 on the CMU-Pepper router, '
                        '10.42.0.204 on the PepperNet AP.'),
        DeclareLaunchArgument(
            'nao_port', default_value='9559',
            description='Port used for the NAOqi connection.'),
        DeclareLaunchArgument(
            'user', default_value='nao',
            description='Username for the connection.'),
        DeclareLaunchArgument(
            'password', default_value='no_password',
            description='Password for the connection.'),
        DeclareLaunchArgument(
            'network_interface', default_value='eth0',
            description='LOCAL interface whose address is handed to NAOqi for '
                        'the return connection. Looked up by name; the node '
                        'throws if it does not exist. Use the Wi-Fi interface '
                        '(see `ip -br addr`) when not on wired ethernet.'),
        DeclareLaunchArgument(
            'qi_listen_url', default_value='tcp://0.0.0.0:0',
            description='Endpoint to listen for incoming NAOqi connections '
                        '(audio).'),
        DeclareLaunchArgument(
            'namespace', default_value='',
            description='Namespace for the driver node.'),
        # Overrides the included bringup's own 'true': this system's vision is
        # the bottom RealSense, and the front camera needs a stream started on
        # the robot by hand first.
        DeclareLaunchArgument(
            'use_camera', default_value='false',
            description='Also start the gscam2 Pepper front camera. Requires '
                        '~/start_camera.sh to have been run on the robot.'),

        # publish_wheel_odom_tf is deliberately NOT declared here. It is
        # defined once, in naoqi_driver.launch.py (default false). Includes
        # share the launch context, so `publish_wheel_odom_tf:=true` on this
        # command line still reaches the node -- but see the header: with
        # FAST-LIO running, it should stay false.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup_launch),
            launch_arguments={
                'nao_ip': LaunchConfiguration('nao_ip'),
                'nao_port': LaunchConfiguration('nao_port'),
                'user': LaunchConfiguration('user'),
                'password': LaunchConfiguration('password'),
                'network_interface': LaunchConfiguration('network_interface'),
                'qi_listen_url': LaunchConfiguration('qi_listen_url'),
                'namespace': LaunchConfiguration('namespace'),
                'use_camera': LaunchConfiguration('use_camera'),
            }.items(),
        ),
    ])
