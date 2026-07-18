import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('speech_event'),
        'config',
        'speech_event_configuration.yaml'
    )

    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='speech_event',
            executable='speech_event',
            name='speech_recognition',
            parameters=[config],
            output='screen',
        ),
        launch_ros.actions.Node(
            package='speech_event',
            executable='speech_event_localization',
            name='sound_localization',
            parameters=[config],
            output='screen',
        ),
        launch_ros.actions.Node(
            package='speech_event',
            executable='speech_event_recorder',
            name='audio_recorder',
            parameters=[config],
            output='screen',
        ),
    ])
