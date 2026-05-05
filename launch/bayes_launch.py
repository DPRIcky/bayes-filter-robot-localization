import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    stage_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('stage_ros2'), 'launch', 'stage.launch.py')
        ),
        launch_arguments={'world': 'cave'}.items(),
    )

    rviz_config = os.path.join(
        get_package_share_directory('ras598_assignment_3'), 'rviz', 'bayes.rviz'
    )

    bayes_node = Node(
        package='ras598_assignment_3',
        executable='bayes_filter',
        name='bayes_filter_3d_node',
        output='screen',
        emulate_tty=True,
    )

    rviz_process = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config],
        output='screen',
        additional_env={
            'QT_QPA_PLATFORM': os.environ.get('QT_QPA_PLATFORM', 'xcb'),
            'DISPLAY': os.environ.get('DISPLAY', ':0'),
        },
    )

    return LaunchDescription([
        SetEnvironmentVariable('QT_QPA_PLATFORM', 'wayland'),
        stage_launch,
        bayes_node,
        rviz_process,
    ])
