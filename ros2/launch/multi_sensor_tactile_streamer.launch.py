from launch import LaunchDescription
from launch_ros.actions import Node

'''
This launch file starts multiple tactile streamer nodes for different sensors.
Each node streams tactile data from a specified sensor serial number and publishes it to a unique topic.'''

def generate_launch_description():
    # List your sensor serials here
    sensors = ["D21275", "D21276"]  # <-- Edit as needed
    nodes = []
    for serial in sensors:
        nodes.append(
            Node(
                package="gs_sdk",  # Replace with your ROS2 package name if different
                executable="tactile_streamer_node",
                name=f"tactile_streamer_{serial}",
                output="screen",
                parameters=[
                    {"serial": serial,
                     "sensors_root": "gs_sdk/sensors",
                     "mode": "pointcloud",
                     "topic": f"/tactile/pointcloud_{serial}"}
                ],
            )
        )
    return LaunchDescription(nodes)