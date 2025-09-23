from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
import os

'''
This launch file starts multiple tactile streamer nodes for different sensors.
Each node streams tactile data from a specified sensor serial number and publishes it to a unique topic.

Launch Arguments:
- sensors_root: Root directory for sensor configurations
- mode: Processing mode (depth, gradient, pointcloud) 
- model_device: Device for model execution (cuda, cpu)
- use_mask: Whether to apply contact mask
- rate: Publishing rate in Hz
- contact_mode: Contact detection mode (standard, flat)

Usage Examples:
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py mode:=depth
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py mode:=pointcloud model_device:=cpu
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py sensors_root:=/path/to/sensors
'''

def launch_setup(context, *args, **kwargs):
    # Get launch arguments
    sensors_root = LaunchConfiguration('sensors_root').perform(context)
    mode = LaunchConfiguration('mode').perform(context)
    model_device = LaunchConfiguration('model_device').perform(context)
    use_mask = LaunchConfiguration('use_mask').perform(context) == 'true'
    rate = float(LaunchConfiguration('rate').perform(context))
    contact_mode = LaunchConfiguration('contact_mode').perform(context)
    
    # Auto-discover sensors from sensors_root directory
    sensors = []
    if os.path.exists(sensors_root):
        for item in os.listdir(sensors_root):
            sensor_path = os.path.join(sensors_root, item)
            config_file = os.path.join(sensor_path, f"{item}.yaml")
            if os.path.isdir(sensor_path) and os.path.exists(config_file):
                sensors.append(item)
    
    # Fallback to hardcoded list if no sensors found or invalid path
    if not sensors:
        sensors = ["D21275", "D21273", "D21242", "D21119"]  # Common DIGIT sensor serials
    
    nodes = []
    for serial in sensors:
        # Determine topic name based on mode
        topic_name = f"/tactile/{mode}_{serial}"
        
        node = Node(
            package="vistac_sdk",  # Update this to match your actual ROS2 package name
            executable="tactile_streamer_node",
            name=f"tactile_streamer_{serial}",
            output="screen",
            parameters=[{
                "serial": serial,
                "sensors_root": sensors_root,
                "mode": mode,
                "contact_mode": contact_mode,
                "model_device": model_device,
                "use_mask": use_mask,
                "refine_mask": True,
                "relative": False,
                "relative_scale": 1.0,
                "mask_only_pointcloud": False,
                "return_color": False,
                "color_dist_threshold": 15,
                "height_threshold": 0.2,
                "topic": topic_name,
                "rate": rate,
                "verbose": True
            }],
        )
        nodes.append(node)
    
def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments with defaults
        DeclareLaunchArgument(
            'sensors_root',
            default_value='/home/bhsong/ros2/gaussianfeels/src/gaussianfeels_robot/tactile_sensor/vistac_sdk/sensors',
            description='Root directory for sensor configurations'
        ),
        DeclareLaunchArgument(
            'mode',
            default_value='depth',
            description='Processing mode: depth, gradient, or pointcloud'
        ),
        DeclareLaunchArgument(
            'model_device',
            default_value='cuda',
            description='Device for model execution: cuda or cpu'
        ),
        DeclareLaunchArgument(
            'use_mask',
            default_value='true',
            description='Whether to apply contact mask'
        ),
        DeclareLaunchArgument(
            'rate',
            default_value='15.0',
            description='Publishing rate in Hz'
        ),
        DeclareLaunchArgument(
            'contact_mode',
            default_value='standard',
            description='Contact detection mode: standard or flat'
        ),
        
        # Use OpaqueFunction to handle dynamic node creation
        OpaqueFunction(function=launch_setup)
    ])