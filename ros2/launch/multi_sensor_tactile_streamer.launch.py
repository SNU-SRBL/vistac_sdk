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
- mode: Processing mode (depth, gradient, pointcloud, pointcloud_force, force_field, force_vector)
    - `pointcloud_force` produces a pointcloud colored by force field (requires enable_force:=true)
- model_device: Device for model execution (cuda, cpu)
- use_mask: Whether to apply contact mask
- rate: Publishing rate in Hz
- contact_mode: Contact detection mode (standard, flat)
- enable_force: Enable force estimation using Sparsh models (default: false)
- temporal_stride: Temporal stride for force estimation (default: 5)
- outputs: Explicit list of outputs (comma-separated, overrides mode)
- refine_mask: Refine contact mask edges (default: true)
- relative: Use relative depth instead of absolute (default: false)
- mask_only_pointcloud: Only include masked region in pointcloud (default: false)
- return_color: Include RGB color in pointcloud (default: false)
- pointcloud_color: Which source to use for pointcloud coloring (none|image|force) (default: none)
- pointcloud_color_format: How to encode colors in PointCloud2 (rgb_packed|r_g_b) (default: rgb_packed)
- publish_force_fields: If true, include per-point fx,fy,fz fields in PointCloud2 (default: false)
- force_mapping: Mapping method from force image -> points (nearest|bilinear) (default: nearest)
- height_threshold: Contact detection height threshold in mm (default: 0.2)

Usage Examples:
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py mode:=depth
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py mode:=pointcloud model_device:=cpu
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py sensors_root:=/path/to/sensors
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py enable_force:=true mode:=force_vector
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py outputs:=depth,force_field,force_vector
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py outputs:=pointcloud,force_field return_color:=true
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py mode:=pointcloud_force enable_force:=true pointcloud_color:=force publish_force_fields:=true
'''

def launch_setup(context, *args, **kwargs):
    # Get launch arguments
    sensors_root = LaunchConfiguration('sensors_root').perform(context)
    mode = LaunchConfiguration('mode').perform(context)
    model_device = LaunchConfiguration('model_device').perform(context)
    use_mask = LaunchConfiguration('use_mask').perform(context) == 'true'
    rate = float(LaunchConfiguration('rate').perform(context))
    contact_mode = LaunchConfiguration('contact_mode').perform(context)
    enable_force = LaunchConfiguration('enable_force').perform(context) == 'true'
    temporal_stride = int(LaunchConfiguration('temporal_stride').perform(context))
    outputs_str = LaunchConfiguration('outputs').perform(context)
    outputs = [s.strip() for s in outputs_str.split(',')] if outputs_str else []
    
    # Pointcloud-specific parameters
    refine_mask = LaunchConfiguration('refine_mask').perform(context) == 'true'
    relative = LaunchConfiguration('relative').perform(context) == 'true'
    mask_only_pointcloud = LaunchConfiguration('mask_only_pointcloud').perform(context) == 'true'
    return_color = LaunchConfiguration('return_color').perform(context) == 'true'
    pointcloud_color = LaunchConfiguration('pointcloud_color').perform(context)
    pointcloud_color_format = LaunchConfiguration('pointcloud_color_format').perform(context)
    publish_force_fields = LaunchConfiguration('publish_force_fields').perform(context) == 'true'
    force_mapping = LaunchConfiguration('force_mapping').perform(context)
    height_threshold = float(LaunchConfiguration('height_threshold').perform(context))
    force_field_scale = float(LaunchConfiguration('force_field_scale').perform(context))
    force_field_baseline = LaunchConfiguration('force_field_baseline').perform(context) == 'true'
    
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

        node_parameters = {
            "serial": serial,
            "sensors_root": sensors_root,
            "mode": mode,
            "contact_mode": contact_mode,
            "model_device": model_device,
            "use_mask": use_mask,
            "refine_mask": refine_mask,
            "relative": relative,
            "relative_scale": 1.0,
            "mask_only_pointcloud": mask_only_pointcloud,
            "return_color": return_color,
            "color_dist_threshold": 15,
            "height_threshold": height_threshold,
            "topic": topic_name,
            "rate": rate,
            "verbose": True,
            "enable_force": enable_force,
            "temporal_stride": temporal_stride,
            "pointcloud_color": pointcloud_color,
            "pointcloud_color_format": pointcloud_color_format,
            "publish_force_fields": publish_force_fields,
            "force_mapping": force_mapping,
            "force_field_scale": force_field_scale,
            "force_field_baseline": force_field_baseline,
        }
        # Do not pass empty array params; ROS2 launch cannot infer type from [] and rejects with tuple () error.
        if outputs:
            node_parameters["outputs"] = outputs
        
        node = Node(
            package="vistac_sdk",  # Update this to match your actual ROS2 package name
            executable="tactile_streamer_node",
            name=f"tactile_streamer_{serial}",
            output="screen",
            parameters=[node_parameters],
        )
        nodes.append(node)
    
    return nodes
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
        DeclareLaunchArgument(
            'enable_force',
            default_value='false',
            description='Enable force estimation using Sparsh models (requires models to be downloaded)'
        ),
        DeclareLaunchArgument(
            'temporal_stride',
            default_value='5',
            description='Temporal stride for force estimation (number of frames between temporal pairs)'
        ),
        DeclareLaunchArgument(
            'outputs',
            default_value='',
            description='Comma-separated list of outputs (e.g., depth,force_field,force_vector). Overrides mode if specified.'
        ),
        DeclareLaunchArgument(
            'refine_mask',
            default_value='true',
            description='Refine contact mask edges for smoother boundaries'
        ),
        DeclareLaunchArgument(
            'relative',
            default_value='false',
            description='Use relative depth measurement instead of absolute'
        ),
        DeclareLaunchArgument(
            'mask_only_pointcloud',
            default_value='false',
            description='Only include masked (contact) region in pointcloud output'
        ),
        DeclareLaunchArgument(
            'return_color',
            default_value='false',
            description='Include RGB color information in pointcloud (creates PointCloud2 with color fields)'
        ),
        DeclareLaunchArgument(
            'pointcloud_color',
            default_value='none',
            description='Source for pointcloud color: none|image|force'
        ),
        DeclareLaunchArgument(
            'pointcloud_color_format',
            default_value='rgb_packed',
            description='PointCloud color encoding: rgb_packed|r_g_b'
        ),
        DeclareLaunchArgument(
            'publish_force_fields',
            default_value='false',
            description='Include per-point fx,fy,fz fields in PointCloud2'
        ),
        DeclareLaunchArgument(
            'force_mapping',
            default_value='nearest',
            description='Force -> point mapping: nearest|bilinear'
        ),
        DeclareLaunchArgument(
            'height_threshold',
            default_value='0.2',
            description='Height threshold in mm for contact detection'
        ),
        # New: force_field runtime controls
        DeclareLaunchArgument(
            'force_field_scale',
            default_value='1.0',
            description='Global scale applied to force_field outputs (SDK-wide)'
        ),
        DeclareLaunchArgument(
            'force_field_baseline',
            default_value='false',
            description='Enable runtime per-pixel baseline subtraction for force_field'
        ),
        
        # Use OpaqueFunction to handle dynamic node creation
        OpaqueFunction(function=launch_setup)
    ])