import os
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix

'''
This launch file starts camera, raw_bridge, and depth nodes for each
DIGIT sensor. Three node types per sensor set:
  1. camera_shm: reads DIGIT at 60Hz, writes BGR frames to SharedMemory
     (plain Python, no rclpy)
  2. raw_bridge_node: reads SHM, publishes /tactile/{serial}/raw on DDS
     (rclpy, dedicated process, no depth model)
  3. process_node: reads SHM, runs depth/force, publishes
     depth/pc/force (rclpy, ProcessingEngine in-process)

camera_shm is a plain Python process (NO rclpy/DDS) — it writes RGB frames
+ metadata to SharedMemory. raw_bridge_node publishes raw frames to ROS
topics; process_node handles depth/force computation. Separating raw into
its own node removes the 230KB/frame DDS bottleneck from depth processing.

Usage Examples:
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py mode:=depth
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py mode:=pointcloud model_device:=cpu
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py enable_force:=true mode:=force_vector
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py outputs:=depth,force_field,force_vector
'''


def launch_setup(context, *args, **kwargs):
    # ── Pre-launch cleanup: kill stale processes, clear SHM ──
    # SIGINT (-2) lets rclpy shutdown cleanly and FastRTPS clean up SHM.
    # Do NOT use pkill -9 (SIGKILL) — corrupts DDS SHM.
    # Do NOT use plain pkill (SIGTERM) — ROS2 nodes need SIGINT for rclpy.
    # /dev/shm/tactile_* is our app SHM (unrelated to DDS); cleanup stays.
    # /dev/shm/fastdds* is DDS SHM safety net for crash recovery.
    subprocess.run(
        "pkill -2 -f 'camera_shm|process_node|raw_bridge' 2>/dev/null; "
        "rm -f /dev/shm/tactile_* /dev/shm/fastdds* 2>/dev/null; "
        "sleep 1",
        shell=True, timeout=5)

    # Get launch arguments
    sensors_root = LaunchConfiguration('sensors_root').perform(context)
    mode = LaunchConfiguration('mode').perform(context)
    model_device = LaunchConfiguration('model_device').perform(context)
    use_mask = LaunchConfiguration('use_mask').perform(context) == 'true'
    rate = float(LaunchConfiguration('rate').perform(context))
    contact_mode = LaunchConfiguration('contact_mode').perform(context)
    enable_force = LaunchConfiguration(
        'enable_force').perform(context) == 'true'
    temporal_stride = int(LaunchConfiguration(
        'temporal_stride').perform(context))
    outputs_str = LaunchConfiguration('outputs').perform(context)
    outputs = [s.strip()
               for s in outputs_str.split(',')] if outputs_str else []

    # Depth-specific parameters
    refine_mask = LaunchConfiguration(
        'refine_mask').perform(context) == 'true'
    relative = LaunchConfiguration('relative').perform(context) == 'true'
    mask_only_pointcloud = LaunchConfiguration(
        'mask_only_pointcloud').perform(context) == 'true'
    height_threshold = float(LaunchConfiguration(
        'height_threshold').perform(context))
    force_field_scale = float(LaunchConfiguration(
        'force_field_scale').perform(context))
    force_field_baseline = LaunchConfiguration(
        'force_field_baseline').perform(context) == 'true'
    point_sample_mm = float(LaunchConfiguration(
        'point_sample_mm').perform(context))

    # Auto-discover sensors from sensors_root directory
    sensors = []
    if os.path.exists(sensors_root):
        for item in os.listdir(sensors_root):
            sensor_path = os.path.join(sensors_root, item)
            config_file = os.path.join(sensor_path, f"{item}.yaml")
            if os.path.isdir(sensor_path) and os.path.exists(config_file):
                sensors.append(item)

    # Fallback to hardcoded list
    if not sensors:
        sensors = ["D21275", "D21273", "D21242", "D21119"]

    # Path to executables
    pkg_prefix = get_package_prefix('digit_sdk')
    camera_shm_exe = os.path.join(pkg_prefix, 'lib', 'digit_sdk', 'camera_shm')

    nodes = []
    for i, serial in enumerate(sensors):
        # --- CAMERA PROCESS (plain Python, no rclpy) ---
        nodes.append(ExecuteProcess(
            cmd=[camera_shm_exe, '--serial', serial,
                 '--sensors-root', sensors_root,
                 '--cpu-core', str(i)],
            name=f"camera_{serial}",
            output="screen",
        ))

    # --- PUBLISHER NODE (rclpy, ProcessingEngine in-process) ---
    process_params = {
        "serials": sensors,
        "sensors_root": sensors_root,
        "mode": mode,
        "contact_mode": contact_mode,
        "model_device": model_device,
        "use_mask": use_mask,
        "refine_mask": refine_mask,
        "relative": relative,
        "relative_scale": 1.0,
        "mask_only_pointcloud": mask_only_pointcloud,
        "color_dist_threshold": 15,
        "height_threshold": height_threshold,
        "rate": rate,
        "enable_force": enable_force,
        "temporal_stride": temporal_stride,
        "force_field_scale": force_field_scale,
        "force_field_baseline": force_field_baseline,
        "point_sample_mm": point_sample_mm,
    }
    if outputs:
        process_params["outputs"] = outputs

    nodes.append(Node(
        package="digit_sdk",
        executable="process_node",
        name="tactile_process_node",
        output="screen",
        parameters=[process_params],
    ))

    # --- RAW BRIDGE NODES (one per sensor — independent GIL) ---
    for i, serial in enumerate(sensors):
        nodes.append(Node(
            package="digit_sdk",
            executable="raw_bridge_node",
            name=f"raw_bridge_{serial}",
            output="screen",
            parameters=[{"serial": serial, "rate": rate, "cpu_core": i + 4}],
        ))

    return nodes


def generate_launch_description():
    # Resolve sensors_root for both source and install trees.
    _launch_dir = os.path.dirname(os.path.abspath(__file__))
    _src_path = os.path.abspath(
        os.path.join(_launch_dir, '..', '..', 'sensors'))
    _inst_path = os.path.abspath(
        os.path.join(_launch_dir, '..', 'sensors'))
    default_sensors_root = _src_path if os.path.exists(
        _src_path) else _inst_path

    return LaunchDescription([
        DeclareLaunchArgument(
            'sensors_root',
            default_value=default_sensors_root,
            description='Root directory for sensor configurations'),
        DeclareLaunchArgument(
            'mode',
            default_value='depth',
            description='Processing mode: depth, gradient, pointcloud, force_field, force_vector'),
        DeclareLaunchArgument(
            'model_device',
            default_value='cuda',
            description='Device for model execution: cuda or cpu'),
        DeclareLaunchArgument(
            'use_mask',
            default_value='true',
            description='Whether to apply contact mask'),
        DeclareLaunchArgument(
            'rate',
            default_value='60.0',
            description='Publishing rate in Hz'),
        DeclareLaunchArgument(
            'contact_mode',
            default_value='standard',
            description='Contact detection mode: standard or flat'),
        DeclareLaunchArgument(
            'enable_force',
            default_value='false',
            description='Enable force estimation using Sparsh models'),
        DeclareLaunchArgument(
            'temporal_stride',
            default_value='5',
            description='Temporal stride for force estimation'),
        DeclareLaunchArgument(
            'outputs',
            default_value='',
            description='Comma-separated list of outputs (overrides mode)'),
        DeclareLaunchArgument(
            'refine_mask',
            default_value='true',
            description='Refine contact mask for smoother boundaries'),
        DeclareLaunchArgument(
            'relative',
            default_value='false',
            description='Use relative depth measurement'),
        DeclareLaunchArgument(
            'mask_only_pointcloud',
            default_value='false',
            description='Only include masked region in pointcloud'),
        DeclareLaunchArgument(
            'height_threshold',
            default_value='0.2',
            description='Height threshold in mm for contact detection'),
        DeclareLaunchArgument(
            'force_field_scale',
            default_value='1.0',
            description='Global scale for force_field outputs'),
        DeclareLaunchArgument(
            'force_field_baseline',
            default_value='false',
            description='Enable per-pixel baseline subtraction'),
        DeclareLaunchArgument(
            'point_sample_mm',
            default_value='0.0',
            description='Point spacing in mm for pointcloud subsampling'),

        OpaqueFunction(function=launch_setup),
    ])
