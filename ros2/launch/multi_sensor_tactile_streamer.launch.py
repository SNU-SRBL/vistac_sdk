import os
import subprocess
import yaml
from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix

# Centralised CPU affinity from system_config.yaml (fallback to defaults)
try:
    from core.config.config_mapper import map_tactile_cpu_affinity
except ImportError:
    map_tactile_cpu_affinity = None


_DEFAULT_TACTILE_AFFINITY = {
    "camera_shm": [0, 1, 2, 3],
    "raw_publisher": [4, 5, 10, 11],
    "surface_publisher": [8, 9],
    "force_publisher": [8, 9],
}

'''
This launch file starts camera, raw, pipeline, and publisher nodes for each
DIGIT sensor. Five node types per sensor set:
  1. camera_shm: reads DIGIT at 60Hz, writes BGR frames to SharedMemory
     (plain Python, no rclpy)
  2. raw_publisher: reads camera SHM, publishes /tactile/{serial}/raw on DDS
     (rclpy, dedicated process, no depth model)
  3. pipeline_node: reads camera SHM, runs depth/force GPU pipeline,
     writes surface/force SHM, publishes gradient inline
  4. surface_publisher: reads surface SHM, publishes depth + pointcloud
  5. force_publisher (if enable_force=true): reads force SHM, publishes force

Surface/force publishers have their own GIL, avoiding Fast-DDS publish blocking
that limited depth to ~30 Hz in the single-process architecture.

Usage Examples:
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py mode:=depth
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py mode:=pointcloud model_device:=cpu
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py enable_force:=true mode:=force_vector
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py outputs:=depth,force_field,force_vector
'''


def _list_to_affinity_string(cores: List[int]) -> str:
    """Convert a list of core indices to a compact string for ``os.sched_setaffinity``.

    Examples::
        [4,5,10,11] -> "4-5,10-11"
        [4,5]       -> "4-5"
        [2]         -> "2"
        [8,9]       -> "8-9"
    """
    if not cores:
        return ""
    sorted_cores = sorted(set(cores))
    parts = []
    start = sorted_cores[0]
    end = start
    for c in sorted_cores[1:]:
        if c == end + 1:
            end = c
        else:
            parts.append(f"{start}-{end}" if start != end else str(start))
            start = c
            end = c
    parts.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(parts)


def _get_tactile_affinity() -> dict:
    """Read tactile CPU affinity from central config, falling back to defaults."""
    ca = None
    if map_tactile_cpu_affinity is not None:
        try:
            ca = map_tactile_cpu_affinity()
        except Exception:
            pass
    if ca:
        return ca
    return dict(_DEFAULT_TACTILE_AFFINITY)


def launch_setup(context, *args, **kwargs):
    # ── Pre-launch cleanup: kill stale processes, clear SHM ──
    subprocess.run(
        "pkill -2 -f 'camera_shm|pipeline_node|raw_publisher|surface_publisher|force_publisher' 2>/dev/null; "
        "rm -f /dev/shm/tactile_* 2>/dev/null; "
        # DISABLED: rm /dev/shm/fastdds* breaks DDS discovery at scale (25+ participants, SHM port collision)
        # "rm -f /dev/shm/fastdds* 2>/dev/null; "
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

    # Auto-discover sensors
    sensors = []
    if os.path.exists(sensors_root):
        for item in os.listdir(sensors_root):
            sensor_path = os.path.join(sensors_root, item)
            config_file = os.path.join(sensor_path, f"{item}.yaml")
            if os.path.isdir(sensor_path) and os.path.exists(config_file):
                sensors.append(item)

    if not sensors:
        sensors = ["D21275", "D21273", "D21242", "D21119"]

    pkg_prefix = get_package_prefix('digit_sdk')
    camera_shm_exe = os.path.join(pkg_prefix, 'lib', 'digit_sdk', 'camera_shm')

    nodes = []

    # Read CPU affinity from central config (or fallback defaults)
    affinity = _get_tactile_affinity()
    aff_camera = affinity.get("camera_shm", [0, 1, 2, 3])
    aff_raw = _list_to_affinity_string(
        affinity.get("raw_publisher", [4, 5, 10, 11])
    )
    aff_surface = _list_to_affinity_string(
        affinity.get("surface_publisher", [8, 9])
    )
    aff_force = _list_to_affinity_string(
        affinity.get("force_publisher", [8, 9])
    )

    # --- CAMERA PROCESSES (plain Python, no rclpy) ---
    for i, serial in enumerate(sensors):
        cpu_affinity = str(aff_camera[i]) if i < len(aff_camera) else str(i)
        nodes.append(ExecuteProcess(
            cmd=[camera_shm_exe, '--serial', serial,
                 '--sensors-root', sensors_root,
                 '--cpu-affinity', cpu_affinity],
            name=f"camera_{serial}",
            output="screen",
        ))

    # --- PIPELINE NODE (rclpy, ProcessingEngine in-process, SHM writer) ---
    pipeline_params = {
        "serials": sensors,
        "sensors_root": sensors_root,
        "mode": mode,
        "contact_mode": contact_mode,
        "model_device": model_device,
        "use_mask": use_mask,
        "refine_mask": refine_mask,
        "relative": relative,
        "relative_scale": 1.0,
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
        pipeline_params["outputs"] = outputs

    nodes.append(Node(
        package="digit_sdk",
        executable="pipeline_node",
        name="pipeline_node",
        output="screen",
        parameters=[pipeline_params],
    ))

    # --- RAW PUBLISHERS (one per sensor) ---
    for i, serial in enumerate(sensors):
        nodes.append(Node(
            package="digit_sdk",
            executable="raw_publisher",
            name=f"raw_pub_{serial}",
            output="screen",
            parameters=[{"serial": serial, "rate": rate,
                         "cpu_affinity": aff_raw}],
        ))

    # --- SURFACE PUBLISHERS (one per sensor, own GIL) ---
    for serial in sensors:
        nodes.append(Node(
            package="digit_sdk",
            executable="surface_publisher",
            name=f"surface_pub_{serial}",
            output="screen",
            parameters=[{"serial": serial, "rate": rate,
                         "cpu_affinity": aff_surface}],
        ))

    # --- FORCE PUBLISHERS (one per sensor, only if force enabled) ---
    if enable_force:
        for serial in sensors:
            nodes.append(Node(
                package="digit_sdk",
                executable="force_publisher",
                name=f"force_pub_{serial}",
                output="screen",
                parameters=[{"serial": serial, "rate": 30.0,
                             "cpu_affinity": aff_force}],
            ))

    # --- DIGIT OPTICAL FRAME + BRIDGE (per sensor, from YAML) ---
    for serial in sensors:
        # Read optical frame params from per-sensor YAML
        sensor_dir = os.path.join(sensors_root, serial)
        sensor_yaml_path = os.path.join(sensor_dir, f"{serial}.yaml")
        optical = {}
        try:
            with open(sensor_yaml_path) as f:
                sensor_cfg = yaml.safe_load(f) or {}
            optical = sensor_cfg.get("optical_frame", {})
        except Exception:
            pass

        optical_rpy = [str(v) for v in optical.get("optical_rpy", [0.0, 0.0, 0.0])]
        optical_xyz = [str(v) for v in optical.get("optical_xyz", [0.0, 0.0, 0.0])]

        # Body → optical frame (from YAML)
        nodes.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"optical_{serial}",
            output="log",
            arguments=optical_xyz + optical_rpy + [
                f"tactile_{serial}_base_link",
                f"tactile_{serial}_optical_frame",
            ],
        ))

    return nodes


def generate_launch_description():
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
