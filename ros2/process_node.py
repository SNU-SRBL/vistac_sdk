#!/usr/bin/env python3
"""Process node: reads raw SHM, runs ProcessingEngine in-process, publishes ROS.

ProcessingEngine handles SHM reading, corruption filter, background collection,
and TactileProcessor management — all pure Python, no ROS.

This node reads frames from SHM, feeds the frame to the engine's processor,
polls results, and publishes depth/pc/force.

Raw frames are published by a separate raw_bridge_node which reads the same
SHM blocks independently.

Topics (published per serial):
  /tactile/{serial}/depth       — Image (mono8)
  /tactile/{serial}/gradient    — Image (32FC2)
  /tactile/{serial}/pointcloud  — PointCloud2
  /tactile/{serial}/force_field — Image (32FC3)
"""

import time
from typing import Dict

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# BestEffort QoS — fire-and-forget, no ACK/retransmission overhead
_BE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
import numpy as np

from vistac_sdk.processing_engine import ProcessingEngine


class TactileProcessNode(Node):
    """ROS publisher: reads raw from SHM, feeds ProcessingEngine, publishes."""

    def __init__(self):
        super().__init__('tactile_process_node')

        # ---- Declare parameters ----
        self.declare_parameter(
            'serials', value=[''],
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY))
        self.declare_parameter('sensors_root', '../sensors')
        self.declare_parameter('mode', 'depth')
        self.declare_parameter('model_device', 'cuda')
        self.declare_parameter('enable_force', False)
        self.declare_parameter('temporal_stride', 5)
        self.declare_parameter(
            'outputs', value=[''],
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY))
        self.declare_parameter('use_mask', True)
        self.declare_parameter('refine_mask', True)
        self.declare_parameter('relative', True)
        self.declare_parameter('relative_scale', 0.5)
        self.declare_parameter('mask_only_pointcloud', False)
        self.declare_parameter('point_sample_mm', 0.0)
        self.declare_parameter('contact_mode', 'standard')
        self.declare_parameter('rate', 60.0)
        self.declare_parameter('force_field_scale', 1.0)
        self.declare_parameter('force_field_baseline', False)

        serials_raw = self.get_parameter('serials').value
        self._serials: list = (
            [] if not serials_raw or serials_raw == ['']
            else list(serials_raw))

        sensors_root = self.get_parameter('sensors_root').value
        mode = self.get_parameter('mode').value
        model_device = self.get_parameter('model_device').value
        enable_force = self.get_parameter('enable_force').value
        temporal_stride = self.get_parameter('temporal_stride').value
        outputs_param = [s for s in
                         self.get_parameter('outputs').value if s]
        use_mask = self.get_parameter('use_mask').value
        refine_mask = self.get_parameter('refine_mask').value
        relative = self.get_parameter('relative').value
        relative_scale = self.get_parameter('relative_scale').value
        mask_only_pointcloud = self.get_parameter(
            'mask_only_pointcloud').value
        point_sample_mm = self.get_parameter('point_sample_mm').value
        contact_mode = self.get_parameter('contact_mode').value
        rate = self.get_parameter('rate').value
        force_field_scale = self.get_parameter(
            'force_field_scale').value
        force_field_baseline = bool(
            self.get_parameter('force_field_baseline').value)

        if outputs_param:
            outputs = list(outputs_param)
        else:
            mode_map = {
                'depth': ['depth'],
                'gradient': ['gradient'],
                'pointcloud': ['pointcloud'],
                'force_field': ['force_field'],
                'force_vector': ['force_vector'],
                'pointcloud_force': ['pointcloud', 'force_field', 'mask'],
            }
            outputs = mode_map.get(mode, ['depth'])

        # ---- ProcessingEngine (in-process, pure Python) ----
        self.get_logger().info('Creating ProcessingEngine...')
        self._engine = ProcessingEngine(
            serials=self._serials,
            sensors_root=sensors_root,
            model_device=model_device,
            enable_force=enable_force,
            temporal_stride=temporal_stride,
            outputs=outputs,
            use_mask=use_mask,
            refine_mask=refine_mask,
            relative=relative,
            relative_scale=relative_scale,
            mask_only_pointcloud=mask_only_pointcloud,
            point_sample_mm=point_sample_mm,
            contact_mode=contact_mode,
            force_field_scale=force_field_scale,
            force_field_baseline=force_field_baseline,
        )

        active_sensors = [s for s in self._serials
                          if s in self._engine._shms]
        if not active_sensors:
            self.get_logger().fatal('No sensors initialized')
            raise RuntimeError('No sensors initialized')

        # Sequential background collection
        self.get_logger().info('Collecting backgrounds...')
        for serial in active_sensors:
            self._engine.collect_background(serial)

        # ---- Publishers (BestEffort QoS) ----
        self.output_publishers: Dict[str, dict] = {}
        for serial in active_sensors:
            base_topic = f'tactile/{serial}'
            self.output_publishers[serial] = {}
            for output in outputs:
                if output == 'depth':
                    self.output_publishers[serial]['depth'] = \
                        self.create_publisher(
                            Image, f'{base_topic}/depth', _BE_QOS)
                elif output == 'gradient':
                    self.output_publishers[serial]['gradient'] = \
                        self.create_publisher(
                            Image, f'{base_topic}/gradient', _BE_QOS)
                elif output == 'pointcloud':
                    self.output_publishers[serial]['pointcloud'] = \
                        self.create_publisher(
                            PointCloud2, f'{base_topic}/pointcloud',
                            _BE_QOS)
                elif output == 'force_field':
                    self.output_publishers[serial]['force_field'] = \
                        self.create_publisher(
                            Image, f'{base_topic}/force_field', _BE_QOS)
                    self.output_publishers[serial]['force_field_viz'] = \
                        self.create_publisher(
                            Image, f'{base_topic}/force_field_viz',
                            _BE_QOS)
                elif output == 'force_vector':
                    self.output_publishers[serial]['force_vector'] = \
                        self.create_publisher(
                            WrenchStamped,
                            f'{base_topic}/force_vector', _BE_QOS)
                elif output == 'raw':
                    self.output_publishers[serial]['raw'] = \
                        self.create_publisher(
                            Image, f'{base_topic}/raw', _BE_QOS)

        # ---- Per-sensor timers (parallel via ReentrantCallbackGroup + MultiThreadedExecutor) ----
        self._sensor_timers: Dict[str, object] = {}
        for serial in active_sensors:
            cbg = ReentrantCallbackGroup()
            self._sensor_timers[serial] = self.create_timer(
                1.0 / rate,
                lambda s=serial: self._handle_sensor(s),
                callback_group=cbg,
            )

        self.get_logger().info(
            f'Process node ready for {len(active_sensors)} sensors '
            f'({", ".join(active_sensors)}) on {model_device}')

    # ------------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------------

    def _handle_sensor(self, serial: str):
        """Read frame from engine, feed processor, publish results.
        Raw frames are published by raw_bridge_node."""
        bgr = self._engine.read_frame(serial)
        if bgr is not None:
            self._engine.feed_frame(serial, bgr)

        result = self._engine.get_result(serial)
        if result:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = f'tactile_{serial}'
            self._publish_results(serial, result, header)

    # ------------------------------------------------------------------
    # Result publishing
    # ------------------------------------------------------------------

    def _publish_results(
            self, serial: str, result: dict, header: Header):
        """Publish all results for one sensor."""
        pubs = self.output_publishers.get(serial, {})
        for output_name, pub_data in result.items():
            if output_name not in pubs:
                continue
            publisher = pubs[output_name]
            if pub_data is None:
                continue

            if output_name == 'depth':
                if len(pub_data.shape) == 2:
                    msg = self._cv2_to_imgmsg(
                        pub_data, encoding='mono8')
                    msg.header = header
                    publisher.publish(msg)

            elif output_name == 'gradient':
                if (len(pub_data.shape) == 3
                        and pub_data.shape[2] == 2):
                    msg = self._cv2_to_imgmsg(
                        pub_data.astype(np.float32),
                        encoding='32FC2')
                    msg.header = header
                    publisher.publish(msg)

            elif output_name == 'pointcloud':
                colors = result.get('pointcloud_colors')
                forces = result.get('pointcloud_forces')
                msg = self.create_pointcloud2_msg(
                    pub_data, header,
                    colors=colors, forces=forces)
                if msg is not None:
                    publisher.publish(msg)

            elif output_name == 'force_field':
                if isinstance(pub_data, dict):
                    normal = pub_data.get('normal')
                    shear = pub_data.get('shear')
                    if normal is not None and shear is not None:
                        force_rgb = np.zeros(
                            (normal.shape[0], normal.shape[1], 3),
                            dtype=np.float32)
                        force_rgb[:, :, 0] = shear[:, :, 0]
                        force_rgb[:, :, 1] = shear[:, :, 1]
                        force_rgb[:, :, 2] = normal
                        msg = self._cv2_to_imgmsg(
                            force_rgb, encoding='32FC3')
                        msg.header = header
                        publisher.publish(msg)

                        viz_pub = pubs.get('force_field_viz')
                        if viz_pub is not None:
                            from vistac_sdk.viz_utils import \
                                force_field_to_rgb
                            rgb8 = force_field_to_rgb(normal, shear)
                            viz_msg = self._cv2_to_imgmsg(
                                rgb8, encoding='rgb8')
                            viz_msg.header = header
                            viz_pub.publish(viz_msg)

            elif output_name == 'force_vector':
                if isinstance(pub_data, dict):
                    msg = WrenchStamped()
                    msg.header = header
                    msg.wrench.force.x = float(
                        pub_data.get('fx', 0.0))
                    msg.wrench.force.y = float(
                        pub_data.get('fy', 0.0))
                    msg.wrench.force.z = float(
                        pub_data.get('fz', 0.0))
                    publisher.publish(msg)

    # ------------------------------------------------------------------
    # Message conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cv2_to_imgmsg(arr: np.ndarray, encoding: str) -> Image:
        """numpy -> ROS Image message."""
        msg = Image()
        if arr.ndim == 2:
            msg.height, msg.width = arr.shape
        else:
            msg.height, msg.width = arr.shape[0], arr.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = arr.shape[1] * arr.dtype.itemsize * (
            1 if arr.ndim == 2 else arr.shape[2])
        msg.data = np.ascontiguousarray(arr).tobytes()
        return msg

    @staticmethod
    def create_pointcloud2_msg(
            points: np.ndarray, header: Header,
            colors: np.ndarray = None,
            forces: np.ndarray = None) -> PointCloud2:
        """Create PointCloud2 from point array with optional color/force."""
        import struct as _struct
        if points is None or len(points) == 0:
            return None
        pts = points.astype(np.float32)
        n = len(pts)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32,
                       count=1),
        ]
        offset = 12
        has_rgb = colors is not None and len(colors) == n
        has_f = forces is not None and len(forces) == n
        if has_rgb:
            rgb_f32 = np.clip(colors.astype(np.float32), 0.0, 1.0)
            fields.append(PointField(
                name='rgb', offset=offset,
                datatype=PointField.FLOAT32, count=3))
            offset += 12
        if has_f:
            f_f32 = forces.astype(np.float32)
            fields.append(PointField(
                name='force', offset=offset,
                datatype=PointField.FLOAT32, count=3))
            offset += 12

        data_list = [pts.tobytes()]
        if has_rgb:
            data_list.append(rgb_f32.tobytes())
        if has_f:
            data_list.append(f_f32.tobytes())

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = n
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = offset
        msg.row_step = offset * n
        msg.is_dense = True
        msg.data = b''.join(data_list)
        return msg

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        self._engine.shutdown()
        self.output_publishers.clear()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TactileProcessNode()
    num_threads = max(len(node._serials), 1)
    executor = MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
