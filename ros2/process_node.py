#!/usr/bin/env python3
"""Process node: subscribes raw camera images, runs TactileProcessor, publishes depth/pointcloud/force.

Runs in its own process (separate GIL from camera node).
Subscribes to /tactile/{serial}/raw published by camera_node.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import pathlib
import time

from vistac_sdk.tactile_processor import TactileProcessor
from vistac_sdk.utils import load_config
from vistac_sdk.viz_utils import force_field_to_rgb

# Background collection constants
BG_COLLECTION_FRAMES = 10
BG_COLLECTION_DELAY_SEC = 0.2


class TactileProcessNode(Node):
    """Subscribes to /tactile/{serial}/raw, runs depth/force estimation,
    and publishes results (depth, pointcloud, force_field, force_vector).

    Topics (published):
      /tactile/{serial}/depth
      /tactile/{serial}/gradient
      /tactile/{serial}/pointcloud
      /tactile/{serial}/force_field
      /tactile/{serial}/force_field_viz
      /tactile/{serial}/force_vector
    """

    def __init__(self):
        super().__init__('tactile_process_node')

        # Declare all parameters
        self.declare_parameter('serial', 'YOUR_SENSOR_SERIAL')
        self.declare_parameter('sensors_root', '../sensors')
        self.declare_parameter('mode', 'depth')
        self.declare_parameter('model_device', 'cuda')
        self.declare_parameter('enable_force', False)
        self.declare_parameter('temporal_stride', 5)
        self.declare_parameter(
            'outputs',
            value=[''],
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

        serial = self.get_parameter('serial').value
        sensors_root = self.get_parameter('sensors_root').value
        mode = self.get_parameter('mode').value
        model_device = self.get_parameter('model_device').value
        enable_force = self.get_parameter('enable_force').value
        temporal_stride = self.get_parameter('temporal_stride').value
        outputs_param = [s for s in self.get_parameter('outputs').value if s]
        use_mask = self.get_parameter('use_mask').value
        refine_mask = self.get_parameter('refine_mask').value
        relative = self.get_parameter('relative').value
        relative_scale = self.get_parameter('relative_scale').value
        mask_only_pointcloud = self.get_parameter(
            'mask_only_pointcloud').value
        point_sample_mm = self.get_parameter('point_sample_mm').value
        contact_mode = self.get_parameter('contact_mode').value
        rate = self.get_parameter('rate').value
        self.force_field_scale = self.get_parameter(
            'force_field_scale').value

        # Determine outputs based on parameter or mode
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
        self.outputs = outputs

        self._depth_kwargs = {
            'use_mask': use_mask,
            'refine_mask': refine_mask,
            'relative': relative,
            'relative_scale': relative_scale,
            'mask_only_pointcloud': mask_only_pointcloud,
            'point_sample_mm': point_sample_mm,
        }

        # Load sensor config
        root = pathlib.Path(sensors_root)
        config_path = str(root / serial / f'{serial}.yaml')
        config = load_config(config_path=config_path)
        ppmm = config['ppmm']

        # Resolve model paths
        model_path = str(root / serial / 'model' / 'nnmodel.pth')
        force_encoder_path = str(
            pathlib.Path(sensors_root).parent / 'models' / 'sparsh_dino_base_encoder.ckpt')
        force_decoder_path = str(
            pathlib.Path(sensors_root).parent / 'models' / 'sparsh_digit_forcefield_decoder.pth')

        # Read force config
        force_cfg = config.get('force', {}) or {}
        force_vector_scale_cfg = force_cfg.get(
            'force_vector_scale', [1.0, 1.0, 1.0])
        force_field_baseline_flag = bool(
            self.get_parameter('force_field_baseline').value)

        # Initialize TactileProcessor
        depth_outputs = {'depth', 'gradient', 'pointcloud', 'mask'}
        enable_depth = any(o in depth_outputs for o in outputs)
        force_outputs = {'force_field', 'force_vector'}
        enable_force_est = enable_force or any(
            o in force_outputs for o in outputs)

        self.processor = TactileProcessor(
            model_path=model_path if enable_depth else None,
            enable_depth=enable_depth,
            enable_force=enable_force_est,
            force_encoder_path=force_encoder_path,
            force_decoder_path=force_decoder_path,
            temporal_stride=temporal_stride,
            bg_offset=0.5,
            device=model_device,
            ppmm=ppmm,
            contact_mode=contact_mode,
            force_field_baseline=force_field_baseline_flag,
            force_vector_scale=force_vector_scale_cfg,
        )

        # Background collection state
        self._bg_buffer = []
        self._bg_frame_count = 0
        self._bg_last_ts = 0.0
        self._bg_done = False
        self.ppmm = ppmm

        # Subscribe to raw camera frames
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, f'tactile/{serial}/raw',
            self.raw_callback, 10)

        # Create publishers
        self.output_publishers = {}
        base_topic = f'tactile/{serial}'
        for output in outputs:
            if output == 'depth':
                self.output_publishers['depth'] = self.create_publisher(
                    Image, f'{base_topic}/depth', 10)
            elif output == 'gradient':
                self.output_publishers['gradient'] = self.create_publisher(
                    Image, f'{base_topic}/gradient', 10)
            elif output == 'pointcloud':
                self.output_publishers['pointcloud'] = \
                    self.create_publisher(
                        PointCloud2, f'{base_topic}/pointcloud', 10)
            elif output == 'force_field':
                self.output_publishers['force_field'] = \
                    self.create_publisher(
                        Image, f'{base_topic}/force_field', 10)
                self.output_publishers['force_field_viz'] = \
                    self.create_publisher(
                        Image, f'{base_topic}/force_field_viz', 10)
            elif output == 'force_vector':
                self.output_publishers['force_vector'] = \
                    self.create_publisher(
                        WrenchStamped, f'{base_topic}/force_vector', 10)
            elif output == 'raw':
                # Also support raw pass-through if configured
                self.output_publishers['raw'] = self.create_publisher(
                    Image, f'{base_topic}/raw', 10)

        self.timer = self.create_timer(1.0 / rate, self.timer_callback)
        self.get_logger().info(
            f'Process node ready for {serial} ({model_device})')

    def _raw_to_bgr(self, msg):
        """Convert ROS Image (rgb8) -> BGR numpy array."""
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='rgb8')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception:
            frame = np.frombuffer(
                msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
            frame = frame[..., ::-1]
        return frame

    def raw_callback(self, msg):
        """Handle incoming raw camera frame for background + processing."""
        frame = self._raw_to_bgr(msg)

        # Background collection (first BG_COLLECTION_FRAMES frames)
        if not self._bg_done:
            now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if (len(self._bg_buffer) == 0 or
                    now - self._bg_last_ts >= BG_COLLECTION_DELAY_SEC):
                self._bg_buffer.append(frame)
                self._bg_last_ts = now
                if len(self._bg_buffer) >= BG_COLLECTION_FRAMES:
                    bg_image = np.mean(
                        self._bg_buffer, axis=0).astype(np.uint8)
                    self.processor.load_background(bg_image)
                    self.processor.start_thread(
                        outputs=self.outputs,
                        ppmm=self.ppmm,
                        **self._depth_kwargs)
                    self._bg_done = True
                    self.get_logger().info(
                        f'Background collected '
                        f'({len(self._bg_buffer)} frames)')
            return

        # Route frame to processor
        ts = time.time()
        self.processor.set_input_frame(frame, ts)

    # --- Message conversion helpers ---

    @staticmethod
    def _cv2_to_imgmsg(arr: np.ndarray, encoding: str) -> Image:
        """numpy -> ROS Image message (cv_bridge replacement)."""
        msg = Image()
        if arr.ndim == 2:
            msg.height, msg.width = arr.shape
        else:
            msg.height, msg.width, _ = arr.shape
        msg.encoding = encoding
        msg.is_bigendian = False
        step = arr.strides[0]
        if step <= 0:
            step = arr.shape[1] * arr.itemsize * (
                arr.shape[2] if arr.ndim == 3 else 1)
        msg.step = step
        msg.data = np.ascontiguousarray(arr).tobytes()
        return msg

    def create_pointcloud2_msg(
            self, points, header,
            colors=None, color_format='rgb_packed', forces=None):
        """Create PointCloud2 message from numpy point array."""
        if points.shape[1] != 3:
            return None
        n = len(points)
        fields = [
            PointField(name='x', offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,
                       datatype=PointField.FLOAT32, count=1),
        ]
        columns = [points.astype(np.float32)]
        if colors is not None:
            cols = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
            if color_format == 'rgb_packed':
                r = cols[:, 0].astype(np.uint32)
                g = cols[:, 1].astype(np.uint32)
                b = cols[:, 2].astype(np.uint32)
                rgb_uint32 = (r << 16) | (g << 8) | b
                fields.append(PointField(
                    name='rgb', offset=12,
                    datatype=PointField.FLOAT32, count=1))
                columns.append(
                    rgb_uint32.view(np.float32).reshape(-1, 1))
                point_step = 16
            else:
                fields += [
                    PointField(
                        name='r', offset=12,
                        datatype=PointField.FLOAT32, count=1),
                    PointField(
                        name='g', offset=16,
                        datatype=PointField.FLOAT32, count=1),
                    PointField(
                        name='b', offset=20,
                        datatype=PointField.FLOAT32, count=1),
                ]
                columns.append(cols.astype(np.float32))
                point_step = 24
        else:
            point_step = 12

        if forces is not None:
            offset = point_step
            fields += [
                PointField(
                    name='fx', offset=offset,
                    datatype=PointField.FLOAT32, count=1),
                PointField(
                    name='fy', offset=offset + 4,
                    datatype=PointField.FLOAT32, count=1),
                PointField(
                    name='fz', offset=offset + 8,
                    datatype=PointField.FLOAT32, count=1),
            ]
            columns.append(forces.astype(np.float32))
            point_step = offset + 12

        combined = np.hstack(columns)
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = n
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * n
        msg.data = combined.tobytes()
        msg.is_dense = True
        return msg

    # --- Force field canonicalization ---

    def _canonicalize_force_field(self, result, frame):
        """Apply force_field scaling and recompute pointcloud colors."""
        if (result is None or 'force_field' not in result or
                result['force_field'] is None):
            return result
        ff = result['force_field']
        try:
            normal_arr = np.asarray(ff['normal']).astype(np.float32)
            shear_arr = np.asarray(ff['shear']).astype(np.float32)
        except Exception:
            return result

        normal_vis = np.clip(normal_arr, 0.0, 1.0)
        shear_vis = np.clip(shear_arr, -1.0, 1.0)

        if self.force_field_scale != 1.0:
            s = float(self.force_field_scale)
            normal_vis = (normal_vis.astype(np.float64) * s).astype(
                np.float32)
            shear_vis = (shear_vis.astype(np.float64) * s).astype(
                np.float32)

        ff['normal'] = normal_vis
        ff['shear'] = shear_vis
        result['force_field'] = ff

        # Recompute pointcloud colors/forces
        if frame is not None:
            try:
                pc = result.get('pointcloud')
                if pc is not None:
                    force_rgb = force_field_to_rgb(
                        normal_vis, shear_vis)
                    th, tw = frame.shape[0], frame.shape[1]
                    fh, fw = force_rgb.shape[:2]
                    if (fh, fw) != (th, tw):
                        force_rgb = cv2.resize(
                            force_rgb, (tw, th),
                            interpolation=cv2.INTER_NEAREST)
                    colors_flat = force_rgb.reshape(-1, 3) / 255.0
                    mask = result.get('mask')
                    if (mask is not None
                            and pc.shape[0] != (th * tw)):
                        mask_flat = mask.ravel()
                        if mask_flat.shape[0] == th * tw:
                            colors_flat = colors_flat[mask_flat]
                    result['pointcloud_colors'] = colors_flat

                    fx_img = shear_vis[..., 0]
                    fy_img = shear_vis[..., 1]
                    fz_img = normal_vis
                    if (fx_img.shape[0], fx_img.shape[1]) != (th, tw):
                        fx_img = cv2.resize(
                            fx_img, (tw, th),
                            interpolation=cv2.INTER_NEAREST)
                        fy_img = cv2.resize(
                            fy_img, (tw, th),
                            interpolation=cv2.INTER_NEAREST)
                        fz_img = cv2.resize(
                            fz_img, (tw, th),
                            interpolation=cv2.INTER_NEAREST)
                    fx_flat = fx_img.reshape(-1)
                    fy_flat = fy_img.reshape(-1)
                    fz_flat = fz_img.reshape(-1)
                    if (mask is not None
                            and pc.shape[0] != (th * tw)):
                        fx_flat = fx_flat[mask_flat]
                        fy_flat = fy_flat[mask_flat]
                        fz_flat = fz_flat[mask_flat]
                    result['pointcloud_forces'] = np.stack(
                        [fx_flat, fy_flat, fz_flat], axis=1)
            except Exception:
                self.get_logger().warn(
                    'Failed to recompute pointcloud colors/forces',
                    throttle_duration_sec=10.0)
        return result

    # --- Publish loop ---

    def timer_callback(self):
        """Publish latest processor results."""
        result = self.processor.get_latest_result()
        if not result:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = (
            f'tactile_{self.get_parameter("serial").value}')

        for output_name, pub_data in result.items():
            if output_name not in self.output_publishers:
                continue
            publisher = self.output_publishers[output_name]
            if pub_data is None:
                continue

            if output_name == 'depth':
                if len(pub_data.shape) == 2:
                    msg = self._cv2_to_imgmsg(pub_data, encoding='mono8')
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

                        # viz topic
                        viz_pub = self.output_publishers.get(
                            'force_field_viz')
                        if viz_pub is not None:
                            rgb8 = force_field_to_rgb(
                                normal, shear)
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

    def destroy_node(self):
        self.processor.stop_thread()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TactileProcessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
