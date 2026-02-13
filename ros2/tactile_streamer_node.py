#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import struct

from vistac_sdk.live_core import LiveTactileProcessor

'''
This ROS2 node streams tactile data from a sensor using the LiveReconstructor class.
It publishes the data as Image or PointCloud2 messages on specified topics.

Complete Parameters List:
- serial: Sensor serial number (default: 'YOUR_SENSOR_SERIAL')
- sensors_root: Root directory for sensor configurations (default: '../sensors') 
- mode: Processing mode - 'depth', 'gradient', 'pointcloud', 'force_field', 'force_vector' (default: 'depth')
- contact_mode: Contact detection mode - 'standard' or 'flat' (default: 'standard')
- model_device: Device for model execution - 'cuda' or 'cpu' (default: 'cuda')
- enable_force: Enable force estimation (default: False)
- temporal_stride: Temporal stride for force estimation (default: 5)
- outputs: Explicit list of outputs to compute (default: None, determined by mode)
- use_mask: Whether to apply contact mask (default: True)
- refine_mask: Whether to refine contact mask (default: True)
- relative: Whether to normalize depth to [0,1] (default: True)
- relative_scale: Scale factor for relative depth (default: 0.5)
- mask_only_pointcloud: If True, only output masked points for pointcloud (default: False)
- return_color: Whether to include color in pointcloud (default: False)
- color_dist_threshold: Color distance threshold for contact mask (default: 15)
- height_threshold: Height threshold for contact mask (default: 0.2)
- topic: Topic to publish the data (default: 'tactile/depth')
- rate: Publishing rate in Hz (default: 15.0)
- verbose: Whether to print verbose camera info (default: True)
'''

class TactileStreamerNode(Node):
    def __init__(self):
        super().__init__('tactile_streamer_node')
        
        # Declare all parameters with defaults
        self.declare_parameter('serial', 'YOUR_SENSOR_SERIAL')
        self.declare_parameter('sensors_root', '../sensors')
        self.declare_parameter('mode', 'depth')
        self.declare_parameter('contact_mode', 'standard') 
        self.declare_parameter('model_device', 'cuda')
        self.declare_parameter('enable_force', False)
        self.declare_parameter('temporal_stride', 5)
        self.declare_parameter('outputs', [])
        self.declare_parameter('use_mask', True)
        self.declare_parameter('refine_mask', True)
        self.declare_parameter('relative', True)
        self.declare_parameter('relative_scale', 0.5)
        self.declare_parameter('mask_only_pointcloud', False)
        self.declare_parameter('return_color', False)
        self.declare_parameter('pointcloud_color', 'none')  # none|image|force
        self.declare_parameter('pointcloud_color_format', 'rgb_packed')  # rgb_packed|r_g_b
        self.declare_parameter('publish_force_fields', False)
        self.declare_parameter('force_mapping', 'nearest')  # nearest|bilinear
        self.declare_parameter('color_dist_threshold', 15)
        self.declare_parameter('height_threshold', 0.2)
        # Make force_field options available to ROS users (applies SDK-wide)
        self.declare_parameter('force_field_scale', 1.0)
        self.declare_parameter('force_field_baseline', False)
        self.declare_parameter('topic', 'tactile/depth')
        self.declare_parameter('rate', 15.0)
        self.declare_parameter('verbose', True)

        # Get parameters
        serial = self.get_parameter('serial').get_parameter_value().string_value
        sensors_root = self.get_parameter('sensors_root').get_parameter_value().string_value
        mode = self.get_parameter('mode').get_parameter_value().string_value
        contact_mode = self.get_parameter('contact_mode').get_parameter_value().string_value
        model_device = self.get_parameter('model_device').get_parameter_value().string_value
        enable_force = self.get_parameter('enable_force').get_parameter_value().bool_value
        temporal_stride = self.get_parameter('temporal_stride').get_parameter_value().integer_value
        outputs_param = self.get_parameter('outputs').get_parameter_value().string_array_value
        use_mask = self.get_parameter('use_mask').get_parameter_value().bool_value
        refine_mask = self.get_parameter('refine_mask').get_parameter_value().bool_value
        relative = self.get_parameter('relative').get_parameter_value().bool_value
        relative_scale = self.get_parameter('relative_scale').get_parameter_value().double_value
        mask_only_pointcloud = self.get_parameter('mask_only_pointcloud').get_parameter_value().bool_value
        return_color = self.get_parameter('return_color').get_parameter_value().bool_value
        pointcloud_color = self.get_parameter('pointcloud_color').get_parameter_value().string_value
        pointcloud_color_format = self.get_parameter('pointcloud_color_format').get_parameter_value().string_value
        publish_force_fields = self.get_parameter('publish_force_fields').get_parameter_value().bool_value
        force_mapping = self.get_parameter('force_mapping').get_parameter_value().string_value
        color_dist_threshold = self.get_parameter('color_dist_threshold').get_parameter_value().integer_value
        height_threshold = self.get_parameter('height_threshold').get_parameter_value().double_value
        # ROS-level force_field options (passed into LiveTactileProcessor)
        force_field_scale = float(self.get_parameter('force_field_scale').get_parameter_value().double_value)
        force_field_baseline = self.get_parameter('force_field_baseline').get_parameter_value().bool_value
        topic = self.get_parameter('topic').get_parameter_value().string_value
        rate = self.get_parameter('rate').get_parameter_value().double_value
        verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        # Store mode and return_color for message handling
        self.mode = mode
        self.return_color = return_color
        self.pointcloud_color = pointcloud_color
        self.pointcloud_color_format = pointcloud_color_format
        self.publish_force_fields = publish_force_fields
        self.force_mapping = force_mapping

        # Determine outputs based on explicit parameter or mode
        if outputs_param:
            outputs = list(outputs_param)
        else:
            # Derive from mode
            if mode == 'depth':
                outputs = ['depth']
            elif mode == 'gradient':
                outputs = ['gradient']
            elif mode == 'pointcloud':
                outputs = ['pointcloud']
            elif mode == 'pointcloud_force':
                # Combined pointcloud colored by force_field
                outputs = ['pointcloud', 'force_field', 'mask']
                # Default to using force colors for this mode if unspecified
                if self.pointcloud_color == 'none':
                    self.pointcloud_color = 'force'
            elif mode == 'force_field':
                outputs = ['force_field']
            elif mode == 'force_vector':
                outputs = ['force_vector']
            else:
                outputs = ['depth']
        
        # Store outputs for publishing
        self.outputs = outputs

        # Initialize LiveTactileProcessor with all parameters
        try:
            self.processor = LiveTactileProcessor(
                serial=serial,
                sensors_root=sensors_root,
                model_device=model_device,
                enable_depth=True,
                enable_force=enable_force,
                temporal_stride=temporal_stride,
                force_field_scale=force_field_scale,
                force_field_baseline=force_field_baseline,
                outputs=outputs,
                use_mask=use_mask,
                refine_mask=refine_mask,
                relative=relative,
                relative_scale=relative_scale,
                mask_only_pointcloud=mask_only_pointcloud,
                color_dist_threshold=color_dist_threshold,
                height_threshold=height_threshold
            )
        except RuntimeError as e:
            self.get_logger().error(f"Failed to initialize sensor {serial}: {e}")
            self.get_logger().info(f"Tactile sensor {serial} not detected. Exiting gracefully.")
            raise SystemExit(0)  # Exit gracefully without error

        self.bridge = CvBridge()
        
        # Create publishers based on outputs
        self.publishers = {}
        
        # For backward compatibility, create legacy mode-based publisher
        if mode == 'pointcloud':
            self.publisher = self.create_publisher(PointCloud2, topic, 10)
            self.publishers['pointcloud'] = self.publisher
        elif mode in ['depth', 'gradient']:
            self.publisher = self.create_publisher(Image, topic, 10)
            self.publishers[mode] = self.publisher
        elif mode == 'force_field':
            self.publisher = self.create_publisher(Image, topic, 10)
            self.publishers['force_field'] = self.publisher
        elif mode == 'force_vector':
            self.publisher = self.create_publisher(WrenchStamped, topic, 10)
            self.publishers['force_vector'] = self.publisher
        else:
            self.publisher = self.create_publisher(Image, topic, 10)
            self.publishers['depth'] = self.publisher
        
        # Create additional publishers for multi-output mode
        base_topic = f"tactile/{serial}"
        for output in outputs:
            if output not in self.publishers:
                if output == 'depth':
                    self.publishers['depth'] = self.create_publisher(Image, f"{base_topic}/depth", 10)
                elif output == 'gradient':
                    self.publishers['gradient'] = self.create_publisher(Image, f"{base_topic}/gradient", 10)
                elif output == 'pointcloud':
                    self.publishers['pointcloud'] = self.create_publisher(PointCloud2, f"{base_topic}/pointcloud", 10)
                elif output == 'force_field':
                    self.publishers['force_field'] = self.create_publisher(Image, f"{base_topic}/force_field", 10)
                elif output == 'force_vector':
                    self.publishers['force_vector'] = self.create_publisher(WrenchStamped, f"{base_topic}/force_vector", 10)
            
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

    def timer_callback(self):
        frame, result_dict = self.processor.get_latest_output()
        if not result_dict:
            return
            
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = f"tactile_{self.get_parameter('serial').get_parameter_value().string_value}"

        # Publish all available outputs
        for output_name, result in result_dict.items():
            if result is None:
                continue  # Skip None results (e.g., force during warmup)
            
            if output_name not in self.publishers:
                continue  # Skip outputs without publishers
            
            publisher = self.publishers[output_name]
            
            if output_name == 'pointcloud':
                # Handle pointcloud output
                pc = result
                # Prefer colors provided in result dict (computed from force_field)
                colors = result_dict.get('pointcloud_colors')
                forces = result_dict.get('pointcloud_forces')  # Nx3 array of raw forces (fx,fy,fz)

                if colors is not None:
                    msg = self.create_pointcloud2_msg(pc, header, colors=colors, color_format=self.pointcloud_color_format, forces=forces if self.publish_force_fields else None)
                elif isinstance(result, tuple) and self.return_color:
                    pc, colors = result
                    msg = self.create_pointcloud2_msg(pc, header, colors=colors, color_format=self.pointcloud_color_format)
                else:
                    msg = self.create_pointcloud2_msg(pc, header)

                if msg is not None:
                    publisher.publish(msg)
                
            elif output_name == 'depth':
                # Handle depth output (2D array, uint8)
                if len(result.shape) == 2:
                    msg = self.bridge.cv2_to_imgmsg(result, encoding='mono8')
                    msg.header = header
                    publisher.publish(msg)
                    
            elif output_name == 'gradient':
                # Handle gradient output (H, W, 2) - convert to 2-channel float32 image
                if len(result.shape) == 3 and result.shape[2] == 2:
                    msg = self.bridge.cv2_to_imgmsg(result.astype(np.float32), encoding='32FC2')
                    msg.header = header
                    publisher.publish(msg)
            
            elif output_name == 'force_field':
                # Handle force field output - dict with 'normal' and 'shear'
                if isinstance(result, dict):
                    normal = result.get('normal')  # [224, 224]
                    shear = result.get('shear')    # [224, 224, 2]
                    
                    if normal is not None and shear is not None:
                        # Create 3-channel RGB image: R=Fx, G=Fz, B=Fy
                        # Map forces to RGB image for publishing: R=Fx, G=Fy, B=Fz
                        force_rgb = np.zeros((normal.shape[0], normal.shape[1], 3), dtype=np.float32)
                        force_rgb[:, :, 0] = shear[:, :, 0]  # Fx
                        force_rgb[:, :, 1] = shear[:, :, 1]  # Fy
                        force_rgb[:, :, 2] = normal         # Fz

                        msg = self.bridge.cv2_to_imgmsg(force_rgb, encoding='32FC3')
                        msg.header = header
                        publisher.publish(msg)
            
            elif output_name == 'force_vector':
                # Handle force vector output - dict with 'fx', 'fy', 'fz'
                if isinstance(result, dict):
                    fx = result.get('fx', 0.0)
                    fy = result.get('fy', 0.0)
                    fz = result.get('fz', 0.0)
                    
                    msg = WrenchStamped()
                    msg.header = header
                    msg.wrench.force.x = float(fx)
                    msg.wrench.force.y = float(fy)
                    msg.wrench.force.z = float(fz)
                    msg.wrench.torque.x = 0.0
                    msg.wrench.torque.y = 0.0
                    msg.wrench.torque.z = 0.0
                    publisher.publish(msg)

    def create_pointcloud2_msg(self, points, header, colors=None, color_format='rgb_packed', forces=None):
        """Create a PointCloud2 message from numpy point array.

        Args:
            points: (N,3) float32
            colors: optional (N,3) float in 0..1 RGB
            color_format: 'rgb_packed' or 'r_g_b'
            forces: optional (N,3) float32 array of (fx,fy,fz)
        """
        if points.shape[1] != 3:
            self.get_logger().error(f"Expected points with shape (N, 3), got {points.shape}")
            return None

        n = len(points)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        # Prepare combined array columns
        columns = [points.astype(np.float32)]

        # Colors
        if colors is not None:
            cols = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
            if color_format == 'rgb_packed':
                # Pack into single uint32 and reinterpret as float32
                r = cols[:, 0].astype(np.uint32)
                g = cols[:, 1].astype(np.uint32)
                b = cols[:, 2].astype(np.uint32)
                rgb_uint32 = (r << 16) | (g << 8) | b
                rgb_float32 = rgb_uint32.view(np.float32)
                fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
                columns.append(rgb_float32.reshape(-1, 1))
                point_step = 16
            else:
                # r, g, b as float32 fields
                fields.append(PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1))
                fields.append(PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1))
                fields.append(PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1))
                columns.append(cols.astype(np.float32))
                point_step = 24
        else:
            point_step = 12

        # Forces as additional float fields (fx, fy, fz)
        if forces is not None:
            forces_arr = np.asarray(forces, dtype=np.float32)
            if forces_arr.shape[0] != n or forces_arr.shape[1] != 3:
                self.get_logger().error("Forces must have shape (N, 3)")
                forces = None
            else:
                # Add fx, fy, fz fields
                offset = point_step
                fields.append(PointField(name='fx', offset=offset, datatype=PointField.FLOAT32, count=1))
                fields.append(PointField(name='fy', offset=offset + 4, datatype=PointField.FLOAT32, count=1))
                fields.append(PointField(name='fz', offset=offset + 8, datatype=PointField.FLOAT32, count=1))
                columns.append(forces_arr)
                point_step = offset + 12

        # Construct combined array
        combined = np.hstack(columns)

        # Create PointCloud2 message
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

    def destroy_node(self):
        self.processor.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TactileStreamerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()