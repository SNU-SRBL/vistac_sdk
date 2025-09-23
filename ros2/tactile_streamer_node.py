import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import struct

from vistac_sdk.live_core import LiveReconstructor

'''
This ROS2 node streams tactile data from a sensor using the LiveReconstructor class.
It publishes the data as Image or PointCloud2 messages on specified topics.

Complete Parameters List:
- serial: Sensor serial number (default: 'YOUR_SENSOR_SERIAL')
- sensors_root: Root directory for sensor configurations (default: '../sensors') 
- mode: Processing mode - 'depth', 'gradient', 'pointcloud' (default: 'depth')
- contact_mode: Contact detection mode - 'standard' or 'flat' (default: 'standard')
- model_device: Device for model execution - 'cuda' or 'cpu' (default: 'cuda')
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
        self.declare_parameter('use_mask', True)
        self.declare_parameter('refine_mask', True)
        self.declare_parameter('relative', True)
        self.declare_parameter('relative_scale', 0.5)
        self.declare_parameter('mask_only_pointcloud', False)
        self.declare_parameter('return_color', False)
        self.declare_parameter('color_dist_threshold', 15)
        self.declare_parameter('height_threshold', 0.2)
        self.declare_parameter('topic', 'tactile/depth')
        self.declare_parameter('rate', 15.0)
        self.declare_parameter('verbose', True)

        # Get parameters
        serial = self.get_parameter('serial').get_parameter_value().string_value
        sensors_root = self.get_parameter('sensors_root').get_parameter_value().string_value
        mode = self.get_parameter('mode').get_parameter_value().string_value
        contact_mode = self.get_parameter('contact_mode').get_parameter_value().string_value
        model_device = self.get_parameter('model_device').get_parameter_value().string_value
        use_mask = self.get_parameter('use_mask').get_parameter_value().bool_value
        refine_mask = self.get_parameter('refine_mask').get_parameter_value().bool_value
        relative = self.get_parameter('relative').get_parameter_value().bool_value
        relative_scale = self.get_parameter('relative_scale').get_parameter_value().double_value
        mask_only_pointcloud = self.get_parameter('mask_only_pointcloud').get_parameter_value().bool_value
        return_color = self.get_parameter('return_color').get_parameter_value().bool_value
        color_dist_threshold = self.get_parameter('color_dist_threshold').get_parameter_value().integer_value
        height_threshold = self.get_parameter('height_threshold').get_parameter_value().double_value
        topic = self.get_parameter('topic').get_parameter_value().string_value
        rate = self.get_parameter('rate').get_parameter_value().double_value
        verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        # Store mode and return_color for message handling
        self.mode = mode
        self.return_color = return_color

        # Initialize LiveReconstructor with all parameters
        self.recon = LiveReconstructor(
            serial=serial,
            sensors_root=sensors_root,
            model_device=model_device,
            mode=mode,
            use_mask=use_mask,
            refine_mask=refine_mask,
            relative=relative,
            relative_scale=relative_scale,
            mask_only_pointcloud=mask_only_pointcloud,
            color_dist_threshold=color_dist_threshold,
            height_threshold=height_threshold
        )

        self.bridge = CvBridge()
        
        # Create appropriate publisher based on mode
        if mode == 'pointcloud':
            self.publisher = self.create_publisher(PointCloud2, topic, 10)
        else:  # depth or gradient
            self.publisher = self.create_publisher(Image, topic, 10)
            
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

    def timer_callback(self):
        frame, result = self.recon.get_latest_output()
        if result is None:
            return
            
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = f"tactile_{self.get_parameter('serial').get_parameter_value().string_value}"

        if self.mode == 'pointcloud':
            # Handle pointcloud output
            if isinstance(result, tuple) and self.return_color:
                pc, colors = result
                msg = self.create_pointcloud2_msg(pc, header, colors)
            else:
                pc = result
                msg = self.create_pointcloud2_msg(pc, header)
            self.publisher.publish(msg)
            
        elif self.mode == 'depth':
            # Handle depth output (2D array, uint8)
            if len(result.shape) == 2:
                msg = self.bridge.cv2_to_imgmsg(result, encoding='mono8')
                msg.header = header
                self.publisher.publish(msg)
                
        elif self.mode == 'gradient':
            # Handle gradient output (H, W, 2) - convert to 2-channel float32 image
            if len(result.shape) == 3 and result.shape[2] == 2:
                msg = self.bridge.cv2_to_imgmsg(result.astype(np.float32), encoding='32FC2')
                msg.header = header
                self.publisher.publish(msg)

    def create_pointcloud2_msg(self, points, header, colors=None):
        """Create a PointCloud2 message from numpy point array."""
        if points.shape[1] != 3:
            self.get_logger().error(f"Expected points with shape (N, 3), got {points.shape}")
            return None
            
        # Define fields
        if colors is not None:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
            ]
            point_step = 16
            # Combine points and colors
            combined = np.hstack([points.astype(np.float32), colors.astype(np.float32)])
        else:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            point_step = 12
            combined = points.astype(np.float32)
        
        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * len(points)
        msg.data = combined.tobytes()
        msg.is_dense = True
        
        return msg

    def destroy_node(self):
        self.recon.release()
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