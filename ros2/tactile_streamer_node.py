import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np

from vistac_sdk.live_core import LiveReconstructor

'''
This ROS2 node streams tactile data from a sensor using the LiveReconstructor class.
It publishes the data as an Image message on a specified topic.
Parameters:
- serial: Sensor serial number (default: 'YOUR_SENSOR_SERIAL')
- sensors_root: Root directory for sensor configurations (default: '../sensors')
- device_type: Device type for model execution (default: 'cuda')
- mode: Mode of operation (default: 'depth')
- use_mask: Whether to apply a mask to the data (default: False)
- topic: Topic to publish the data (default: 'tactile/depth')
- rate: Publishing rate in Hz (default: 15.0)
'''

class TactileStreamerNode(Node):
    def __init__(self):
        super().__init__('tactile_streamer_node')
        # Declare and get parameters
        self.declare_parameter('serial', 'YOUR_SENSOR_SERIAL')
        self.declare_parameter('sensors_root', '../sensors')
        self.declare_parameter('device_type', 'cuda')
        self.declare_parameter('mode', 'depth')
        self.declare_parameter('use_mask', False)
        self.declare_parameter('topic', 'tactile/depth')
        self.declare_parameter('rate', 15.0)

        serial = self.get_parameter('serial').get_parameter_value().string_value
        sensors_root = self.get_parameter('sensors_root').get_parameter_value().string_value
        device_type = self.get_parameter('device_type').get_parameter_value().string_value
        mode = self.get_parameter('mode').get_parameter_value().string_value
        use_mask = self.get_parameter('use_mask').get_parameter_value().bool_value
        topic = self.get_parameter('topic').get_parameter_value().string_value
        rate = self.get_parameter('rate').get_parameter_value().double_value

        self.recon = LiveReconstructor(
            serial=serial,
            sensors_root=sensors_root,
            model_device=device_type,
            mode=mode,
            use_mask=use_mask
        )
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, topic, 10)
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

    def timer_callback(self):
        frame, result = self.recon.get_latest_output()
        if result is None:
            return
        # For depth mode, result is a 2D numpy array
        if len(result.shape) == 2:
            msg = self.bridge.cv2_to_imgmsg(result.astype(np.float32), encoding='32FC1')
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(msg)
        # For pointcloud or gradient, adapt as needed

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