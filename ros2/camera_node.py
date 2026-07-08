#!/usr/bin/env python3
"""Camera-only node: reads DIGIT frames at 60Hz, publishes /tactile/{serial}/raw."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
from vistac_sdk.vistac_device import Camera


class TactileCameraNode(Node):
    """Reads DIGIT camera frames and publishes raw images via ROS.

    Runs in its own process (separate GIL from depth processing).
    Topics:
      /tactile/{serial}/raw  -- sensor_msgs/Image (rgb8)
    """

    def __init__(self):
        super().__init__('tactile_camera_node')

        self.declare_parameter('serial', 'YOUR_SENSOR_SERIAL')
        self.declare_parameter('sensors_root', '../sensors')
        self.declare_parameter('rate', 60.0)
        self.declare_parameter('verbose', True)

        serial = self.get_parameter('serial').value
        sensors_root = self.get_parameter('sensors_root').value
        rate = self.get_parameter('rate').value
        verbose = self.get_parameter('verbose').value

        self.bridge = CvBridge()

        self.camera = Camera(serial=serial, sensors_root=sensors_root)
        self.camera.connect(verbose=verbose)

        self.pub = self.create_publisher(
            Image, f'tactile/{serial}/raw', 10)
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

        self.get_logger().info(
            f'Camera node ready for {serial} ({int(rate)}Hz)')

    def timer_callback(self):
        frame = self.camera.get_image()
        if frame is None:
            return
        # OpenCV BGR -> ROS rgb8
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        msg.header.frame_id = (
            f'tactile_{self.get_parameter("serial").value}')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

    def destroy_node(self):
        self.camera.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TactileCameraNode()
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
