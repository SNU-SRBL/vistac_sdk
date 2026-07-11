#!/usr/bin/env python3
"""Surface publisher: reads SHM tactile_{serial}_surface, publishes depth+PC DDS.

One process per sensor — independent GIL.  Reads depth (mono8) and pointcloud
from shared memory written by pipeline_node, publishes as ROS topics.

Topics:
  /tactile/{serial}/depth       — sensor_msgs/Image (mono8)
  /tactile/{serial}/pointcloud  — sensor_msgs/PointCloud2
"""

import struct
import time
from multiprocessing import shared_memory

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
import numpy as np

_BE_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

SHM_HEADER = 32


class SurfacePublisher(Node):
    """Reads surface SHM, publishes depth + pointcloud over DDS."""

    def __init__(self):
        super().__init__('surface_publisher')

        self.declare_parameter('serial', value='')
        self.declare_parameter('rate', 60.0)

        serial = self.get_parameter('serial').value
        rate = self.get_parameter('rate').value

        if not serial:
            self.get_logger().error('No serial parameter — nothing to do')
            return

        # Connect to SHM (retry up to 10s)
        self._shm = None
        for _ in range(100):
            try:
                self._shm = shared_memory.SharedMemory(
                    name=f'tactile_{serial}_surface', create=False)
                break
            except FileNotFoundError:
                time.sleep(0.1)
        if self._shm is None:
            self.get_logger().error(
                f'Surface SHM not found for {serial} — exiting')
            return

        self._last_seq = -1
        self._serial = serial

        # Publishers
        self._pub_depth = self.create_publisher(
            Image, f'tactile/{serial}/depth', _BE_QOS)
        self._pub_pc = self.create_publisher(
            PointCloud2, f'tactile/{serial}/pointcloud', _BE_QOS)

        self._timer = self.create_timer(1.0 / rate, self._handle_sensor)

        self.get_logger().info(
            f'Surface publisher ready for {serial} @ {rate:.0f}Hz')

    def _handle_sensor(self):
        """Read latest depth + PC from SHM, publish."""
        if self._shm is None:
            return
        buf = self._shm.buf

        if not buf[28]:
            return

        seq = struct.unpack_from('<Q', buf, 0)[0]
        if seq == self._last_seq:
            return
        self._last_seq = seq

        h, w, pc_count = struct.unpack_from('<III', buf, 16)
        if h == 0 or w == 0:
            return

        # Depth
        depth_data = np.frombuffer(
            buf[32:32 + h * w], dtype=np.uint8).reshape(h, w)

        depth_msg = Image()
        depth_msg.height = h
        depth_msg.width = w
        depth_msg.encoding = 'mono8'
        depth_msg.is_bigendian = False
        depth_msg.step = w
        depth_msg.data = depth_data.tobytes()
        depth_msg.header.frame_id = f'tactile_{self._serial}'
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        self._pub_depth.publish(depth_msg)

        # Pointcloud (if any)
        if pc_count > 0:
            pc_offset = 32 + h * w
            pc_bytes = buf[pc_offset:pc_offset + pc_count * 12]
            pts = np.frombuffer(pc_bytes, dtype=np.float32).reshape(-1, 3)

            pc_msg = PointCloud2()
            pc_msg.header = depth_msg.header
            pc_msg.height = 1
            pc_msg.width = pc_count
            pc_msg.fields = [
                PointField(name='x', offset=0,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,
                           datatype=PointField.FLOAT32, count=1),
            ]
            pc_msg.is_bigendian = False
            pc_msg.point_step = 12
            pc_msg.row_step = 12 * pc_count
            pc_msg.is_dense = True
            pc_msg.data = pts.tobytes()
            self._pub_pc.publish(pc_msg)

    def destroy_node(self):
        if self._shm is not None:
            self._shm.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SurfacePublisher()
    executor = SingleThreadedExecutor()
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
