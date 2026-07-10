#!/usr/bin/env python3
"""Raw bridge: reads SHM, publishes /tactile/{serial}/raw over DDS.

One process per sensor — independent Python GIL for each camera.
Launched by multi_sensor_tactile_streamer.launch.py with serial:=... param.
"""
import time
from multiprocessing import shared_memory

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
import numpy as np

_BE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

SHM_HEADER = 32


class RawBridgeNode(Node):
    """Reads a single sensor's SHM block, publishes BGR raw frames over DDS.

    One instance per camera — no GIL contention with other sensors.
    """

    def __init__(self):
        super().__init__('raw_bridge_node')

        self.declare_parameter('serial', value='')
        self.declare_parameter('rate', 60.0)
        self.declare_parameter('cpu_core', -1)

        serial = self.get_parameter('serial').value
        rate = self.get_parameter('rate').value
        cpu_core = self.get_parameter('cpu_core').value

        if cpu_core >= 0:
            import os
            os.sched_setaffinity(0, {cpu_core})

        if not serial:
            self.get_logger().error('No serial parameter — nothing to do')
            return

        # Connect to SHM (retry up to 10s)
        self._shm = None
        for _ in range(100):
            try:
                self._shm = shared_memory.SharedMemory(
                    name=f'tactile_{serial}', create=False)
                break
            except FileNotFoundError:
                time.sleep(0.1)
        if self._shm is None:
            self.get_logger().error(f'SHM not found for {serial} — exiting')
            return

        self._last_seq = -1
        self._serial = serial

        # Publisher
        self._pub = self.create_publisher(
            Image, f'tactile/{serial}/raw', _BE_QOS)

        # Single timer — one sensor, no GIL contention
        self._timer = self.create_timer(1.0 / rate, self._handle_sensor)

        self.get_logger().info(f'Raw bridge ready for {serial} @ {rate:.0f}Hz')

    def _handle_sensor(self):
        """Read latest frame from SHM, publish as BGR Image."""
        if self._shm is None:
            return
        buf = self._shm.buf

        if not buf[24]:
            return

        seq = int.from_bytes(buf[0:8], 'little')
        if seq == self._last_seq:
            return
        self._last_seq = seq

        h = int.from_bytes(buf[16:20], 'little')
        w = int.from_bytes(buf[20:24], 'little')
        if h == 0 or w == 0:
            return

        bgr = np.frombuffer(
            buf[SHM_HEADER:SHM_HEADER + h * w * 3],
            dtype=np.uint8,
        ).reshape(h, w, 3)

        msg = Image()
        msg.height = h
        msg.width = w
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = w * 3
        msg.data = bgr.tobytes()
        msg.header.frame_id = f'tactile_{self._serial}'
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)

    def destroy_node(self):
        if self._shm is not None:
            self._shm.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RawBridgeNode()
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
