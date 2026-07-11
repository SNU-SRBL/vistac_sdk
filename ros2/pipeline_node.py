#!/usr/bin/env python3
"""Pipeline node: reads camera SHM, runs tactile processing, writes outputs to SHM.

ProcessingEngine handles SHM reading, background collection,
and TactileProcessor management via async worker threads.
Worker threads own GPU processing; timer callbacks never block on GPU.

Writes to SHM:
  /dev/shm/tactile_{serial}_surface   — depth + pointcloud
  /dev/shm/tactile_{serial}_force     — force_field + force_vector

Publishes inline (DDS):
  /tactile/{serial}/gradient          — gradient map (lower rate, fine to block)
"""

import struct
import time
from multiprocessing import shared_memory
from typing import Dict

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

_BE_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np

from digit_sdk.processing_engine import ProcessingEngine

SHM_SURFACE_HEADER = 32
SHM_FORCE_HEADER = 40


class PipelineNode(Node):
    """Runs tactile processing pipeline, writes results to SHM for publishers."""

    def __init__(self):
        super().__init__('pipeline_node')

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

        for serial in active_sensors:
            self._engine.collect_background(serial)
        self._engine.start_workers()

        # ---- Initialize SHM segments ----
        self._surface_shms: Dict[str, shared_memory.SharedMemory] = {}
        self._surface_seqs: Dict[str, int] = {}
        self._force_shms: Dict[str, shared_memory.SharedMemory] = {}
        self._force_seqs: Dict[str, int] = {}
        has_force = any(o in outputs for o in ['force_field', 'force_vector'])

        for serial in active_sensors:
            # Surface SHM: depth + PC
            try:
                s = shared_memory.SharedMemory(
                    name=f'tactile_{serial}_surface', create=False)
                s.close()
                s.unlink()
            except FileNotFoundError:
                pass
            # Max size: 32 header + 320*240 depth + 921600 PC ≈ 1MB
            self._surface_shms[serial] = shared_memory.SharedMemory(
                name=f'tactile_{serial}_surface', create=True,
                size=SHM_SURFACE_HEADER + 320 * 240 + 921600)
            self._surface_seqs[serial] = 0

            # Force SHM: only if force enabled
            if has_force:
                try:
                    s = shared_memory.SharedMemory(
                        name=f'tactile_{serial}_force', create=False)
                    s.close()
                    s.unlink()
                except FileNotFoundError:
                    pass
                # 40 header + 3 x 224x224x4 (normal, shear_x, shear_y)
                self._force_shms[serial] = shared_memory.SharedMemory(
                    name=f'tactile_{serial}_force', create=True,
                    size=SHM_FORCE_HEADER + 3 * 224 * 224 * 4)
                self._force_seqs[serial] = 0

        # ---- Publishers (DDS) — only gradient stays inline ----
        self._grad_publishers: Dict[str, object] = {}
        for serial in active_sensors:
            if 'gradient' in outputs:
                self._grad_publishers[serial] = self.create_publisher(
                    Image, f'tactile/{serial}/gradient', _BE_QOS)

        # ---- Per-sensor timers ----
        self._sensor_timers: Dict[str, object] = {}
        for serial in active_sensors:
            cbg = ReentrantCallbackGroup()
            self._sensor_timers[serial] = self.create_timer(
                1.0 / rate,
                lambda s=serial: self._handle_sensor(s),
                callback_group=cbg,
            )

        self.get_logger().info(
            f'Pipeline node ready for {len(active_sensors)} sensors '
            f'({", ".join(active_sensors)}) on {model_device}')

    # ------------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------------

    def _handle_sensor(self, serial: str):
        """Submit frame to worker, get result, write SHM, publish gradient."""
        bgr = self._engine.read_frame(serial)
        if bgr is not None:
            self._engine.submit_frame(serial, bgr)

        result = self._engine.get_latest_result(serial)
        if result:
            # Write to SHM for surface/force publishers
            self._write_surface_shm(serial, result)
            if serial in self._force_shms:
                self._write_force_shm(serial, result)

            # Publish gradient inline (lower rate, fine to block)
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = f'tactile_{serial}'
            self._publish_gradient(serial, result, header)

    # ------------------------------------------------------------------
    # SHM write methods
    # ------------------------------------------------------------------

    def _write_surface_shm(self, serial: str, result: dict) -> None:
        """Write depth + pointcloud to surface SHM."""
        shm = self._surface_shms.get(serial)
        if shm is None:
            return
        depth = result.get('depth')
        if depth is None:
            return
        pc = result.get('pointcloud')
        buf = shm.buf
        h, w = depth.shape
        pc_count = 0
        pc_bytes = b''
        if pc is not None and len(pc) > 0:
            pc_count = len(pc)
            pc_bytes = np.asarray(pc, dtype=np.float32).tobytes()

        # Write header
        buf[28] = 0  # valid=0
        seq = self._surface_seqs.get(serial, 0) + 1
        ts = time.time_ns()
        struct.pack_into('<QQIIIB', buf, 0, seq, ts, h, w, pc_count, 0)

        # Write depth data
        offset = SHM_SURFACE_HEADER  # 32
        buf[offset:offset + h * w] = depth.tobytes()
        offset += h * w

        # Write pointcloud data
        if pc_bytes:
            buf[offset:offset + len(pc_bytes)] = pc_bytes

        buf[28] = 1  # valid=1
        self._surface_seqs[serial] = seq

    def _write_force_shm(self, serial: str, result: dict) -> None:
        """Write force_field + force_vector to force SHM."""
        shm = self._force_shms.get(serial)
        if shm is None:
            return
        ff = result.get('force_field')
        fv = result.get('force_vector')
        if ff is None:
            return
        normal = ff.get('normal')
        shear = ff.get('shear')
        if normal is None or shear is None:
            return

        buf = shm.buf
        h, w = normal.shape
        fx = float(fv.get('fx', 0.0)) if fv else 0.0
        fy = float(fv.get('fy', 0.0)) if fv else 0.0
        fz = float(fv.get('fz', 0.0)) if fv else 0.0

        buf[36] = 0  # valid=0
        seq = self._force_seqs.get(serial, 0) + 1
        ts = time.time_ns()
        struct.pack_into('<QQIIfffB', buf, 0, seq, ts, h, w, fx, fy, fz, 0)

        offset = SHM_FORCE_HEADER  # 40
        np.asarray(normal, dtype=np.float32).tofile(buf, offset)
        offset += h * w * 4
        np.asarray(shear[..., 0], dtype=np.float32).tofile(buf, offset)
        offset += h * w * 4
        np.asarray(shear[..., 1], dtype=np.float32).tofile(buf, offset)

        buf[36] = 1  # valid=1
        self._force_seqs[serial] = seq

    # ------------------------------------------------------------------
    # Gradient publish (inline DDS)
    # ------------------------------------------------------------------

    def _publish_gradient(
            self, serial: str, result: dict, header: Header) -> None:
        """Publish gradient if configured."""
        pub = self._grad_publishers.get(serial)
        if pub is None:
            return
        grad = result.get('gradient')
        if grad is None:
            return
        if len(grad.shape) == 3 and grad.shape[2] == 2:
            msg = Image()
            msg.height, msg.width = grad.shape[0], grad.shape[1]
            msg.encoding = '32FC2'
            msg.is_bigendian = False
            msg.step = grad.shape[1] * 4 * 2
            msg.data = np.ascontiguousarray(
                grad.astype(np.float32)).tobytes()
            msg.header = header
            pub.publish(msg)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        self._engine.shutdown()
        for shm in self._surface_shms.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        for shm in self._force_shms.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PipelineNode()
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
