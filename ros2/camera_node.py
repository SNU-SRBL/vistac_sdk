#!/usr/bin/env python3
"""Camera process: reads DIGIT frames, writes them to shared memory.

Plain Python (no rclpy, no DDS). Writes BGR→RGB frames plus metadata
to a SharedMemory block named 'tactile_{serial}'.

Layout (230432 bytes):
  Offset   Size  Field        Type     Description
  ------   ----  -----        ----     -----------
      0      8   seq          uint64   Monotonic frame counter
      8      8   timestamp_ns uint64   time.monotonic_ns() at capture
     16      4   height       uint32   Frame height (240)
     20      4   width        uint32   Frame width (320)
     24      1   valid        uint8    0=writing, 1=complete
     25      7   (padding)    --       Alignment to 32
     32  230400  data         uint8[]  RGB frame (320×240×3)

Synchronization (lock-free, single writer / single reader):
  1. Set valid=0
  2. Write data
  3. Increment seq
  4. Set valid=1
"""

import argparse
import signal
import struct
import sys
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

from vistac_sdk.vistac_device import Camera

SHM_SIZE = 230432
DATA_OFFSET = 32


def run(serial: str, sensors_root: str, verbose: bool = False):
    """Main capture loop. Writes frames to shared memory."""
    shm_name = f"tactile_{serial}"

    # Clean up stale shm if any
    try:
        stale = shared_memory.SharedMemory(name=shm_name, create=False)
        stale.close()
        stale.unlink()
        if verbose:
            print(f"Unlinked stale shared memory '{shm_name}'")
    except FileNotFoundError:
        pass

    shm = shared_memory.SharedMemory(
        name=shm_name, create=True, size=SHM_SIZE)

    camera = Camera(serial=serial, sensors_root=sensors_root)
    camera.connect(verbose=verbose)

    running = True

    def _cleanup(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    seq = 0
    buf = shm.buf

    if verbose:
        print(f"Camera {serial}: device={camera.device} "
              f"id={camera.dev_id}", flush=True)

    TARGET_DT = 1.0 / 60  # pace at 60 Hz max

    while running:
        t0 = time.monotonic()
        frame = camera.get_image()
        if frame is None:
            time.sleep(0.001)
            continue

        # BGR→RGB (process_node expects BGR, so store RGB here)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        ts_ns = time.monotonic_ns()

        # Write: valid=0, metadata, data, seq++, valid=1
        buf[24] = 0
        struct.pack_into('<QQII', buf, 0, seq, ts_ns, h, w)
        buf[DATA_OFFSET:DATA_OFFSET + h * w * 3] = np.ascontiguousarray(rgb).tobytes()
        seq += 1
        buf[24] = 1

        # Pace at 60 Hz to avoid USB bus starvation
        elapsed = time.monotonic() - t0
        if elapsed < TARGET_DT:
            time.sleep(TARGET_DT - elapsed)

    # Clean up
    camera.release()
    shm.close()
    shm.unlink()
    if verbose:
        print(f"Camera {serial}: shut down", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="DIGIT camera → shared memory writer")
    parser.add_argument('--serial', required=True, help='Sensor serial')
    parser.add_argument('--sensors-root', required=True,
                        help='Path to sensors config directory')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    try:
        run(serial=args.serial, sensors_root=args.sensors_root,
            verbose=args.verbose)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        # Clean up shm on fatal error
        try:
            tmp = shared_memory.SharedMemory(
                name=f"tactile_{args.serial}", create=False)
            tmp.close()
            tmp.unlink()
        except Exception:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()
