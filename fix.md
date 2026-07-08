# Frame Corruption: Root Cause, Fixes Tried, and Remaining Options

**Date**: 2026-07-08 | **Session**: #9 | **Status**: document only

## Problem

4 DIGIT tactile sensors (D21275, D21273, D21242, D21119) produce corrupted raw
frames when running under ROS load. A corrupted frame shows a **horizontal
tear** — the top half comes from one camera capture, the bottom half from
another. This causes RViz to show torn images, and without filtering, depth
output becomes 100% garbage (background collection captures corrupt frames).

## Root Cause

### Proven chain

```
ROS DDS threads consume CPU
  → uvcvideo kernel workers (I< priority) descheduled
  → USB isochronous interrupt handling delayed
  → STM32 firmware internal frame buffer wraps
  → horizontal tear in delivered frame
```

### Evidence

| Test | Result |
|------|--------|
| 4 cameras, no ROS, no DDS | **0% corruption** |
| 4 cameras, with ROS (any DDS) | **5-41% corruption** |
| Single camera, with ROS | 0% corruption |
| 2 cameras, with ROS | ~5% corruption |
| USB bandwidth | 50% headroom (not a saturation issue) |
| Raw YUYV via v4l2-ctl (no OpenCV) | 69.9% corrupt under full load |
| ROW-CONTINUITY mx/med | >3.0 reliably detects tear |

### Eliminated hypotheses

| Hypothesis | Test | Result |
|-----------|------|--------|
| OpenCV buffer sharing | `cap.read()` returns owned copies | Disproved |
| V4L2 kernel buffer mixing | ftrace monotonic sequences | Disproved |
| USB packet loss/corruption | usbtop stable bandwidth | Disproved |
| GIL contention within one process | Separate camera/process | Reduced but not eliminated |
| DDS network traffic | `ROS_LOCALHOST_ONLY=1` | Worse |
| DDS overhead (CycloneDDS) | RMW swap | Broken discovery on this system |

## Architecture Evolution

### v1: Original (pre-session #9)

```
LiveTactileProcessor (single process per sensor, ~290 lines)
  ├── Camera (threaded, ffmpeg subprocess)
  ├── TactileProcessor (depth model on CPU/GPU)
  └── ROS publisher (raw + depth + pointcloud)
```

**Problem**: CPU depth model (200ms) hogs GIL, camera thread starved, STM32
buffer wraps. When corruption becomes continuous, `last_good_frame` never
updates → 406 consecutive identical frames → 6.8 second freeze.

### v2: Process separation (session #9)

```
camera_node (99 lines, rclpy)              process_node (522 lines, rclpy)
  Camera.get_image()                          subscribe /raw (DDS)
  → BGR→RGB                                  → corruption filter (mx/med > 3.0)
  → pub.publish(/tactile/{ser}/raw)          → TactileProcessor (depth)
                                              → publish /depth, /pointcloud, /force
```

**Effect**: Separate GIL per camera, separate GIL per processor. Corruption
dropped from 100% to 5-41%. camera_node went through 3 iterations:
- Synchronous timer callback → 100% corrupt
- Capture thread + publish thread → 17-25% corrupt
- **Single-threaded capture+publish** → 5-41% corrupt (current)

Corruption filter in process_node: skips mx/med > 3.0 frames for background
collection and processing. Depth output clean.

### v3: Shared Memory IPC (this session)

```
camera_shm (130 lines, NO rclpy, NO DDS)    process_node (564 lines, rclpy)
  Camera.get_image()                          poll shm (no DDS subscription)
  → BGR→RGB                                  → re-publish /tactile/{ser}/raw
  → write to SharedMemory                    → corruption filter
  → seq++ / valid flag                       → TactileProcessor
                                              → publish /depth, /pointcloud, /force
```

**Shared memory structure** (per sensor, named `tactile_{serial}`):
```
Offset  Size    Field          Type
------  ------  -------------  ----------
     0       8  seq             uint64    monotonic counter
     8       8  timestamp_ns    uint64    time.monotonic_ns()
    16       4  height          uint32    frame height (240)
    20       4  width           uint32    frame width (320)
    24       1  valid           uint8     0=writing, 1=complete
    32  230400  data            uint8[]   RGB frame (320×240×3)
```

Lock-free sync: camera sets valid→0, writes data+metadata, sets seq++, valid→1.
Process checks valid==1 then reads.

**Goal**: Eliminate 180 DDS threads from camera side (45 per camera_node × 4).

**Effect**: Corruption **33% unpaced, 23% paced at 60 Hz**. Slightly worse than
v2 because camera_shm has zero backpressure (shm writes are microseconds vs DDS
publish taking milliseconds). Without publish throttling, camera poll loop
hammers USB bus harder.

## Corruption Rate Summary

| Architecture | Rate | Notes |
|-------------|:----:|-------|
| No ROS, no DDS | **0%** | Proof that USB subsystem works |
| v2 single-threaded camera_node + FastRTPS | **5-41%** | Bimodal, environment-dependent |
| v3 SHM unpaced | **33%** | Too fast, hammers USB bus |
| v3 SHM paced at 60 Hz | **23%** | Better but worse than v2 |
| CycloneDDS | N/A | Discovery broken on this system |
| ROS_LOCALHOST_ONLY=1 | N/A | Worse than default |

## Remaining Bottleneck

The v3 camera_shm has **zero DDS threads** (0 vs v2's 45 per camera), yet
corruption persists at 23%. The remaining bottleneck is the **4 process_node
instances: 148 DDS threads** (37 per node) competing on 12 logical CPUs.

Kernel-level evidence:
```
IRQ 73 (xhci Bus 1):  1.1B interrupts — D21273 + D21242
IRQ 82 (xhci Bus 3):  0.8B interrupts — D21275 + D21119
8 uvcvideo kernel workers at I< priority (starved by userspace threads)
```

## Suggested Methods

### Option A: Merge process_node into one process (code change)

One ROS node handles all 4 sensors. 148 DDS threads → 37.

```
process_node (all 4 sensors, 37 DDS threads)
  ├── poll shm D21275 → raw_pub → filter → processor[0] → publish depth
  ├── poll shm D21273 → raw_pub → filter → processor[1] → publish depth
  ├── poll shm D21242 → raw_pub → filter → processor[2] → publish depth
  └── poll shm D21119 → raw_pub → filter → processor[3] → publish depth
```

**Pros**: 111 fewer threads, simpler launch (1 process instead of 4), shared
background collection logic. CUDA inference already serialized on one GPU.

**Cons**: 4x Python work in one GIL, single point of failure (crash takes down
all 4 sensors), significant code changes.

### Option B: Core isolation (kernel config, no code changes)

Pin USB interrupts + uvcvideo workers to dedicated CPU cores. DDS/ROS threads
cannot touch those cores → interrupts always serviced immediately.

```bash
# Reserve cores 10-11 for USB only
isolcpus=10,11 nohz_full=10,11 rcu_nocbs=10,11

# Pin USB IRQs to reserved cores
echo 400 > /proc/irq/73/smp_affinity   # Bus 1 → core 10
echo 800 > /proc/irq/82/smp_affinity   # Bus 3 → core 11
```

**Pros**: Proven solution on NVIDIA Jetson with identical DIGIT/STM32 issues.
Guarantees zero competition for USB interrupts. No code changes.

**Cons**: Requires `sudo` and reboot. Loses 2 of 12 cores for ROS (8 remaining).
May starve SLAM/allegro controller on reduced core count. Hard to undo remotely.

### Option C: Accept current state (default)

The v2 architecture (single-threaded camera_node + FastRTPS + process_node
corruption filter) handles the residual 5-41% corruption. Depth output is clean
because corrupt frames are rejected during background collection and processing.

**Pros**: Already implemented and tested. 10+ minutes stable at 58-62 Hz depth.

**Cons**: Raw topic shows torn frames ~5-41% of the time. Only cosmetic for
debugging.

## Current Code State

### Files (vistac_sdk submodule)

| File | Lines | Description |
|------|------:|-------------|
| `ros2/camera_node.py` | 136 | SHM-based camera process (plain Python, no rclpy) |
| `ros2/process_node.py` | 564 | SHM reader + raw re-publish + depth processor |
| `ros2/launch/multi_sensor_tactile_streamer.launch.py` | modified | ExecuteProcess for camera_shm, Node for process_node |
| `CMakeLists.txt` | modified | RENAME camera_node → camera_shm |
| `vistac_sdk/vistac_device.py` | 169 | Synchronous Camera (simplified from ~290) |

### Deleted (session #9)

| File | Reason |
|------|--------|
| `vistac_sdk/live_core.py` | LiveTactileProcessor — coupled Camera+Processor |
| `ros2/tactile_streamer_node.py` | Combined node — replaced by camera_node + process_node |
| `vistac_sdk/test_camera.py` | Investigation artifact |
| All investigation diagnostics | MD5 freeze detector, watchdog, reconnect, signal handlers |

### Downstream topics (unchanged)

```
/tactile/{serial}/raw          → sensor_msgs/Image (bgr8)
/tactile/{serial}/depth        → sensor_msgs/Image (32FC1)
/tactile/{serial}/pointcloud   → sensor_msgs/PointCloud2
/tactile/{serial}/force_field  → sensor_msgs/Image (32FC3)
/tactile/{serial}/force_vector → geometry_msgs/WrenchStamped
```

## Related Commits

| Hash | Description |
|------|-------------|
| `d6953c2` (vistac_sdk) | refactor: separate camera and depth/force into independent processes |
| `403360b` (vistac_sdk) | fix: V4L2 buffer exhaustion causes frame freeze at ~4000 reads |
| `a819664` (vistac_sdk) | refactor: replace ffmpeg subprocess with cv2.VideoCapture |
| `65504dc` (gaussianfeels) | session #9: root cause + architecture refactor |
