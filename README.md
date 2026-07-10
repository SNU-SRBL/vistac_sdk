# DIGIT SDK

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Modified version of [gs_sdk](https://github.com/joehjhuang/gs_sdk) (depth estimation via per-sensor calibration) integrating [Sparsh](https://github.com/facebookresearch/sparsh) (cross-sensor force estimation), with automatic DIGIT identification, parallel processing, STM32 corruption recovery, and ROS2 integration.

**Authors**: [Byung-Hyun Song](https://github.com/bhsong1011) (bh.song@snu.ac.kr)

## Support

- Ubuntu 22.04, ROS2 Humble
- DIGIT tactile sensors (YUYV, up to 640×480 @ 60Hz)
- Python >= 3.9, CUDA (optional, for force estimation)

## Installation

```bash
git clone git@github.com:SNU-SRBL/digit_sdk.git
cd digit_sdk
pip install -e .
```

### Force Estimation (Optional)

```bash
python scripts/download_models.py    # downloads Sparsh models (~1.7 GB)
pip install -e .[gpu]                 # optional xformers for GPU acceleration
```

Models saved to `models/`. Depth pipeline works on CPU. Force pipeline recommended on GPU (~50-80ms vs 500-1000ms CPU).

## Sensor Registration

Place `{serial}.yaml` in `sensors/{serial}/{serial}.yaml` for each DIGIT sensor.

## ROS2 Launch

```bash
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py
```

With options:

```bash
ros2 launch digit_sdk multi_sensor_tactile_streamer.launch.py \
  mode:=depth outputs:=depth model_device:=cuda rate:=60.0 enable_force:=true
```

**Published topics**:
| Topic | Type | Description |
|-------|------|-------------|
| `/tactile/{serial}/raw` | `sensor_msgs/Image` (bgr8) | Raw camera frame (from camera config), Best Effort |
| `/tactile/{serial}/depth` | `sensor_msgs/Image` (mono8) | Depth in mm |
| `/tactile/{serial}/pointcloud` | `sensor_msgs/PointCloud2` | XYZ point cloud |
| `/tactile/{serial}/force_field` | `sensor_msgs/Image` (32FC3) | Force field (R=Fx, G=Fy, B=Fz) |
| `/tactile/{serial}/force_field_viz` | `sensor_msgs/Image` (rgb8) | RViz-friendly force visualization |
| `/tactile/{serial}/force_vector` | `geometry_msgs/WrenchStamped` | Aggregated force vector |

**Launch parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rate` | 60.0 | Camera capture rate (Hz) |
| `mode` | depth | Processing mode: depth, gradient, pointcloud, force_field, force_vector |
| `outputs` | — | Comma-separated output list (overrides mode) |
| `model_device` | cuda | Device: cuda or cpu |
| `enable_force` | false | Enable Sparsh force estimation |
| `use_mask` | true | Apply contact mask |
| `sensors_root` | auto | Path to sensor configs |

**RViz notes**: Set Fixed Frame to `tactile_{serial}`. Use `force_field_viz` for image display (32FC3 not supported by RViz Image).

## Architecture

### Process Layout (4-camera deployment)

```
Launch auto-cleanup: pkill stale + rm /dev/shm before start.

camera_shm ×4  ──→  /dev/shm/tactile_*  (lock-free, dynamic size)
  Core 0-3           plain Python, no rclpy, own GIL
  │                   Camera.get_image():
  │                     ├ _is_corrupt() → chan_absdiff step-detector
  │                     ├ FPS watchdog → _recover() if gap > 1.5× expected
  │                     └ _recover() → STREAMOFF/STREAMON (~270ms)

raw_bridge ×4   ←──  /dev/shm/tactile_*  ──→  /tactile/{serial}/raw (DDS)
  Core 4-7           rclpy SingleThreadedExecutor(1), own GIL
                      Reads SHM → tobytes() → publish Image (bgr8)

process_node ×1  ←──  /dev/shm/tactile_*  ──→  depth/pc/force (DDS)
  Core free           rclpy MultiThreadedExecutor(4), CUDA
                      ProcessingEngine: read SHM → feed model → publish
```

**9 OS processes total** — each camera_shm and raw_bridge pinned to a dedicated core via `os.sched_setaffinity`. process_node unbound (CUDA-heavy).

### Shared Memory Layout (header + dynamic payload)

| Offset | Size | Field | Type |
|--------|------|-------|------|
| 0 | 8 | seq | uint64 — monotonic frame counter |
| 8 | 8 | timestamp_ns | uint64 — capture time |
| 16 | 4 | height | uint32 (from camera config) |
| 20 | 4 | width | uint32 (from camera config) |
| 24 | 1 | valid | uint8 — 0=writing, 1=complete |
| 25 | 7 | (padding) | — alignment to 32 |
| 32 | H×W×3 | data | uint8[] — BGR frame (height × width × 3) |

Lock-free single-writer/multi-reader: camera sets valid=0, writes data, increments seq, sets valid=1. Readers check seq for new frames.

### STM32 Corruption Detection & Recovery

DIGIT sensors at QVGA (320×240) 60 Hz suffer from STM32 DMA buffer aliasing —
top half of frame N and bottom half of frame N+1 get mixed, producing a horizontal
tear. This issue is specific to QVGA at 60 Hz; lower resolutions or framerates are
not affected. No USB/V4L2 health signal exists for this.

**Detector** (`Camera._is_corrupt`, in `digit_device.py`):
1. `cv2.absdiff` per BGR channel → max → (H-1)×W row-diff
2. Spike mask: each pixel 3× larger than neighbors above AND below, AND >20 absolute
3. Modal row: row with most spike-columns. Must have >50% of W columns agreeing.
4. Isolation ratio: peak-diff at tear row must be 10× larger than ±5-row neighborhood mean
5. Neighbor flatness: >80% of spike-columns have neighbor diffs <15 (tear neighbors are identical copies)

**Recovery**: On corrupt detection → `_recover()` (STREAMOFF/STREAMON cycle, ~270 ms, 20 warmup frames). Resets STM32 DMA state. Corruption rate: <1%.

**FPS Watchdog**: Per-frame gap tracking in `Camera.get_image()`. If 10 consecutive
frame gaps exceed 1.5× the expected interval (derived from configured framerate),
triggers `_recover()` even without visible tears. Catches silent camera slowdown.

### Key Design Decisions

- SHM stores BGR directly — zero-copy view for raw_bridge, one copy for process_node
- `is_corrupt` single source of truth in `digit_device.py` (not duplicated in processing_engine)
- `connect()` uses `/dev/video*` path string, survives STREAMOFF/STREAMON V4L2 index shifts
- Flatness gate proven contact-safe: tear flatness=0.96-1.00, contact=0.85, clean=0.75-0.88
- Best Effort QoS for raw/depth (fire-and-forget, no ACK overhead)

## Performance (4 cameras, 8-core machine)

| Metric | Value |
|--------|-------|
| Camera capture (SHM) | 47–59 fps per sensor |
| Raw DDS | 54–60 Hz per sensor |
| Depth DDS | ~30 Hz (CUDA model inference bound) |
| Corruption rate | <1% |
| Camera CPU | ~6.5% per process |
| Raw bridge CPU | 82–90% per node (DDS discovery overhead, cosmetic) |

## Examples

### Live Viewer

```bash
python apps/live_viewer.py --serial D21273 --mode depth
python apps/live_viewer.py --serial D21273 --mode force_field --enable_force
python apps/live_viewer.py --serial D21273 --outputs depth,force_field,force_vector
```

### Python API

```python
from digit_sdk import Camera

camera = Camera(serial="D21273", sensors_root="sensors")
camera.connect()
# Auto-detects corruption + triggers recovery + watchdog
# Consumer just calls get_image():
while True:
    frame = camera.get_image()
    if frame is not None:
        process(frame)
```

See `apps/live_viewer.py` for full ProcessingEngine integration.

## Sensor Calibration

See [Calibration README](calibration/README.md).

## License

Force estimation uses [Sparsh](https://github.com/facebookresearch/sparsh) models (CC-BY-NC 4.0, non-commercial). Other components retain the original gs_sdk license.

## References

1. Huang et al., "NormalFlow," IEEE RA-L, 2024.
2. Akhter et al., "Sparsh," CoRL, 2024. [GitHub](https://github.com/facebookresearch/sparsh)
3. Lambeta et al., "DIGIT," IEEE RA-L, 2020. [Website](https://digit.ml/)
