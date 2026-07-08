# Visual-tactile SDK
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) &nbsp;

This repository is a modified version of [gs_sdk](https://github.com/joehjhuang/gs_sdk) with automatic DIGIT identifying, threaded image collection and calculation, and ROS2 implementation.

Authors:
* [Byung-Hyun Song](https://github.com/bhsong1011) (bh.song@snu.ac.kr)

## Support System
* Tested on Ubuntu 22.04
* Tested on Digit
* Python >= 3.9

## Installation
Clone and install vistac_sdk from source:
```bash
git clone git@github.com:SNU-SRBL/vistac_sdk.git
cd vistac_sdk
pip install -e .
```

### Force Estimation Setup (Optional)
To enable force estimation capabilities using [Sparsh](https://github.com/facebookresearch/sparsh), download the pretrained models:
```bash
python scripts/download_models.py
```

Optional GPU acceleration dependency (recommended on CUDA systems):
```bash
pip install -e .[gpu]
```

If `xformers` wheel resolution fails on your platform, continue without it. The force stack still runs, but with reduced performance.

This will download:
- **Encoder**: `sparsh-dino-base` (ViT-base, ~1.7 GB)
- **Decoder**: `sparsh-digit-forcefield` (~15 MB)

Models are saved to `models/` directory.

**Requirements**:
- GPU recommended (CUDA-capable) for real-time performance (~50-80ms)
- Depth pipeline supports CPU execution.
- Force pipeline is intended for GPU execution in this project scope.

**Note**: Force estimation is disabled by default and requires explicit activation.

## Sensor Identification
For multiple DIGIT sensor usage, sensor identification method was implemented from [digit-interface](https://github.com/facebookresearch/digit-interface).
### Sensor Registeration
For a sensor with {serial} number, you need a {serial}.yaml inside sensors/{serial}/{serial}.yaml

## Sensor Calibration
For more details on sensor calibration, see the [Calibration README](calibration/README.md).

## Examples
These examples show basic usage.

### Sensor Streaming
Stream images from a connected DIGIT sensor:
```python
python apps/live_viewer.py --serial D21273 --mode depth
```

### Depth Reconstruction
Stream reconstructed depth from a connected DIGIT sensor:
```python
python apps/live_viewer.py --serial D21273 --mode depth
```

With advanced options:
```python
python apps/live_viewer.py --serial D21273 --use_mask --mode depth --relative --relative_scale 0.5 --height_threshold 0.5
```

### Force Estimation
**Prerequisites**: Download models first (`python scripts/download_models.py`)

#### Force Field (Dense Heatmap)
Visualize force distribution as RGB heatmap (R=Fx, G=Fy, B=Fz):
```python
python apps/live_viewer.py --serial D21273 --mode force_field --enable_force
```

#### PointCloud colored by Force
Visualize point cloud colored by the force field (per-point RGB: R=Fx, G=Fy, B=Fz):
```python
python apps/live_viewer.py --serial D21273 --mode pointcloud_force --enable_force
```

#### Force Vector (Arrow Overlay)
Show aggregated force vector with magnitude:
```python
python apps/live_viewer.py --serial D21273 --mode force_vector --enable_force
```

#### Combined Outputs (Multi-Panel)
Display multiple outputs simultaneously:
```python
python apps/live_viewer.py --serial D21273 --enable_force --outputs depth force_field force_vector
```

### Python API Usage

#### Standalone (without ROS)

```python
from vistac_sdk import Camera, TactileProcessor
from vistac_sdk.utils import load_config
import numpy as np, time

# Open camera
camera = Camera(serial="D21273", sensors_root="sensors")
camera.connect()

# Collect background (10 frames, 200ms apart)
bg_images = []
for _ in range(10):
    time.sleep(0.2)
    frame = camera.get_image()
    while frame is None:
        time.sleep(0.01)
        frame = camera.get_image()
    bg_images.append(frame)
bg = np.mean(bg_images, axis=0).astype(np.uint8)

# Load processor
config = load_config(serial="D21273", sensors_root="sensors")
processor = TactileProcessor(
    model_path=f"sensors/D21273/model/nnmodel.pth",
    enable_depth=True,
    enable_force=False,
    ppmm=config["ppmm"])
processor.load_background(bg)
processor.start_thread(outputs=['depth', 'pointcloud'])

# Process frames
while True:
    frame = camera.get_image()
    if frame is not None:
        processor.set_input_frame(frame, time.time())
    result = processor.get_latest_result()
    # result = {'depth': ndarray, 'pointcloud': ndarray, ...}
```

#### Select outputs per frame

```python
processor.start_thread(outputs=['depth', 'force_field', 'force_vector'])
# processor.process() computes only requested outputs per frame
```

### ROS2 Integration

#### Depth Streaming (Default)
```bash
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py
```

#### Force Estimation
```bash
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py enable_force:=true
```

#### Combined Depth + Force
```bash
ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py \
    enable_force:=true \
    outputs:=depth,force_field,force_vector
```

**Published Topics**:
- `/tactile/{serial}/depth` - `sensor_msgs/Image` (mono8)
- `/tactile/{serial}/gradient` - `sensor_msgs/Image` (32FC2)
- `/tactile/{serial}/pointcloud` - `sensor_msgs/PointCloud2`
- `/tactile/{serial}/force_field` - `sensor_msgs/Image` (32FC3, RGB=Fx,Fy,Fz)  # channels: R=fx, G=fy, B=fz
- `/tactile/{serial}/force_field_viz` - `sensor_msgs/Image` (rgb8, RViz-friendly force visualization)
- `/tactile/{serial}/force_vector` - `geometry_msgs/WrenchStamped`

**RViz notes**:
- Set Fixed Frame to `tactile_{serial}` (e.g., `tactile_D21242`).
- Use `/tactile/{serial}/force_field_viz` for image display in RViz. (`32FC3` is not directly supported by RViz Image display.)

## Architecture

### Modular Design
- **Camera**: Synchronous DIGIT camera capture (own GIL in ROS mode)
- **TactileProcessor**: Unified depth/force inference with selective outputs
- **DepthEstimator**: Fast MLP-based depth reconstruction (~1-2ms)
- **ForceEstimator**: Sparsh ViT-based force estimation (~50-80ms on GPU)
- **TemporalBuffer**: Circular buffer for temporal frame pairs

### ROS2 Architecture
Two separate processes per sensor:
1. **camera_node**: Reads DIGIT at 60Hz, publishes `/tactile/{serial}/raw` (rgb8)
2. **process_node**: Subscribes `/tactile/{serial}/raw`, runs TactileProcessor, publishes depth/pointcloud/force topics

Each node runs in its own process (own GIL), so camera capture never competes with depth inference for CPU time.

### Force Estimation Pipeline
1. **Temporal Buffering**: Maintains circular buffer of recent frames
2. **Background Subtraction**: Normalizes contact appearance (configurable offset)
3. **Encoder**: ViT-base (768-dim) extracts visual features from temporal pairs
4. **Decoder**: DPT-style multi-scale fusion predicts normal/shear force fields
5. **Aggregation**: Spatial averaging produces force vector (Fx, Fy, Fz)

### Selective Execution
Only requested outputs are computed per frame, enabling efficient performance:
- Depth-only: ~1-2ms
- Force-only: ~50-80ms (GPU)
- Combined: ~50-80ms (GPU, force dominates)

### Output Formats
**Depth outputs**:
- `depth`: `[H, W]` uint8, depth in mm
- `gradient`: `[H, W, 2]` float32, surface gradients
- `pointcloud`: `[N, 3]` float32, XYZ coordinates in meters
- `pointcloud_colors` (optional): `[N, 3]` float32 RGB colors derived from the force field (R=fx, G=fy, B=fz)
- `pointcloud_forces` (optional): `[N, 3]` float32 per-point force values (fx, fy, fz)
- `mask`: `[H, W]` bool, contact mask

**Force outputs** (None during warmup):
- `force_field`: dict with `normal` [224, 224] and `shear` [224, 224, 2]
- `force_vector`: dict with `fx`, `fy`, `fz` scalars (normalized [-1, 1])

### Force Visualization Policy
- `ForceEstimator` outputs are model-native: `normal` is sigmoid-bounded to `[0,1]`; `shear` is model-scaled (Sparsh head semantics).
- The process_node and live_viewer apply presentation policy: `normal` is clamped to `[0,1]`, `shear` is clamped to `[-1,1]`.
- `force_field_scale` is a **display-only multiplier** used for visualization consistency; it is not a physical calibration.
- RGB mapping is centralized as `(R,G,B)=(Fx,Fy,Fz)` via `force_field_to_rgb` in `viz_utils`.
- `visualize_force_field` clips out-of-range `normal` values to `[0,1]` (no `[-1,1]` remap fallback).

## Performance Characteristics

| Component | GPU (CUDA) | CPU |
|-----------|------------|-----|
| Depth MLP | ~1-2ms | ~10-20ms |
| Force ViT | ~30-50ms | ~500-1000ms |
| Force Decoder | ~15-25ms | Included above |
| **Total (both)** | **~50-80ms** | **~500-1000ms** |

**Memory Usage**: ~500MB GPU VRAM for force estimation

## License

Force estimation feature uses [Sparsh](https://github.com/facebookresearch/sparsh) models licensed under **CC-BY-NC 4.0** (research/non-commercial use only). Depth reconstruction and other features retain original license.


## References
1. Huang, Hung-Jui and Kaess, Michael and Yuan, Wenzhen, "NormalFlow: Fast, Robust, and Accurate Contact-based Object 6DoF Pose Tracking with Vision-based Tactile Sensors," IEEE Robotics and Automation Letters, 2024.
2. Akhter, Mohammadreza et al., "Sparsh: Self-supervised touch representations for vision-based tactile sensing," Conference on Robot Learning (CoRL), 2024. [GitHub](https://github.com/facebookresearch/sparsh) | [Paper](https://arxiv.org/abs/2410.24090)
3. Lambeta, Mike et al., "DIGIT: A novel design for a low-cost compact high-resolution tactile sensor with application to in-hand manipulation," IEEE Robotics and Automation Letters, 2020. [Website](https://digit.ml/)