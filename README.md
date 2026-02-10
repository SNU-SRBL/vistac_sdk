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

This will download:
- **Encoder**: `sparsh-dino-base` (ViT-base, ~1.7 GB)
- **Decoder**: `sparsh-digit-forcefield` (~15 MB)

Models are saved to `models/` directory.

**Requirements**:
- GPU recommended (CUDA-capable) for real-time performance (~50-80ms)
- CPU fallback available (slower, ~500-1000ms per frame)

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
python vistac_sdk/test_camera.py
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

#### Depth-Only Mode (Default)
```python
from vistac_sdk import LiveTactileProcessor

processor = LiveTactileProcessor(
    serial="D21273",
    enable_depth=True,
    enable_force=False  # Default
)

processor.start()
frame, result = processor.get_latest_output()
# result = {'depth': ndarray, 'gradient': ndarray, 'pointcloud': ndarray}
```

#### Force-Only Mode
```python
from vistac_sdk import LiveTactileProcessor

processor = LiveTactileProcessor(
    serial="D21273",
    enable_depth=False,
    enable_force=True
)

processor.start()
# Wait for temporal buffer warmup (5+ frames)
frame, result = processor.get_latest_output()
# result = {
#     'force_field': {'normal': ndarray, 'shear': ndarray},
#     'force_vector': {'fx': float, 'fy': float, 'fz': float}
# }
```

#### Combined Mode with Selective Outputs
```python
from vistac_sdk import TactileProcessor
import cv2

processor = TactileProcessor(
    model_path="sensors/D21273/model/nnmodel.pth",
    enable_depth=True,
    enable_force=True
)

# Load background
bg_image = cv2.imread("sensors/D21273/calibration/background_data.npz")
processor.load_background(bg_image)

# Selective computation (only depth this frame)
result = processor.process(image, outputs=['depth'])
# Only depth estimator runs, force estimator skipped

# Only force this frame
result = processor.process(image, outputs=['force_vector'])
# Only force estimator runs, depth estimator skipped

# Both outputs
result = processor.process(image, outputs=['depth', 'force_field', 'force_vector'])
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
- `/tactile/{serial}/force_vector` - `geometry_msgs/WrenchStamped`

## Architecture

### Modular Design
- **DepthEstimator**: Fast MLP-based depth reconstruction (~1-2ms)
- **ForceEstimator**: Sparsh ViT-based force estimation (~50-80ms on GPU)
- **TactileProcessor**: Unified interface with selective output computation
- **LiveTactileProcessor**: Threaded streaming with background processing

### Force Estimation Pipeline
1. **Temporal Buffering**: Maintains circular buffer of recent frames
2. **Background Subtraction**: Normalizes contact appearance (configurable offset)
3. **Encoder**: ViT-base (768-dim) extracts visual features from temporal pairs
4. **Decoder**: DPT-style multi-scale fusion predicts normal/shear force fields
5. **Aggregation**: Spatial averaging produces force vector (Fx, Fy, Fz)

### Selective Execution
Only requested outputs are computed per frame, enabling efficient performance:
- Depth-only: ~1-2ms
- Force-only: ~50-80ms (GPU) or ~500-1000ms (CPU)
- Combined: ~50-80ms (GPU, force dominates)

### Output Formats
**Depth outputs**:
- `depth`: `[H, W]` uint8, depth in mm
- `gradient`: `[H, W, 2]` float32, surface gradients
- `pointcloud`: `[N, 3]` float32, XYZ coordinates in meters
- `pointcloud_colors` (optional): `[N, 3]` float32 RGB colors derived from force field (R=fx, G=fy, B=fz)
- `pointcloud_forces` (optional): `[N, 3]` float32 per-point force values (fx, fy, fz)
- `mask`: `[H, W]` bool, contact mask

**Force outputs** (None during warmup):
- `force_field`: dict with `normal` [224, 224] and `shear` [224, 224, 2]
- `force_vector`: dict with `fx`, `fy`, `fz` scalars (normalized [-1, 1])

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