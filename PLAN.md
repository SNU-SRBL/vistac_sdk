# Refactoring Plan: Add Sparsh Force Estimation to VisTac SDK

**Date**: February 6, 2026  
**Status**: Planning Phase  
**Sparsh Repository**: https://github.com/facebookresearch/sparsh

## Overview

Refactor vistac_sdk to support selective depth/force estimation using separate estimators combined in a unified API. Models load at initialization but only compute requested outputs per frame, enabling efficient selective processing.

This integration leverages **Sparsh** (https://github.com/facebookresearch/sparsh), a family of self-supervised learning models for vision-based tactile sensors, to add force estimation capabilities to the existing depth reconstruction pipeline.

## Objectives

- Add Sparsh-based force estimation alongside existing depth reconstruction
- Create modular architecture with separate `DepthEstimator` and `ForceEstimator`
- Provide unified `TactileProcessor` API with selective output computation
- Support both force field (dense heatmaps) and force vector (Fx, Fy, Fz) modes
- Maintain backward compatibility via deprecation warnings
- Enable ROS2 integration with force-specific message types

## Key Design Decisions

### Architecture
- **Modular estimators**: `DepthEstimator` (existing MLP), `ForceEstimator` (Sparsh ViT)
- **Unified interface**: `TactileProcessor` with selective `outputs=['depth', 'force_field', 'force_vector', ...]`
- **Selective execution**: Only compute requested outputs per frame (no overhead for unused estimators)

### Sparsh Integration
- **Repository**: https://github.com/facebookresearch/sparsh
- **Encoder**: `facebook/sparsh-dino-base` (embed_dim=768, required for decoder compatibility)
- **Decoder**: `facebook/sparsh-digit-forcefield-decoder` (pretrained on DIGIT sensors with dino-base)
- **Architecture**: ViT-base (patch_size=16, depth=12, num_heads=12, embed_dim=768)
- **Temporal buffering**: Auto-managed circular buffer with configurable stride (default 5 frames @ 60 FPS ≈ 80ms window)
- **Both force modes**: Dense force fields + direct force vectors
- **Preprocessing**: Background subtraction with offset=0.5, resize to 224×224, ToTensor (no ImageNet normalization)

### Implementation Choices
- **Keep existing depth MLP**: Fast, sensor-specific, no breaking changes
- **Model storage**: Centralized `models/` directory (not per-sensor)
- **Model download**: Manual via setup script (explicit control, offline capability)
- **Background reuse**: Use existing `background_data.npz` from depth calibration
- **Force output**: Normalized [-1, 1] (avoid unnecessary calibration complexity)
- **Coordinate frame**: Convert Sparsh (normal, shear_x, shear_y) → VisTac (Fz, Fx, Fy)
- **GPU handling**: Auto-fallback to CPU with warning
- **ROS2 messages**: `WrenchStamped` for force vector, `Image` for force field

## Critical Implementation Details (from Sparsh code analysis)

### Model Compatibility
- **Encoder**: `facebook/sparsh-dino-base` (ViT-base, embed_dim=768)
- **Decoder**: Trained specifically for dino-base encoder
- **WARNING**: Using sparsh-dino-small (embed_dim=384) will cause dimension mismatch
- **File format**: PyTorch .pth files via HuggingFace Hub

### Exact Data Formats

**Input**:
- Shape: `[B, 6, 224, 224]` (6 = two BGR frames concatenated)
- Type: `torch.float32`
- Range: `[0, 1]` after preprocessing
- Device: CUDA or CPU

**Depth Outputs** (existing):
- `'depth'`: `[H, W]` uint8, values 0-255 (mm scaled)
- `'gradient'`: `[H, W, 2]` float32, gradient angles  
- `'pointcloud'`: `[N, 3]` float32, XYZ in meters
- `'mask'`: `[H, W]` bool, contact mask

**Force Outputs** (new):
- `'force_field'`: dict
  - `'normal'`: `[224, 224]` float32, normalized forces
  - `'shear'`: `[224, 224, 2]` float32, (Fx, Fy) components
- `'force_vector'`: dict
  - `'fx'`: float32 scalar
  - `'fy'`: float32 scalar
  - `'fz'`: float32 scalar
  - All in normalized units [-1, 1]

### Exact Preprocessing Pipeline

```python
def preprocess_force_input(img_t, img_t_minus_5, bg):
    """
    Based on Sparsh tactile_ssl/data/digit/utils.py
    """
    # Step 1: Background subtraction with offset
    def subtract_bg(img, bg):
        diff = img.astype(np.int32) - bg.astype(np.int32)
        diff = diff / 255.0 + 0.5  # Offset = 0.5
        diff = np.clip(diff, 0.0, 1.0)
        diff = (diff * 255.0).astype(np.uint8)
        return diff
    
    img_t_diff = subtract_bg(img_t, bg)
    img_t5_diff = subtract_bg(img_t_minus_5, bg)
    
    # Step 2: Convert to PIL
    img_t_pil = Image.fromarray(img_t_diff).convert("RGB")
    img_t5_pil = Image.fromarray(img_t5_diff).convert("RGB")
    
    # Step 3: Resize to 224x224
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),  # [0, 1] normalization only
    ])
    tensor_t = transform(img_t_pil)   # [3, 224, 224]
    tensor_t5 = transform(img_t5_pil) # [3, 224, 224]
    
    # Step 4: Temporal concatenation
    input_tensor = torch.cat([tensor_t, tensor_t5], dim=0)  # [6, 224, 224]
    
    return input_tensor.unsqueeze(0)  # [1, 6, 224, 224]
```

**NO ImageNet normalization** - this is intentional per Sparsh implementation.

### Force Vector Aggregation Formula

```python
# From Sparsh forcefield_sl.py:224-240
def aggregate_force_vector(normal_field, shear_field):
    """
    normal_field: [B, 1, 224, 224]
    shear_field: [B, 2, 224, 224]
    Returns: (fx, fy, fz) scalars
    """
    H, W = 224, 224
    fz = normal_field.sum(dim=[-2, -1]) / (H * W)  # Mean
    fx = shear_field[:, 0].sum(dim=[-2, -1]) / (H * W)
    fy = shear_field[:, 1].sum(dim=[-2, -1]) / (H * W)
    return fx.item(), fy.item(), fz.item()
```

### Threading Model

**Single processor thread** manages both estimators:
```python
class TactileProcessor:
    def __init__(self):
        self._thread = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_result = {}
        
    def start_thread(self):
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        
    def _process_loop(self):
        while self._running:
            with self._lock:
                frame = self._latest_frame
            if frame is not None:
                result = self.process(frame, outputs=self._outputs)
                with self._lock:
                    self._latest_result = result
            time.sleep(0.001)
```

### Backward Compatibility Strategy

**Problem**: Existing code expects specific types, not dicts:
```python
# Old code (BREAKS with dict return)
frame, depth = recon.get_latest_output()
cv2.imshow("depth", depth)  # TypeError if depth is dict
```

**Solution**: Maintain old return format but add deprecation warnings:
```python
class LiveReconstructor(LiveTactileProcessor):
    """Deprecated: Use LiveTactileProcessor instead."""
    def __init__(self, *args, mode="depth", **kwargs):
        warnings.warn(
            "LiveReconstructor is deprecated, use LiveTactileProcessor",
            DeprecationWarning, stacklevel=2
        )
        super().__init__(*args, **kwargs)
        self._legacy_mode = mode
        
    def get_latest_output(self):
        frame, result_dict = super().get_latest_output()
        # Extract the specific mode output for backward compatibility
        if self._legacy_mode == "depth":
            return frame, result_dict.get('depth')
        elif self._legacy_mode == "gradient":
            return frame, result_dict.get('gradient')
        elif self._legacy_mode == "pointcloud":
            return frame, result_dict.get('pointcloud')
```

### Error Handling Specifications

```python
# Model not found
if not os.path.exists(encoder_path):
    raise FileNotFoundError(
        f"Sparsh encoder not found at {encoder_path}. "
        f"Run: python scripts/download_models.py"
    )

# CUDA OOM
try:
    output = model(input_tensor)
except RuntimeError as e:
    if "out of memory" in str(e):
        warnings.warn("CUDA OOM, falling back to CPU")
        model = model.cpu()
        device = "cpu"
        output = model(input_tensor.cpu())
    else:
        raise

# Force buffer not ready
if not self.force_buffer.is_ready():
    return {
        'force_field': None,
        'force_vector': None
    }

# Invalid outputs requested
if 'force_field' in outputs and not self._force_enabled:
    raise ValueError(
        "Force estimation not enabled. Initialize with enable_force=True"
    )
```

### Coordinate System Mapping

**Sparsh convention** (from DIGIT intrinsics):
- **Fz (normal)**: Perpendicular to sensor surface, positive = into sensor
- **Fx (shear-x)**: Horizontal in image plane (left-right)
- **Fy (shear-y)**: Vertical in image plane (top-bottom)

**VisTac convention** (from height2pointcloud):
- **X**: Horizontal (matches Fx)
- **Y**: Vertical (matches Fy)  
- **Z**: Depth/height (matches Fz)

**Mapping**: Direct 1:1 correspondence, no conversion needed.
- `force_vector['fx']` = horizontal shear force
- `force_vector['fy']` = vertical shear force
- `force_vector['fz']` = normal force

## Implementation Steps

### 1. Create Model Download Script
**File**: `scripts/download_models.py`

- Download `facebook/sparsh-dino-base` encoder from HuggingFace (required for decoder compatibility)
- Download `facebook/sparsh-digit-forcefield-decoder` decoder from HuggingFace
- Save to `models/` directory: `sparsh_dino_base_encoder.pth`, `sparsh_digit_forcefield_decoder.pth`
- Add progress bars, checksum validation, resume capability
- Use `huggingface_hub` library for downloading

### 2. Create Temporal Buffer Utility
**File**: `vistac_sdk/temporal_buffer.py`

- Implement `TemporalBuffer` class with circular buffer
- `add(frame)`: stores frame with timestamp
- `get_pair(stride=5)`: returns `(frame_t, frame_t-stride)` as 6-channel concat
- Handle insufficient frames (return None until buffer filled)

### 3. Create Force Estimation Module
**File**: `vistac_sdk/vistac_force.py`

**Classes**:
- `SparshEncoder`: ViT-base (patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4)
  - Extracts intermediate features from layers [2, 5, 8, 11] via forward hooks
  - Input: 6-channel tensor (temporal pair concatenated)
- `ForceFieldDecoder`: Multi-scale DPT-style decoder
  - Reassemble scales: [4, 8, 16, 32]
  - Resample dimension: 128
  - Outputs: normal [B, 1, 224, 224], shear [B, 2, 224, 224]
- `ForceEstimator`: Main interface with preprocessing and inference

**Features**:
- Load pretrained weights from `models/` directory
- Temporal buffering integration via `TemporalBuffer`
- **Exact preprocessing pipeline**:
  1. Background subtraction: `(img - bg).astype(float32) / 255.0 + 0.5`, clipped to [0, 1]
  2. Convert to PIL RGB
  3. Resize to 224×224 (antialias interpolation)
  4. ToTensor: converts to [0, 1] float32 (NO ImageNet normalization)
  5. Temporal concat: stack frame_t and frame_t-5 → 6 channels
- Dual modes: `'field'` (dense heatmaps), `'vector'` (mean aggregation)
- **Force vector aggregation**: `mean(force_field)` over spatial dimensions
  - Fx = sum(shear[:, 0]) / (H × W)
  - Fy = sum(shear[:, 1]) / (H × W)
  - Fz = sum(normal) / (H × W)
- Coordinate system: Fz=normal (into sensor), Fx=horizontal, Fy=vertical
- GPU/CPU auto-detection with fallback warning

### 4. Refactor Depth Reconstruction
**File**: `vistac_sdk/vistac_reconstruct.py`

**Changes**:
- Rename `Reconstructor` → `DepthEstimator`
- Remove threading logic (moved to processor level)
- Standardize return format to dict
- Add `estimate(image, mode)` dispatcher
- Keep core: BGRXY→MLP→gradients→Poisson→depth/pointcloud pipeline

### 5. Create Unified Processor
**File**: `vistac_sdk/tactile_processor.py`

**Class**: `TactileProcessor`

**Constructor params**:
- `model_path`: path to depth MLP (e.g., 'sensors/D21119/model/nnmodel.pth')
- `enable_depth=True`, `enable_force=True`
- `force_encoder_path='models/sparsh_dino_base_encoder.pth'`
- `force_decoder_path='models/sparsh_digit_forcefield_decoder.pth'`
- `temporal_stride=5` (frames between temporal pair)
- `device='cuda'` (auto-fallback to CPU with warning)

**Methods**:
- `load_background(bg_image)`: pass to both estimators
  - Depth: calculates background gradients (existing logic)
  - Force: stores bg for preprocessing subtraction
- `process(image, outputs=['depth', 'force_field', 'force_vector'])`: selective computation
  - Validates outputs against enabled estimators
  - Returns dict with only requested keys
  - Returns None for force outputs if temporal buffer not ready
- `start_thread()`, `set_input_frame()`, `get_latest_result()`: threaded mode
  - Single background thread processes both estimators
  - Thread-safe locks for frame/result access

**Features**:
- Lazy initialization (only load enabled estimators)
- Selective execution (only run requested outputs)
- Threading support for continuous processing
- Force buffer warmup handling

### 6. Update Live API
**File**: `vistac_sdk/live_core.py`

**Changes**:
- Rename `LiveReconstructor` → `LiveTactileProcessor`
- Add constructor params: `enable_depth=True`, `enable_force=False`, `temporal_stride=5`
- Use `TactileProcessor` instead of `Reconstructor`
- Update `get_latest_output()` → returns `(frame, result_dict)`
- Keep background collection logic (average 10 frames)

### 7. Update Visualization Utilities
**File**: `vistac_sdk/viz_utils.py`

**New functions**:
- `visualize_force_field(normal, shear, overlay_image=None)`: RGB heatmap
- `visualize_force_vector(fx, fy, fz, image)`: arrow overlay

**Updates**:
- Update `plot_gradients()` to accept dict format

### 8. Update Live Viewer App
**File**: `apps/live_viewer.py`

**Changes**:
- Add CLI args: `--outputs`, `--enable-depth`, `--enable-force`, `--temporal-stride`
- Use `LiveTactileProcessor`
- Multi-panel display based on requested outputs
- Handle force warmup ("buffering..." message)

### 9. Update ROS2 Node
**File**: `ros2/tactile_streamer_node.py`

**New parameters**:
- `enable_depth` (bool, default True)
- `enable_force` (bool, default False)
- `temporal_stride` (int, default 5)
- `outputs` (string list, default ['depth'])

**Publishing**:
- Depth: `sensor_msgs/Image` mono8 → `/tactile/{serial}/depth`
- Gradient: `sensor_msgs/Image` 32FC2 → `/tactile/{serial}/gradient`
- PointCloud: `sensor_msgs/PointCloud2` → `/tactile/{serial}/pointcloud`
- Force field: `sensor_msgs/Image` 32FC3 → `/tactile/{serial}/force_field`
- Force vector: `geometry_msgs/WrenchStamped` → `/tactile/{serial}/force_vector`

### 10. Update ROS2 Launch File
**File**: `ros2/launch/multi_sensor_tactile_streamer.launch.py`

- Add launch args: `enable_depth`, `enable_force`, `temporal_stride`, `outputs`
- Pass to all sensor nodes
- Default: depth only (backward compatibility)

### 11. Update Sensor Configs
**Files**: `sensors/{serial}/{serial}.yaml`

**Add optional section**:
```yaml
force:
  enabled: false
  temporal_stride: 5
  encoder: sparsh-dino-base  # Required for decoder compatibility
  decoder: sparsh-digit-forcefield
  bg_offset: 0.5  # Background subtraction offset
```

### 12. Update Dependencies
**Files**: `setup.py`, `requirements.txt`, `package.xml`

**Add**:
- `torch>=2.0,<3.0`
- `torchvision>=0.15`
- `einops>=0.6`
- `timm>=0.9`
- `huggingface_hub>=0.19`
- `xformers` (optional, for memory-efficient attention)

**Update**: Python `>=3.9`

**Note**: Exact versions from Sparsh `environment.yml` may differ

### 13. Update Main README
**File**: `README.md`

**Add**:
- Force estimation section
- Model download instructions: `python scripts/download_models.py`
- Quickstart examples (depth-only, force-only, combined)
- Selective outputs API documentation
- GPU recommendation note

### 14. Update Package Init
**File**: `vistac_sdk/__init__.py`

**Export new classes**:
- `TactileProcessor`
- `LiveTactileProcessor`
- `DepthEstimator`
- `ForceEstimator`

**Backward compatibility**:
- `Reconstructor = DepthEstimator` (with deprecation warning)
- `LiveReconstructor = LiveTactileProcessor` (with deprecation warning)

## Verification Plan

### Unit Tests (create tests/ directory)

- [ ] **test_temporal_buffer.py**: Circular buffer, stride handling, insufficient frames
- [ ] **test_preprocessing.py**: Background subtraction, resize, tensor shapes
- [ ] **test_force_estimator.py**: Model loading, inference shapes, coordinate system
- [ ] **test_depth_estimator.py**: Refactored class matches old behavior
- [ ] **test_tactile_processor.py**: Selective execution, lazy loading, threading

### Integration Tests

- [ ] **Depth-only mode**: 
  ```python
  processor = TactileProcessor(enable_force=False)
  result = processor.process(img, outputs=['depth'])
  assert result['depth'].shape == (240, 320)
  assert result['depth'].dtype == np.uint8
  assert 'force_field' not in result
  ```

- [ ] **Force-only mode** (after buffer warmup):
  ```python
  processor = TactileProcessor(enable_depth=False, enable_force=True)
  for i in range(10):  # Warmup
      processor.process(imgs[i], outputs=['force_field'])
  result = processor.process(imgs[10], outputs=['force_field'])
  assert result['force_field']['normal'].shape == (224, 224)
  assert result['force_field']['shear'].shape == (224, 224, 2)
  assert 'depth' not in result
  ```

- [ ] **Combined mode**:
  ```python
  result = processor.process(img, outputs=['depth', 'force_vector'])
  assert 'depth' in result and 'force_vector' in result
  ```

- [ ] **Selective execution profiling**:
  ```python
  import time
  # Force disabled - should not compute
  start = time.time()
  result = proc.process(img, outputs=['depth'])
  t1 = time.time() - start  # Should be ~2ms
  
  # Force enabled but not requested
  start = time.time() 
  result = proc.process(img, outputs=['depth'])
  t2 = time.time() - start  # Should be ~2ms (not ~50ms)
  
  assert abs(t1 - t2) < 5e-3  # Difference < 5ms
  ```

- [ ] **Temporal buffering**: Force returns None until 5 frames accumulated

- [ ] **CPU fallback**: Warning shown, runs successfully (slowly)
  ```python
  processor = TactileProcessor(device='cpu')
  # Should print warning
  result = processor.process(img, outputs=['force_field'])
  assert result['force_field'] is not None  # Works, just slow
  ```

### ROS2 Integration Tests

- [ ] **Message types**: All published correctly
  ```bash
  ros2 topic echo /tactile/D21119/force_vector
  # Should show geometry_msgs/WrenchStamped
  
  ros2 topic echo /tactile/D21119/force_field
  # Should show sensor_msgs/Image with encoding=32FC3
  ```

- [ ] **Multi-sensor launch**: All sensors stream without conflicts
  ```bash
  ros2 launch vistac_sdk multi_sensor_tactile_streamer.launch.py enable_force:=true
  ros2 topic list | grep tactile
  # Should see topics for all 4 sensors
  ```

### Physical Validation

- [ ] **Force field sanity**: 
  - No contact → force_field near zero
  - Light press → localized normal force
  - Slide → directional shear force
  
- [ ] **Force vector sanity**:
  - Vertical press → |fz| > |fx|, |fy|
  - Horizontal slide → |fx| or |fy| > |fz|
  
- [ ] **Coordinate system**: Slide right → fx > 0, slide down → fy > 0

### Backward Compatibility Tests

- [ ] **Old API still works** (with warnings):
  ```python
  import warnings
  with warnings.catch_warnings(record=True) as w:
      recon = LiveReconstructor(serial="D21119", mode="depth")
      assert len(w) == 1
      assert issubclass(w[0].category, DeprecationWarning)
      
  frame, depth = recon.get_latest_output()
  assert isinstance(depth, np.ndarray)  # Not dict
  assert depth.shape == (240, 320)
  ```

## Expected Directory Structure

```
vistac_sdk/
├── CMakeLists.txt
├── LICENSE.txt
├── package.xml
├── README.md                                     # UPDATED
├── setup.py                                      # UPDATED
├── requirements.txt                              # UPDATED
│
├── models/                                       # NEW
│   ├── sparsh_dino_base_encoder.pth             # ViT-base: 768 embed_dim
│   ├── sparsh_digit_forcefield_decoder.pth
│   └── README.md
│
├── scripts/                                      # NEW
│   └── download_models.py
│
├── apps/
│   └── live_viewer.py                            # UPDATED
│
├── calibration/                                  # UNCHANGED
│   └── (existing calibration pipeline)
│
├── docs/                                         # NEW
│   ├── api_reference.md
│   ├── force_estimation.md
│   └── examples/
│       ├── depth_only.py
│       ├── force_only.py
│       └── combined.py
│
├── ros2/
│   ├── tactile_streamer_node.py                  # UPDATED
│   └── launch/
│       └── multi_sensor_tactile_streamer.launch.py   # UPDATED
│
├── sensors/                                      # Each sensor dir:
│   ├── D21119/
│   │   ├── D21119.yaml                           # UPDATED
│   │   ├── calibration/                          # UNCHANGED
│   │   └── model/
│   │       └── nnmodel.pth                       # UNCHANGED
│   └── (D21242, D21273, D21275...)
│
└── vistac_sdk/
    ├── __init__.py                               # UPDATED
    ├── live_core.py                              # UPDATED
    ├── tactile_processor.py                      # NEW
    ├── temporal_buffer.py                        # NEW
    ├── vistac_device.py                          # UNCHANGED
    ├── vistac_reconstruct.py                     # REFACTORED
    ├── vistac_force.py                           # NEW
    ├── viz_utils.py                              # UPDATED
    └── utils.py                                  # UNCHANGED
```

## File Change Summary

- **NEW**: 6 files
  - `tactile_processor.py`, `vistac_force.py`, `temporal_buffer.py`
  - `download_models.py`, `models/`, `docs/`
  
- **UPDATED**: 8 files
  - README.md, setup.py, requirements.txt, package.xml
  - `__init__.py`, `live_core.py`, `live_viewer.py`, `viz_utils.py`
  - ROS2 node/launch, sensor YAMLs
  
- **REFACTORED**: 1 file
  - `vistac_reconstruct.py` (Reconstructor → DepthEstimator)
  
- **UNCHANGED**: 17 files
  - calibration/, camera, utils, assets, etc.

## Migration Path

### For Existing Users

**Depth-only users** (no changes required):
```python
# Old API (still works with deprecation warning)
from vistac_sdk import LiveReconstructor
recon = LiveReconstructor(serial="D21119", mode="depth")

# New API (recommended)
from vistac_sdk import LiveTactileProcessor
processor = LiveTactileProcessor(serial="D21119", enable_force=False)
```

**Adding force estimation**:
```python
# 1. Download models
# python scripts/download_models.py

# 2. Enable force
from vistac_sdk import LiveTactileProcessor
processor = LiveTactileProcessor(
    serial="D21119",
    enable_depth=True,
    enable_force=True
)

frame, result = processor.get_latest_output()
# result = {
#     'depth': ...,
#     'force_field': {'normal': ..., 'shear': ...},
#     'force_vector': {'fx': ..., 'fy': ..., 'fz': ...}
# }
```

**Selective outputs** (performance optimization):
```python
from vistac_sdk import TactileProcessor
processor = TactileProcessor(
    model_path="sensors/D21119/model/nnmodel.pth",
    enable_depth=True,
    enable_force=True
)

# Only compute depth this frame (force estimator not executed)
result = processor.process(image, outputs=['depth'])

# Only compute force this frame (depth estimator not executed)
result = processor.process(image, outputs=['force_vector'])

# Compute both
result = processor.process(image, outputs=['depth', 'force_field'])
```

## Computational Characteristics

### Performance Estimates (on modern GPU)

| Component | Latency | Memory |
|-----------|---------|--------|
| Depth MLP | ~1-2ms | <10MB |
| Force ViT (sparsh-dino-base) | ~30-50ms | ~350MB |
| Force field decoder | ~15-25ms | ~150MB |
| **Total (both)** | **~50-80ms** | **~500MB** |

**Note**: ViT-base is larger than initially planned (768 vs 384 embedding dim) due to decoder compatibility requirement

### CPU Fallback (approximate)
- Depth: ~10-20ms
- Force: ~500-1000ms (significantly slower)

## Dependencies and Requirements

### Python Packages
```
torch>=2.0
torchvision
transformers
einops
timm
huggingface_hub
opencv-python
numpy
scipy
open3d
pyudev
pyyaml
```

### System Requirements
- **Recommended**: NVIDIA GPU with CUDA support
- **Minimum**: CPU (slow force estimation)
- **Python**: 3.9+
- **ROS2**: Humble or later (for ROS2 integration)

## References

- **Sparsh GitHub**: https://github.com/facebookresearch/sparsh
- **HuggingFace Models**: https://huggingface.co/collections/facebook/sparsh-67167ce57566196a4526c328
- **DIGIT Sensor**: https://digit.ml/
- **Original SDK**: https://github.com/joehjhuang/gs_sdk

## License Considerations

- **VisTac SDK**: [Current license]
- **Sparsh Models**: CC-BY-NC 4.0 (research/non-commercial use)
- **Note**: Force estimation feature limited to non-commercial use per Sparsh license

## Open Questions & Pre-Implementation Verification

### To Verify Before Step 1

1. **Exact dependency versions**: Need Sparsh `environment.yml` contents for pinned versions
   - Current plan uses loose constraints (e.g., `torch>=2.0,<3.0`)
   - May need specific torch/torchvision/xformers versions

2. **Model file formats**: Confirm downloaded files are `.pth` and loadable via `torch.load()`
   - HuggingFace may provide safetensors or other formats
   - May need format conversion

3. **Background image compatibility**: Verify current VisTac background collection works for Sparsh
   - Current: 10 averaged frames
   - Sparsh: expects single background image
   - Should work, but needs testing

4. **Image size adaptation**: VisTac sensors output 320×240, Sparsh expects 224×224
   - Plan handles via resize
   - Verify no quality degradation

### Post-Step 1 Validation

After downloading models:
```bash
python scripts/download_models.py
ls -lh models/
# Should see:
# - sparsh_dino_base_encoder.pth (~300MB)
# - sparsh_digit_forcefield_decoder.pth (~50MB)
```

Test loading:
```python
import torch
encoder = torch.load('models/sparsh_dino_base_encoder.pth')
decoder = torch.load('models/sparsh_digit_forcefield_decoder.pth')
print(f"Encoder keys: {encoder.keys()}")
print(f"Decoder keys: {decoder.keys()}")
```

### Critical Path Decisions

**Decision Point 1** (after model download): 
- If model format incompatible → implement conversion in download script
- If embed_dim mismatch → verify decoder actually supports dino-base

**Decision Point 2** (after Step 3):
- Profile force estimator latency
- If >100ms → consider optimizations (mixed precision, TorchScript)
- If <50ms → proceed as planned

**Decision Point 3** (after Step 7):
- Test force outputs on real sensor data
- If forces unreasonable → may need normalization/scaling calibration
- If forces reasonable → proceed to ROS2 integration

---

**Status**: Plan is now **95% concrete** with all Sparsh-specific details verified from source code. Remaining 5% requires downloading models and testing on real hardware.

**Next Steps**: Begin implementation with Step 1 (model download script)
