# Visual-tactile SDK
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) &nbsp;

This repository is a modified version of [gs_sdk](https://github.com/joehjhuang/gs_sdk) with automatic DIGIT identifying, threaded image collection and calculation, and ROS2 implementation.

Authors:
* [Byung-Hyun Song](https://github.com/bhsong1011) (bh.song@snu.ac.kr)

## Support System
* Tested on Ubuntu 22.04
* Tested on GelSight Mini and Digit
* Python >= 3.9

## Installation
Clone and install gs_sdk from source:
```bash
git clone git@github.com:SNU-SRBL/gs_sdk.git
cd vistac_sdk
pip install -e .
```

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

### Sensor Real-time Reconstruction
Stream reconstructed surface from a connected DIGIT sensor:
```python
python apps/live_viewer.py --serial D21273 --use_mask --mode depth --relative --relative_scale 0.5
```


## References
1. Huang, Hung-Jui and Kaess, Michael and Yuan, Wenzhen, “NormalFlow: Fast, Robust, and Accurate Contact-based Object 6DoF Pose Tracking with Vision-based Tactile Sensors,” IEEE Robotics and Automation Letters, 2024.
