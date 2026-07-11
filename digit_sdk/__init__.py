"""
Digit SDK: Visual-Tactile Sensor Processing

This package provides depth reconstruction and force estimation for vision-based
tactile sensors (DIGIT). It includes both low-level estimators and high-level
streaming interfaces.

Main Classes:
- TactileProcessor: Unified processor with selective output computation
- Camera: Synchronous DIGIT camera capture
- DepthEstimator: MLP-based depth reconstruction from tactile images
- ForceEstimator: Sparsh ViT-based force estimation from temporal pairs
- TemporalBuffer: Circular buffer for temporal frame management

Example:
    >>> from digit_sdk import TactileProcessor
    >>> processor = TactileProcessor(model_path="model.pth", enable_depth=True)
    >>> processor.load_background(bg_image)
    >>> result = processor.process(image=frame, outputs=['depth', 'pointcloud'])
    >>> # result = {'depth': ..., 'pointcloud': ...}
"""

__version__ = "1.0.0"

# Low-level processors
from .tactile_processor import TactileProcessor
from .processing_engine import ProcessingEngine
from .depth_estimator import DepthEstimator
from .force_estimator import ForceEstimator

# Device interface
from .camera import Camera

# Utilities
from .temporal_buffer import TemporalBuffer

# Visualization
from .viz_utils import (
    plot_gradients,
    force_field_to_rgb,
    visualize_force_field,
    visualize_force_vector
)

__all__ = [
    # Main processors
    "TactileProcessor",

    # Processing engine
    "ProcessingEngine",

    # Camera
    "Camera",

    # Estimators
    "DepthEstimator",
    "ForceEstimator",

    # Utilities
    "TemporalBuffer",

    # Visualization
    "plot_gradients",
    "force_field_to_rgb",
    "visualize_force_field",
    "visualize_force_vector",
]
