"""
VisTac SDK: Visual-Tactile Sensor Processing

This package provides depth reconstruction and force estimation for vision-based
tactile sensors (DIGIT). It includes both low-level estimators and high-level
streaming interfaces.

Main Classes:
- LiveTactileProcessor: High-level threaded streaming interface (recommended)
- TactileProcessor: Low-level processor with selective output computation
- DepthEstimator: MLP-based depth reconstruction from tactile images
- ForceEstimator: Sparsh ViT-based force estimation from temporal pairs
- TemporalBuffer: Circular buffer for temporal frame management

Example:
    >>> from vistac_sdk import LiveTactileProcessor
    >>> processor = LiveTactileProcessor(serial="D21273", enable_force=True)
    >>> processor.start()
    >>> frame, result = processor.get_latest_output()
    >>> # result = {'depth': ..., 'force_field': ..., 'force_vector': ...}
"""

__version__ = "1.0.0"

# High-level streaming interface
from .live_core import LiveTactileProcessor

# Low-level processors
from .tactile_processor import TactileProcessor
from .vistac_reconstruct import DepthEstimator
from .vistac_force import ForceEstimator

# Utilities
from .temporal_buffer import TemporalBuffer

# Visualization
from .viz_utils import (
    plot_gradients,
    visualize_force_field,
    visualize_force_vector
)

__all__ = [
    # Main interfaces
    "LiveTactileProcessor",
    "TactileProcessor",
    
    # Estimators
    "DepthEstimator",
    "ForceEstimator",
    
    # Utilities
    "TemporalBuffer",
    
    # Visualization
    "plot_gradients",
    "visualize_force_field",
    "visualize_force_vector",
]
