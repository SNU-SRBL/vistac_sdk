"""
Unified tactile processor combining depth and force estimation.

This module provides a unified interface for selective computation of depth
reconstruction and force estimation outputs from tactile sensor images.
Processing is fully synchronous -- call :meth:`process` directly.
"""

from typing import Dict, List, Optional, Union

import numpy as np

from .digit_reconstruct import DepthEstimator
from .digit_force import ForceEstimator


class TactileProcessor:
    """Synchronous processor for selective depth/force estimation.

    Combines DepthEstimator and ForceEstimator with a unified API.
    No threads, no locks -- call :meth:`process` directly in your event loop.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 enable_depth: bool = True,
                 enable_force: bool = True,
                 force_encoder_path: str = 'models/sparsh_dino_base_encoder.ckpt',
                 force_decoder_path: str = 'models/sparsh_digit_forcefield_decoder.pth',
                 temporal_stride: int = 5,
                 bg_offset: float = 0.5,
                 device: str = 'cuda',
                 ppmm: Optional[float] = None,
                 contact_mode: str = 'standard',
                 force_field_baseline: bool = False,
                 force_vector_scale: Optional[Union[list, tuple]] = None):
        """Initialize TactileProcessor.

        Args:
            model_path: Path to depth MLP model (required if enable_depth=True)
            enable_depth: Enable depth estimation
            enable_force: Enable force estimation
            force_encoder_path: Path to Sparsh encoder (.ckpt)
            force_decoder_path: Path to Sparsh decoder (.pth)
            temporal_stride: Frames between temporal pair for force estimation
            bg_offset: Background subtraction offset for force estimation
            device: 'cuda' or 'cpu'
            ppmm: Pixels per mm for depth estimation (can be set later)
            contact_mode: Contact mode ('standard' or 'flat')
        """
        self.enable_depth = enable_depth
        self.enable_force = enable_force
        self.device = device
        self.ppmm = ppmm

        # Estimators
        self.depth_estimator = None
        self.force_estimator = None

        if enable_depth:
            if model_path is None:
                raise ValueError("model_path required when enable_depth=True")
            self.depth_estimator = DepthEstimator(
                model_path=model_path,
                contact_mode=contact_mode,
                device=device,
            )

        if enable_force:
            self.force_estimator = ForceEstimator(
                encoder_path=force_encoder_path,
                decoder_path=force_decoder_path,
                temporal_stride=temporal_stride,
                bg_offset=bg_offset,
                device=device,
                force_field_baseline=force_field_baseline,
                force_vector_scale=force_vector_scale,
            )

        # Background loaded flag
        self._bg_loaded = False

    def load_background(self, bg_image: np.ndarray):
        """Load background image for both estimators.

        Args:
            bg_image: [H, W, 3] BGR background image (uint8)
        """
        if self.depth_estimator is not None:
            self.depth_estimator.load_bg(bg_image)

        if self.force_estimator is not None:
            self.force_estimator.load_background(bg_image)

        self._bg_loaded = True

    def set_ppmm(self, ppmm: float):
        """Set pixels per mm for depth estimation."""
        self.ppmm = ppmm

    def process(self,
                image: np.ndarray,
                outputs: Optional[List[str]] = None,
                timestamp: Optional[float] = None,
                ppmm: Optional[float] = None,
                **depth_kwargs) -> Dict[str, Union[np.ndarray, Dict, None]]:
        """Process image and return requested outputs synchronously.

        Args:
            image: [H, W, 3] BGR image (uint8)
            outputs: List of outputs to compute. Options:
                'depth', 'gradient', 'pointcloud', 'mask',
                'force_field', 'force_vector'.
                Default: ['depth'] if only depth enabled.
            timestamp: Optional timestamp for force temporal tracking
            ppmm: Pixels per mm (overrides constructor value)
            **depth_kwargs: Additional arguments for depth estimation

        Returns:
            Dictionary with requested outputs.
        """
        if outputs is None:
            outputs = []
            if self.enable_depth:
                outputs.append('depth')
            if self.enable_force:
                outputs.extend(['force_field', 'force_vector'])

        depth_outputs = {'depth', 'gradient', 'pointcloud', 'mask'}
        force_outputs = {'force_field', 'force_vector'}

        for output in outputs:
            if output in depth_outputs and not self.enable_depth:
                raise ValueError(
                    f"Depth output '{output}' requested but enable_depth=False")
            if output in force_outputs and not self.enable_force:
                raise ValueError(
                    f"Force output '{output}' requested but enable_force=False")

        result = {}

        # Depth
        requested_depth_outputs = [o for o in outputs if o in depth_outputs]
        if requested_depth_outputs:
            _ppmm = ppmm if ppmm is not None else self.ppmm
            if _ppmm is None:
                raise ValueError("ppmm must be provided for depth estimation")
            depth_result = self.depth_estimator.estimate(
                image=image,
                outputs=requested_depth_outputs,
                ppmm=_ppmm,
                **depth_kwargs,
            )
            result.update(depth_result)

        # Force
        if any(o in force_outputs for o in outputs):
            force_result = self.force_estimator.estimate(
                image=image, timestamp=timestamp)
            if force_result is None:
                if 'force_field' in outputs:
                    result['force_field'] = None
                if 'force_vector' in outputs:
                    result['force_vector'] = None
            else:
                if 'force_field' in outputs:
                    result['force_field'] = force_result['force_field']
                if 'force_vector' in outputs:
                    result['force_vector'] = force_result['force_vector']
                    result['force_vector_physical'] = \
                        force_result.get('force_vector_physical')

        return result

    def is_background_loaded(self) -> bool:
        """Check if background has been loaded."""
        return self._bg_loaded
