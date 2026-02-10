"""
Unified tactile processor combining depth and force estimation.

This module provides a unified interface for selective computation of depth
reconstruction and force estimation outputs from tactile sensor images.
"""

import threading
import time
import warnings
from typing import Dict, List, Optional, Union

import numpy as np

from .vistac_reconstruct import DepthEstimator
from .vistac_force import ForceEstimator


class TactileProcessor:
    """Unified processor for selective depth/force estimation.
    
    Combines DepthEstimator and ForceEstimator with a unified API that supports:
    - Lazy initialization (only load enabled estimators)
    - Selective execution (only compute requested outputs per frame)
    - Threading support for continuous processing
    - Force buffer warmup handling
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
                 contact_mode: str = 'standard'):
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
            contact_mode: Contact mode for depth estimation ('standard' or 'flat')
        """
        self.enable_depth = enable_depth
        self.enable_force = enable_force
        self.device = device
        self.ppmm = ppmm
        
        # Lazy initialization
        self.depth_estimator = None
        self.force_estimator = None
        
        # Initialize depth estimator if enabled
        if enable_depth:
            if model_path is None:
                raise ValueError("model_path required when enable_depth=True")
            self.depth_estimator = DepthEstimator(
                model_path=model_path,
                contact_mode=contact_mode,
                device=device
            )
        
        # Initialize force estimator if enabled
        if enable_force:
            self.force_estimator = ForceEstimator(
                encoder_path=force_encoder_path,
                decoder_path=force_decoder_path,
                temporal_stride=temporal_stride,
                bg_offset=bg_offset,
                device=device
            )
        
        # Threading state
        self._thread = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame = None
        self._latest_timestamp = None
        self._latest_result = {}
        self._outputs = []  # Thread-safe: set once at thread start
        
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
        """Set pixels per mm for depth estimation.
        
        Args:
            ppmm: Pixels per millimeter
        """
        self.ppmm = ppmm
    
    def process(self,
                image: np.ndarray,
                outputs: Optional[List[str]] = None,
                timestamp: Optional[float] = None,
                ppmm: Optional[float] = None,
                **depth_kwargs) -> Dict[str, Union[np.ndarray, Dict, None]]:
        """Process image and return requested outputs.
        
        Selectively computes only the requested outputs for efficiency.
        
        Args:
            image: [H, W, 3] BGR image (uint8)
            outputs: List of outputs to compute. Options:
                - 'depth': Depth map [H, W] uint8
                - 'gradient': Gradient field [H, W, 2] float32
                - 'pointcloud': Point cloud [N, 3] float32
                - 'mask': Contact mask [H, W] bool
                - 'force_field': {'normal': [224, 224], 'shear': [224, 224, 2]}
                - 'force_vector': {'fx': float, 'fy': float, 'fz': float}
                Default: ['depth'] if only depth enabled, ['force_field', 'force_vector'] if only force enabled
            timestamp: Optional timestamp for force temporal tracking
            ppmm: Pixels per mm (overrides constructor value)
            **depth_kwargs: Additional arguments for depth estimation
            
        Returns:
            Dictionary with requested outputs. Force outputs may be None during warmup.
        """
        # Default outputs based on enabled estimators
        if outputs is None:
            outputs = []
            if self.enable_depth:
                outputs.append('depth')
            if self.enable_force:
                outputs.extend(['force_field', 'force_vector'])
        
        # Validate outputs
        depth_outputs = {'depth', 'gradient', 'pointcloud', 'mask'}
        force_outputs = {'force_field', 'force_vector'}
        
        for output in outputs:
            if output in depth_outputs and not self.enable_depth:
                raise ValueError(
                    f"Depth output '{output}' requested but enable_depth=False"
                )
            if output in force_outputs and not self.enable_force:
                raise ValueError(
                    f"Force output '{output}' requested but enable_force=False"
                )
        
        result = {}
        
        # Compute depth outputs if requested
        requested_depth_outputs = [o for o in outputs if o in depth_outputs]
        if requested_depth_outputs:
            # Use provided ppmm or instance ppmm
            _ppmm = ppmm if ppmm is not None else self.ppmm
            if _ppmm is None:
                raise ValueError("ppmm must be provided for depth estimation")
            
            depth_result = self.depth_estimator.estimate(
                image=image,
                outputs=requested_depth_outputs,
                ppmm=_ppmm,
                **depth_kwargs
            )
            result.update(depth_result)
        
        # Compute force outputs if requested
        if any(o in force_outputs for o in outputs):
            force_result = self.force_estimator.estimate(
                image=image,
                timestamp=timestamp
            )
            
            # During warmup, force_result is None
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

                # If pointcloud and force_field requested together, compute per-point colors
                if 'pointcloud' in result and result['pointcloud'] is not None and 'force_field' in outputs:
                    try:
                        import cv2 as _cv2
                        normal = force_result['force_field']['normal']  # [224,224]
                        shear = force_result['force_field']['shear']    # [224,224,2]
                        # Build RGB: R=Fx, G=Fz, B=Fy
                        normal_n = np.clip((normal + 1.0) / 2.0, 0.0, 1.0)
                        sx_n = np.clip((shear[..., 0] + 1.0) / 2.0, 0.0, 1.0)
                        sy_n = np.clip((shear[..., 1] + 1.0) / 2.0, 0.0, 1.0)
                        # Map forces to RGB: R=Fx, G=Fy, B=Fz (fx->r, fy->g, fz->b)
                        force_rgb = np.stack([sx_n * 255.0, sy_n * 255.0, normal_n * 255.0], axis=-1).astype(np.uint8)
                        # Resize to input image resolution
                        th, tw = image.shape[0], image.shape[1]
                        fh, fw = force_rgb.shape[:2]
                        if (fh, fw) != (th, tw):
                            force_rgb = _cv2.resize(force_rgb, (tw, th), interpolation=_cv2.INTER_NEAREST)
                        colors_flat = force_rgb.reshape(-1, 3) / 255.0
                        pc = result['pointcloud']
                        # If pointcloud was masked (fewer points), use mask if available
                        mask = result.get('mask')
                        if mask is not None and pc.shape[0] != (th * tw):
                            mask_flat = mask.ravel()
                            if mask_flat.shape[0] == th * tw:
                                colors_flat = colors_flat[mask_flat]
                        # Store per-point colors in result dict
                        result['pointcloud_colors'] = colors_flat

                        # Also store raw per-point force values (fx, fy, fz) in same order
                        # fx = shear_x, fy = shear_y, fz = normal
                        fx = shear[..., 0]
                        fy = shear[..., 1]
                        fz = normal
                        # Resize and flatten similar to colors
                        fx_img = fx
                        fy_img = fy
                        fz_img = fz
                        if (fx_img.shape[0], fx_img.shape[1]) != (th, tw):
                            fx_img = _cv2.resize(fx_img, (tw, th), interpolation=_cv2.INTER_NEAREST)
                            fy_img = _cv2.resize(fy_img, (tw, th), interpolation=_cv2.INTER_NEAREST)
                            fz_img = _cv2.resize(fz_img, (tw, th), interpolation=_cv2.INTER_NEAREST)
                        fx_flat = fx_img.reshape(-1)
                        fy_flat = fy_img.reshape(-1)
                        fz_flat = fz_img.reshape(-1)
                        if mask is not None and pc.shape[0] != (th * tw):
                            fx_flat = fx_flat[mask_flat]
                            fy_flat = fy_flat[mask_flat]
                            fz_flat = fz_flat[mask_flat]
                        result['pointcloud_forces'] = np.stack([fx_flat, fy_flat, fz_flat], axis=1)
                    except Exception:
                        # Fail silently and do not provide additional pointcloud fields if error occurs
                        pass
        
        return result
    
    def start_thread(self, outputs: Optional[List[str]] = None, ppmm: Optional[float] = None, **depth_kwargs):
        """Start background processing thread.
        
        Args:
            outputs: List of outputs to compute in thread. Set once at thread start.
            ppmm: Pixels per mm for depth estimation
            **depth_kwargs: Additional arguments for depth estimation
        """
        if self._thread is not None and self._thread.is_alive():
            warnings.warn("Thread already running")
            return
        
        # Default outputs
        if outputs is None:
            outputs = []
            if self.enable_depth:
                outputs.append('depth')
            if self.enable_force:
                outputs.extend(['force_field', 'force_vector'])
        
        # Set ppmm if provided
        if ppmm is not None:
            self.ppmm = ppmm
        
        with self._lock:
            self._outputs = outputs
            self._depth_kwargs = depth_kwargs
            self._running = True
        
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
    
    def stop_thread(self):
        """Stop background processing thread."""
        if self._thread is None:
            return
        
        with self._lock:
            self._running = False
        
        self._thread.join(timeout=2.0)
        # Keep reference to thread object for testing
        # Thread will be marked as not alive after join
    
    def _process_loop(self):
        """Background processing loop (runs in thread)."""
        while True:
            with self._lock:
                if not self._running:
                    break
                frame = self._latest_frame
                timestamp = self._latest_timestamp
                outputs = self._outputs.copy()  # Copy under lock
                depth_kwargs = self._depth_kwargs.copy()
            
            if frame is not None:
                try:
                    result = self.process(
                        image=frame,
                        outputs=outputs,
                        timestamp=timestamp,
                        **depth_kwargs
                    )
                    with self._lock:
                        self._latest_result = result
                except Exception as e:
                    warnings.warn(f"Processing error: {e}")
            
            time.sleep(0.001)  # Small sleep to avoid busy-waiting
    
    def set_input_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """Set input frame for background processing thread.
        
        Args:
            frame: [H, W, 3] BGR image (uint8)
            timestamp: Optional timestamp for force temporal tracking
        
        Note:
            Assumes caller provides fresh frame (not reused buffer).
            Camera.get_image() already returns isolated array, so no copy needed.
        """
        with self._lock:
            self._latest_frame = frame
            self._latest_timestamp = timestamp
    
    def get_latest_result(self) -> Dict[str, Union[np.ndarray, Dict, None]]:
        """Get latest processing result from background thread.
        
        Returns:
            Dictionary with processed outputs (may be empty initially)
        """
        with self._lock:
            return self._latest_result.copy()
    
    def is_background_loaded(self) -> bool:
        """Check if background has been loaded."""
        return self._bg_loaded
    
    def __del__(self):
        """Cleanup thread on deletion."""
        try:
            self.stop_thread()
        except (AttributeError, Exception):
            # Object may not be fully initialized
            pass
