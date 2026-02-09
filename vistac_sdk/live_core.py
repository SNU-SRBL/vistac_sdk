import os
import time
import threading
import numpy as np
import pathlib
from vistac_sdk.vistac_device import Camera
from vistac_sdk.vistac_reconstruct import DepthEstimator
from vistac_sdk.tactile_processor import TactileProcessor
from vistac_sdk.utils import load_config


class LiveTactileProcessor:
    """
    Core class for real-time tactile sensor acquisition and processing.
    
    Combines camera acquisition with depth/force estimation in a unified API.
    Supports selective outputs, background collection, and threaded processing.
    
    Use this in ROS2 nodes, scripts, or other applications.
    """
    def __init__(
        self,
        serial,
        sensors_root: str | os.PathLike | None = None,
        model_device="cuda",
        enable_depth: bool = True,
        enable_force: bool = False,
        temporal_stride: int = 5,
        bg_offset: float = 0.5,
        outputs: list[str] | None = None,
        # Depth-specific parameters
        use_mask=True,
        refine_mask=True,
        relative=True,
        relative_scale=0.5,
        mask_only_pointcloud: bool = False,
        color_dist_threshold=15,
        height_threshold=0.2,
    ):
        """Initialize LiveTactileProcessor.
        
        Args:
            serial: Sensor serial number (e.g., 'D21119')
            sensors_root: Path to sensors directory (auto-detected if None)
            model_device: Device for model inference ('cuda' or 'cpu')
            enable_depth: Enable depth estimation
            enable_force: Enable force estimation
            temporal_stride: Frames between temporal pair for force estimation
            bg_offset: Background subtraction offset for force estimation
            outputs: List of outputs to compute. Options:
                - 'depth': Depth map
                - 'gradient': Gradient field
                - 'pointcloud': Point cloud
                - 'mask': Contact mask
                - 'force_field': Force field heatmap (requires enable_force=True)
                - 'force_vector': Force vector (fx, fy, fz) (requires enable_force=True)
                Default: ['depth'] if depth only, ['force_field', 'force_vector'] if force only
            use_mask: Use contact mask for depth estimation
            refine_mask: Refine contact mask for depth estimation
            relative: Use relative depth estimation
            relative_scale: Scale factor for relative depth
            mask_only_pointcloud: Only return masked points in point cloud
            color_dist_threshold: Color distance threshold for masking
            height_threshold: Height threshold for masking
        """
        self.serial = serial
        self.enable_depth = enable_depth
        self.enable_force = enable_force
        
        # Depth processing parameters
        self._depth_kwargs = {
            'use_mask': use_mask,
            'refine_mask': refine_mask,
            'relative': relative,
            'relative_scale': relative_scale,
            'mask_only_pointcloud': mask_only_pointcloud,
            'color_dist_threshold': color_dist_threshold,
            'height_threshold': height_threshold,
        }
        
        # Determine sensors root
        if sensors_root is None:
            sdk_root = pathlib.Path(__file__).resolve().parents[1]
            sensors_root = sdk_root / "sensors"
        sensors_root = pathlib.Path(sensors_root)
        if not sensors_root.exists():
            raise FileNotFoundError(
                f"sensors_root '{sensors_root}' does not exist")
        
        # Load sensor configuration
        sensor_dir = os.path.join(sensors_root, serial)
        config_path = os.path.join(sensor_dir, f"{serial}.yaml")
        model_path = os.path.join(sensor_dir, "model", "nnmodel.pth")
        config = load_config(config_path=config_path)
        self.device_type = config["device_type"]
        self.ppmm = config["ppmm"]
        
        # Determine model paths for force estimation
        sdk_root = pathlib.Path(__file__).resolve().parents[1]
        force_encoder_path = str(sdk_root / "models" / "sparsh_dino_base_encoder.ckpt")
        force_decoder_path = str(sdk_root / "models" / "sparsh_digit_forcefield_decoder.pth")
        
        # Initialize camera
        self.camera = Camera(serial=serial, sensors_root=sensors_root, thread=True)
        self.camera.connect()
        
        # Initialize unified processor
        self.processor = TactileProcessor(
            model_path=model_path if enable_depth else None,
            enable_depth=enable_depth,
            enable_force=enable_force,
            force_encoder_path=force_encoder_path,
            force_decoder_path=force_decoder_path,
            temporal_stride=temporal_stride,
            bg_offset=bg_offset,
            device=model_device,
            ppmm=self.ppmm,
            contact_mode='standard',
        )
        
        # Collect background (average 10 frames)
        print(f"Collecting background for sensor {serial}...")
        bg_images = []
        for _ in range(10):
            time.sleep(0.2)
            frame = self.camera.get_image()
            while frame is None:
                time.sleep(0.01)
                frame = self.camera.get_image()
            bg_images.append(frame)
        bg_image = np.mean(bg_images, axis=0).astype(np.uint8)
        self.processor.load_background(bg_image)
        print(f"Background collected for sensor {serial}")
        
        # Start background processing thread
        self.processor.start_thread(outputs=outputs, ppmm=self.ppmm, **self._depth_kwargs)

    def get_latest_output(self):
        """Get latest camera frame and processing result.
        
        Returns:
            tuple: (frame, result_dict) where:
                - frame: Latest camera frame [H, W, 3] BGR uint8, or None
                - result_dict: Dictionary with requested outputs (may be empty initially)
                    Keys depend on enabled estimators and requested outputs:
                    - 'depth': [H, W] uint8
                    - 'gradient': [H, W, 2] float32
                    - 'pointcloud': [N, 3] float32
                    - 'mask': [H, W] bool
                    - 'force_field': {'normal': [224, 224], 'shear': [224, 224, 2]} or None
                    - 'force_vector': {'fx': float, 'fy': float, 'fz': float} or None
        """
        # Get latest camera frame
        frame = self.camera.get_image()
        
        # If we have a new frame, send it to processor
        if frame is not None:
            timestamp = time.time()
            self.processor.set_input_frame(frame, timestamp)
        
        # Get latest result from processor
        result = self.processor.get_latest_result()
        
        return frame, result

    def release(self):
        """Release resources and stop threads."""
        self.processor.stop_thread()
        self.camera.release()


class LiveReconstructor:
    """
    DEPRECATED: Legacy wrapper for backward compatibility.
    
    Use LiveTactileProcessor instead for new code.
    
    This class maintains the old API (returns single output, not dict)
    for compatibility with existing code.
    """
    def __init__(
        self,
        serial,
        sensors_root: str | os.PathLike | None = None,
        model_device="cuda",
        mode="depth",
        use_mask=True,
        refine_mask=True,
        relative=True,
        relative_scale=0.5,
        mask_only_pointcloud: bool = False,
        color_dist_threshold=15,
        height_threshold=0.2,
    ):
        """Initialize LiveReconstructor (legacy).
        
        DEPRECATED: Use LiveTactileProcessor instead.
        
        Args:
            serial: Sensor serial number
            sensors_root: Path to sensors directory
            model_device: Device for model inference ('cuda' or 'cpu')
            mode: Output mode ('depth', 'gradient', or 'pointcloud')
            use_mask: Use contact mask
            refine_mask: Refine contact mask
            relative: Use relative depth
            relative_scale: Scale factor for relative depth
            mask_only_pointcloud: Only return masked points
            color_dist_threshold: Color distance threshold
            height_threshold: Height threshold
        """
        import warnings
        warnings.warn(
            "LiveReconstructor is deprecated. Use LiveTactileProcessor instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.mode = mode
        
        # Determine outputs based on mode
        if mode == "depth":
            outputs = ['depth']
        elif mode == "gradient":
            outputs = ['gradient']
        elif mode == "pointcloud":
            outputs = ['pointcloud']
        else:
            outputs = ['depth']
        
        # Create new processor with depth only (force disabled for legacy)
        self._processor = LiveTactileProcessor(
            serial=serial,
            sensors_root=sensors_root,
            model_device=model_device,
            enable_depth=True,
            enable_force=False,  # Force disabled for backward compatibility
            outputs=outputs,
            use_mask=use_mask,
            refine_mask=refine_mask,
            relative=relative,
            relative_scale=relative_scale,
            mask_only_pointcloud=mask_only_pointcloud,
            color_dist_threshold=color_dist_threshold,
            height_threshold=height_threshold,
        )
        
        # Expose for backward compatibility
        self.device = self._processor.camera
        self.estimator = self._processor.processor.depth_estimator
        self.ppmm = self._processor.ppmm
        self.device_type = self._processor.device_type

    def get_latest_output(self):
        """Get latest output (legacy format).
        
        Returns:
            tuple: (frame, result) where:
                - frame: Latest camera frame or None
                - result: Single output array (depth/gradient/pointcloud) or None
        """
        frame, result_dict = self._processor.get_latest_output()
        
        # Extract the specific output for the mode
        result = result_dict.get(self.mode) if result_dict else None
        
        # Warn if result is None (computation failed or not ready)
        if result is None and frame is not None:
            import warnings
            warnings.warn(
                f"Result for mode '{self.mode}' is None. "
                f"This may break legacy code expecting valid output.",
                RuntimeWarning,
                stacklevel=2
            )
        
        return frame, result

    def release(self):
        """Release resources."""
        self._processor.release()