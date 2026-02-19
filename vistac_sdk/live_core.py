import os
import time
import threading
import numpy as np
import cv2
import pathlib
from vistac_sdk.vistac_device import Camera
from vistac_sdk.vistac_reconstruct import DepthEstimator
from vistac_sdk.tactile_processor import TactileProcessor
from vistac_sdk.utils import load_config
from vistac_sdk.viz_utils import force_field_to_rgb

# Background collection constants
BG_COLLECTION_FRAMES = 10  # Number of frames to average for background
BG_COLLECTION_DELAY_SEC = 0.2  # Delay between frames to allow camera auto-exposure
CAMERA_POLL_INTERVAL_SEC = 0.01  # Polling interval when waiting for valid frame


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
        # Runtime baseline override for force_field (None = use YAML)
        force_field_baseline: bool | None = None,
        # Force-field visual scaling (viewer/ROS only; NOT a physical-unit conversion)
        force_field_scale: float = 1.0,
        # Depth-specific parameters
        use_mask=True,
        refine_mask=True,
        relative=True,
        relative_scale=0.5,
        mask_only_pointcloud: bool = False,
        color_dist_threshold=15,
        height_threshold=0.2,
    ):
        # Store user-configurable force_field scale (applied to `force_field` returned by get_latest_output)
        self.force_field_scale = float(force_field_scale)
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

        def resolve_force_model(filename: str) -> str:
            primary = sdk_root / "models" / filename
            if primary.exists():
                return str(primary)
            cwd_candidate = pathlib.Path.cwd() / "models" / filename
            if cwd_candidate.exists():
                return str(cwd_candidate)
            return str(primary)

        force_encoder_path = resolve_force_model("sparsh_dino_base_encoder.ckpt")
        force_decoder_path = resolve_force_model("sparsh_digit_forcefield_decoder.pth")
        
        # Initialize camera
        self.camera = Camera(serial=serial, sensors_root=sensors_root, thread=True)
        self.camera.connect()
        
        # Read optional force config values
        force_cfg = config.get('force', {}) or {}
        # force_field_baseline is runtime-only now; ignore YAML. Use runtime arg.
        force_vector_scale_cfg = force_cfg.get('force_vector_scale', [1.0, 1.0, 1.0])

        # Runtime-only flag: prefer explicit runtime override (defaults to False)
        force_field_baseline_flag = bool(force_field_baseline)
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
            force_field_baseline=force_field_baseline_flag,
            force_vector_scale=force_vector_scale_cfg,
        )
        
        # Collect fresh background at runtime (average multiple frames)
        # This adapts to current lighting conditions, sensor aging, etc.
        # Note: Saved background.png files are only used for offline calibration/training
        print(f"Collecting background for sensor {serial}...")
        bg_images = []
        for _ in range(BG_COLLECTION_FRAMES):
            time.sleep(BG_COLLECTION_DELAY_SEC)
            frame = self.camera.get_image()
            while frame is None:
                time.sleep(CAMERA_POLL_INTERVAL_SEC)
                frame = self.camera.get_image()
            bg_images.append(frame)
        bg_image = np.mean(bg_images, axis=0).astype(np.uint8)
        
        # Load same background into both depth and force estimators
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

        # Canonicalize Sparsh outputs for presentation (centralized here):
        # - Sparsh semantics: normal in [0,1]; shear is model-scaled (tanh * scale_flow).
        # - For UI/ROS we clip visual shear to [-1, 1] and keep normal in [0, 1], then
        #   apply `force_field_scale` (visual multiplier).
        if result and 'force_field' in result and result['force_field'] is not None:
            ff = result['force_field']

            # Defensive: ensure arrays are numeric
            try:
                normal_arr = np.asarray(ff['normal']).astype(np.float32)
                shear_arr = np.asarray(ff['shear']).astype(np.float32)
            except Exception:
                import warnings
                warnings.warn("Skipping force_field processing: force_field arrays are not numeric")
                normal_arr = None
                shear_arr = None

            if normal_arr is not None and shear_arr is not None:
                # Enforce Sparsh visual contract for presentation:
                # - normal already in [0,1] (sigmoid); clamp to be safe
                # - shear: clip for visualization to [-1, 1]
                normal_vis = np.clip(normal_arr, 0.0, 1.0)
                shear_vis = np.clip(shear_arr, -1.0, 1.0)

                # Apply runtime visual scaling (if requested)
                if hasattr(self, 'force_field_scale') and float(self.force_field_scale) != 1.0:
                    scale = float(self.force_field_scale)
                    normal_scaled = (normal_vis.astype(np.float64) * scale).astype(np.float32)
                    shear_scaled = (shear_vis.astype(np.float64) * scale).astype(np.float32)
                else:
                    normal_scaled = normal_vis
                    shear_scaled = shear_vis

                # Replace values returned to callers (viewer, ROS)
                ff['normal'] = normal_scaled
                ff['shear'] = shear_scaled
                result['force_field'] = ff

                # Recompute pointcloud colors/forces from the (possibly scaled) force_field
                try:
                    pc = result.get('pointcloud')
                    if pc is not None:
                        force_rgb = force_field_to_rgb(normal_scaled, shear_scaled)

                        th, tw = frame.shape[0], frame.shape[1]
                        fh, fw = force_rgb.shape[:2]
                        if (fh, fw) != (th, tw):
                            force_rgb = cv2.resize(force_rgb, (tw, th), interpolation=cv2.INTER_NEAREST)
                        colors_flat = force_rgb.reshape(-1, 3) / 255.0

                        mask = result.get('mask')
                        if mask is not None and pc.shape[0] != (th * tw):
                            mask_flat = mask.ravel()
                            if mask_flat.shape[0] == th * tw:
                                colors_flat = colors_flat[mask_flat]

                        result['pointcloud_colors'] = colors_flat

                    if pc is not None:
                        # Recompute per-point raw forces fx,fy,fz from scaled fields
                        fx_img = shear_scaled[..., 0]
                        fy_img = shear_scaled[..., 1]
                        fz_img = normal_scaled
                        if (fx_img.shape[0], fx_img.shape[1]) != (th, tw):
                            fx_img = cv2.resize(fx_img, (tw, th), interpolation=cv2.INTER_NEAREST)
                            fy_img = cv2.resize(fy_img, (tw, th), interpolation=cv2.INTER_NEAREST)
                            fz_img = cv2.resize(fz_img, (tw, th), interpolation=cv2.INTER_NEAREST)
                        fx_flat = fx_img.reshape(-1)
                        fy_flat = fy_img.reshape(-1)
                        fz_flat = fz_img.reshape(-1)
                        if mask is not None and pc.shape[0] != (th * tw):
                            fx_flat = fx_flat[mask_flat]
                            fy_flat = fy_flat[mask_flat]
                            fz_flat = fz_flat[mask_flat]
                        result['pointcloud_forces'] = np.stack([fx_flat, fy_flat, fz_flat], axis=1)
                except Exception:
                    import warnings
                    warnings.warn('Failed to recompute pointcloud colors/forces after scaling')


        return frame, result

    def release(self):
        """Release resources and stop threads."""
        self.processor.stop_thread()
        self.camera.release()