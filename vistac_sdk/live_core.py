import os
import time
import threading
import numpy as np
import pathlib
from vistac_sdk.vistac_device import Camera
from vistac_sdk.vistac_reconstruct import DepthEstimator
from vistac_sdk.utils import load_config

class LiveReconstructor:
    """
    Core class for real-time acquisition and model inference.
    Use this in ROS2 nodes, scripts, or other applications.
    
    NOTE: This class will be replaced by LiveTactileProcessor in Step 6.
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
        self._mask_only = mask_only_pointcloud
        self.mode = mode
        self.use_mask = use_mask
        self.refine_mask = refine_mask
        self.relative = relative
        self.relative_scale = relative_scale
        self.color_dist_threshold = color_dist_threshold
        self.height_threshold = height_threshold

        if sensors_root is None:
            sdk_root = pathlib.Path(__file__).resolve().parents[1]
            sensors_root = sdk_root / "sensors"
        sensors_root = pathlib.Path(sensors_root)
        if not sensors_root.exists():
            raise FileNotFoundError(
                f"sensors_root '{sensors_root}' does not exist")
        
        sensor_dir = os.path.join(sensors_root, serial)
        config_path = os.path.join(sensor_dir, f"{serial}.yaml")
        model_path = os.path.join(sensor_dir, "model", "nnmodel.pth")
        config = load_config(config_path=config_path)
        self.device_type = config["device_type"]
        self.ppmm = config["ppmm"]
        
        # Initialize camera
        self.device = Camera(serial=serial, sensors_root=sensors_root, thread=True)
        self.device.connect()
        
        # Initialize depth estimator (no threading - we handle it here)
        self.estimator = DepthEstimator(model_path, device=model_device)
        
        # Collect background
        bg_images = []
        for _ in range(10):
            time.sleep(0.2)
            frame = self.device.get_image()
            while frame is None:
                time.sleep(0.01)
                frame = self.device.get_image()
            bg_images.append(frame)
        bg_image = np.mean(bg_images, axis=0).astype(np.uint8)
        self.estimator.load_bg(bg_image)
        
        # Threading for background processing
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_result = None
        self._running = False
        self._thread = None
        
        # Start processing thread
        self._start_thread()

    def _start_thread(self):
        """Start background processing thread."""
        with self._lock:
            self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def _process_loop(self):
        """Background thread that continuously processes frames."""
        while True:
            with self._lock:
                if not self._running:
                    break
                frame = self._latest_frame
            
            if frame is not None:
                # Determine outputs based on mode
                if self.mode == "depth":
                    outputs = ['depth']
                elif self.mode == "gradient":
                    outputs = ['gradient']
                elif self.mode == "pointcloud":
                    outputs = ['pointcloud']
                else:
                    outputs = ['depth']  # Default
                
                try:
                    result_dict = self.estimator.estimate(
                        frame,
                        outputs=outputs,
                        ppmm=self.ppmm,
                        use_mask=self.use_mask,
                        refine_mask=self.refine_mask,
                        relative=self.relative,
                        relative_scale=self.relative_scale,
                        mask_only_pointcloud=self._mask_only,
                        color_dist_threshold=self.color_dist_threshold,
                        height_threshold=self.height_threshold,
                    )
                    # Extract the specific output for the mode
                    result = result_dict.get(self.mode)
                    
                    with self._lock:
                        self._latest_result = result
                except Exception as e:
                    print(f"Error in processing loop: {e}")
                    with self._lock:
                        self._latest_result = None
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning

    def get_latest_output(self):
        """
        Returns (frame, result) where:
        - frame: latest camera frame (numpy array)
        - result: model output (depth/gradient/pointcloud), depending on mode
        """
        frame = self.device.get_image()
        if frame is not None:
            with self._lock:
                self._latest_frame = frame
        
        with self._lock:
            result = self._latest_result
        
        return frame, result

    def release(self):
        """Release resources and stop threads."""
        with self._lock:
            self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        
        self.device.release()