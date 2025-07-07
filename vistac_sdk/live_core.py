import os
import time
import numpy as np
import pathlib
from vistac_sdk.vistac_device import Camera
from vistac_sdk.vistac_reconstruct import Reconstructor
from vistac_sdk.utils import load_config

class LiveReconstructor:
    """
    Core class for real-time acquisition and model inference.
    Use this in ROS2 nodes, scripts, or other applications.
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
        relative_scale=0.5
    ):
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
        self.device = Camera(serial=serial, sensors_root=sensors_root, thread=True)
        self.device.connect()
        self.recon = Reconstructor(model_path, device=model_device, thread=True)
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
        self.recon.load_bg(bg_image)
        self.recon.start_thread(
            ppmm=self.ppmm,
            mode=mode,
            use_mask=use_mask,
            refine_mask=refine_mask,
            relative=relative,
            relative_scale=relative_scale
        )
        self.mode = mode

    def get_latest_output(self):
        """
        Returns (frame, result) where:
        - frame: latest camera frame (numpy array)
        - result: model output (depth/gradient/pointcloud), depending on mode
        """
        frame = self.device.get_image()
        if frame is not None:
            self.recon.set_input_frame(frame)
        result = self.recon.get_latest_result()
        return frame, result

    def release(self):
        self.device.release()
        self.recon.stop_thread()