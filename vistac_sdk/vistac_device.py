import os
import re
import subprocess

import cv2
import ffmpeg
import numpy as np
import threading
import time

import pyudev

from vistac_sdk.utils import load_config

'''This module provides a Camera class for low latency image acquisition using FFMpeg.
It supports both DIGIT cameras and generic V4L2 cameras.
'''

class Camera:
    """The camera class with low latency (based on gs_sdk FastCamera)."""

    def __init__(
        self,
        dev_type=None,
        imgh=None,
        imgw=None,
        raw_imgh=None,
        raw_imgw=None,
        framerate=None,
        serial=None,
        sensors_root=None,
        config_path=None,
        default_config=None,
        verbose=True,
        thread=False,
    ):
        if serial is not None or config_path is not None:
            config = load_config(
                serial=serial,
                config_path=config_path,
                sensors_root=sensors_root,
                default_config=default_config,
            )
            dev_type = config["device_type"]
            imgh = config["imgh"]
            imgw = config["imgw"]
            raw_imgh = config["raw_imgh"]
            raw_imgw = config["raw_imgw"]
            framerate = config["framerate"]
            self.serial = serial
            self.device_type = dev_type
        else:
            self.serial = None
            self.device_type = dev_type

        self.raw_imgh = raw_imgh
        self.raw_imgw = raw_imgw
        self.raw_size = self.raw_imgh * self.raw_imgw * 3
        self.framerate = framerate
        self.imgh = imgh
        self.imgw = imgw

        # Configs store raw dimensions inverted (raw_imgh = width, raw_imgw = height).
        # Normalize so raw_imgh = height, raw_imgw = width for numpy reshape.
        if imgh is not None and imgw is not None and imgh > imgw:
            self.raw_imgh, self.raw_imgw = raw_imgw, raw_imgh
            self.raw_size = self.raw_imgh * self.raw_imgw * 3

        # Camera device discovery
        if self.device_type and self.device_type.upper() == "DIGIT" and self.serial:
            digit_info = DigitHandler.find_digit(self.serial)
            if digit_info is None:
                raise RuntimeError(f"Could not find DIGIT camera with serial {self.serial}")
            self.device = digit_info["dev_name"]
            self.dev_id = int(re.search(r"\d+$", self.device).group(0))
        else:
            self.dev_id = get_camera_id(self.device_type, verbose)
            self.device = f"/dev/video{self.dev_id}"

        self.last_good_frame = None
        self._threaded = thread
        self._thread = None
        self._running = False
        self._latest_frame = None
        self._lock = threading.Lock()

    def connect(self, verbose=True):
        """Connect to the camera using OpenCV VideoCapture (digit-interface pattern)."""
        self.cap = cv2.VideoCapture(self.dev_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.raw_imgw)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.raw_imgh)
        self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if verbose:
            print("Warming up the camera...")
        warm_up_frames = 10
        for _ in range(warm_up_frames):
            self.cap.read()
        if verbose:
            print("Camera ready for use!")

        if self._threaded:
            self._running = True
            self._thread = threading.Thread(target=self._thread_loop, daemon=True)
            self._thread.start()

    def _thread_loop(self):
        """Background frame capture thread."""
        while self._running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame = frame.copy()
                    # Validate: check for row discontinuity (USB corruption)
                    if self.last_good_frame is not None:
                        row_d = np.sum(np.abs(
                            frame[:-1].astype(np.int16) - frame[1:].astype(np.int16)
                        ), axis=(1,2))
                        med = float(np.median(row_d))
                        mx = float(np.max(row_d))
                        if med > 0 and mx / med > 3.0:
                            # Corrupted frame - keep last good one
                            with self._lock:
                                self._latest_frame = self.last_good_frame.copy()
                            continue
                    self.last_good_frame = frame
                    with self._lock:
                        self._latest_frame = frame
            except Exception as e:
                print(f"[CAM] Error: {e}", flush=True)
            time.sleep(0.001)

    def get_image(self, flush=False):
        """Get the latest image from the camera."""
        if self._threaded:
            with self._lock:
                has = self._latest_frame is not None
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
                return None
        else:
            ret, frame = self.cap.read()
            return frame if ret else None

    def release(self):
        """Release camera resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


def get_camera_id(camera_name, verbose=True):
    """Find the camera ID by name."""
    cam_num = None
    for file in os.listdir("/sys/class/video4linux"):
        real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
        with open(real_file, "rt") as name_file:
            name = name_file.read().rstrip()
        if camera_name in name:
            cam_num = int(re.search("\d+$", file).group(0))
            if verbose:
                found = "FOUND!"
        else:
            if verbose:
                found = "      "
        if verbose:
            print("{} {} -> {}".format(found, file, name))
    return cam_num


def resize_crop(img, imgw, imgh):
    """Resize and crop the image."""
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
        np.floor(img.shape[1] * (1 / 7))
    )
    cropped_imgh = img.shape[0] - 2 * border_size_x
    cropped_imgw = img.shape[1] - 2 * border_size_y
    extra_border_h = 0
    extra_border_w = 0
    if cropped_imgh * imgw / imgh > cropped_imgw + 1e-8:
        extra_border_h = int(cropped_imgh - cropped_imgw * imgh / imgw)
    elif cropped_imgh * imgw / imgh < cropped_imgw - 1e-8:
        extra_border_w = int(cropped_imgw - cropped_imgh * imgw / imgh)
    img = img[
        border_size_x + extra_border_h : img.shape[0] - border_size_x,
        border_size_y + extra_border_w : img.shape[1] - border_size_y,
    ]
    img = cv2.resize(img, (imgw, imgh))
    return img


if __name__ == "__main__":
    cam = Camera(serial="D21275", sensors_root="sensors")
    cam.connect()
    print("Camera connected.")
    while True:
        image = cam.get_image()
        cv2.imshow("Captured Image", image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


class DigitHandler:
    """Utility class for DIGIT sensor discovery."""

    @staticmethod
    def _parse(digit_dev):
        return {
            "dev_name": digit_dev["DEVNAME"],
            "manufacturer": digit_dev.get("ID_VENDOR", ""),
            "model": digit_dev.get("ID_MODEL", ""),
            "revision": digit_dev.get("ID_REVISION", ""),
            "serial": digit_dev.get("ID_SERIAL_SHORT", ""),
        }

    @staticmethod
    def list_digits():
        if pyudev is None:
            raise ImportError("pyudev is required for DigitHandler. Install with 'pip install pyudev'.")
        context = pyudev.Context()
        digits = context.list_devices(subsystem="video4linux", ID_MODEL="DIGIT")
        return [DigitHandler._parse(device) for device in digits]

    @staticmethod
    def find_digit(serial):
        digits = DigitHandler.list_digits()
        for digit in digits:
            if digit["serial"] == serial:
                return digit
        return None
