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
The camera can be initialized with raw parameters or by providing a serial number and sensors root directory.
The class handles camera connection, image acquisition, and release of resources.
It also supports threaded image acquisition for low latency applications.  
The camera can be auto-rotated based on the image dimensions.
'''

class Camera:
    """
    The camera class with low latency.

    This class handles camera initialization, image acquisition, and camera release with low latency.
    """

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
        """
        Initialize the low latency camera. Raw camera parameters are required to stream with low latency.

        You can now instantiate with just a serial and sensors_root, or with legacy arguments.

        :param dev_type: str; The type of the camera.
        :param imgh: int; The desired height of the image.
        :param imgw: int; The desired width of the image.
        :param raw_imgh: int; The raw height of the image.
        :param raw_imgw: int; The raw width of the image.
        :param framerate: int; The frame rate of the camera.
        :param serial: str; The serial number of the sensor.
        :param sensors_root: str; Root directory for sensors/<serial>/config.yaml.
        :param config_path: str; Direct path to config yaml.
        :param default_config: str; Fallback config path.
        :param verbose: bool; Whether to print the camera information.
        """
        # If serial or config_path is provided, load config
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

        # Raw image size
        self.raw_imgh = raw_imgh
        self.raw_imgw = raw_imgw
        self.raw_size = self.raw_imgh * self.raw_imgw * 3
        self.framerate = framerate
        # desired image size
        self.imgh = imgh
        self.imgw = imgw
        # Sensor dimensions are auto-detected at connect() time via V4L2
        # device probing (_probe_device_resolution). Config values are used
        # as initial defaults; the probe overrides them with the actual
        # camera resolution to avoid blind dimension swaps.
        # Get camera ID
        # --- DIGIT-specific device identification ---
        if self.device_type and self.device_type.upper() == "DIGIT" and self.serial:
            digit_info = DigitHandler.find_digit(self.serial)
            if digit_info is None:
                raise RuntimeError(f"Could not find DIGIT camera with serial {self.serial}")
            self.device = digit_info["dev_name"]
            self.dev_id = int(re.search(r"\d+$", self.device).group(0))
        # --- Generic camera device identification ---
        else:
            self.dev_id = get_camera_id(self.device_type, verbose)
            self.device = f"/dev/video{self.dev_id}"
        self.last_good_frame = None
        # If thread is True, use a separate thread for image acquisition
        self._threaded = thread
        self._thread = None
        self._running = False
        self._latest_frame = None
        self._latest_frame_available = False
        self._lock = threading.Lock()

    def _find_digit_camera_id_by_serial(self, serial, verbose=True):
        """
        Scan /dev/video* devices and match the serial number for DIGIT sensors.
        Returns the camera ID (int) if found, else None.
        """
        video_root = "/sys/class/video4linux"
        for entry in os.listdir(video_root):
            video_path = os.path.join(video_root, entry)
            serial_path = os.path.join(video_path, "device", "serial")
            if os.path.exists(serial_path):
                try:
                    with open(serial_path, "rt") as f:
                        found_serial = f.read().strip()
                    if found_serial == serial:
                        cam_num = int(re.search(r"\d+$", entry).group(0))
                        if verbose:
                            print(f"Found DIGIT camera: /dev/video{cam_num} with serial {serial}")
                        return cam_num
                except Exception as e:
                    if verbose:
                        print(f"Error reading serial for {entry}: {e}")
        if verbose:
            print(f"No DIGIT camera found with serial {serial}")
        return None

    def _probe_device_resolution(self):
        """
        Query the V4L2 device for its actual frame dimensions.

        Runs v4l2-ctl --get-fmt-video and parses the current
        width and height from the device. This replaces the old
        blind DIGIT dimension swap with real sensor detection.

        Returns
        -------
        tuple
            (width, height) in pixels.

        Raises
        ------
        RuntimeError
            If v4l2-ctl fails, returns unparseable
            output, or the device cannot be queried.

        """
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", self.device, "--get-fmt-video"],
                capture_output=True, text=True, timeout=3.0
            )
        except FileNotFoundError:
            raise RuntimeError(
                "v4l2-ctl not found. Install v4l-utils package: "
                "sudo apt install v4l-utils"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"v4l2-ctl timed out querying {self.device}"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"v4l2-ctl failed on {self.device}: {result.stderr.strip()}"
            )
        match = re.search(
            r"Width/Height\s*:\s*(\d+)/(\d+)", result.stdout
        )
        if not match:
            raise RuntimeError(
                f"Cannot parse v4l2-ctl output from {self.device}: "
                f"{result.stdout.strip()}"
            )
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height

    def connect(self, verbose=True):
        """
        Connect to the camera using FFMpeg streamer.

        Before starting ffmpeg, probes the actual V4L2 sensor
        resolution to ensure correct video_size and raw_size.
        Falls back to config values if probing fails.
        """
        # Probe actual sensor resolution via V4L2 — replaces the old
        # blind DIGIT dimension swap with runtime auto-detection.
        try:
            probe_w, probe_h = self._probe_device_resolution()
            if verbose:
                print(f"Detected camera resolution: {probe_w}x{probe_h}")
            self.raw_imgw = probe_w
            self.raw_imgh = probe_h
            self.raw_size = self.raw_imgh * self.raw_imgw * 3
        except (RuntimeError, FileNotFoundError) as e:
            if verbose:
                print(f"Could not probe camera resolution ({e}). "
                      f"Using config values: {self.raw_imgw}x{self.raw_imgh}.")
        # Command to capture video using ffmpeg and high resolution
        self.ffmpeg_command = (
            ffmpeg.input(
                self.device,
                format="v4l2",
                framerate=self.framerate,
                video_size=(self.raw_imgw, self.raw_imgh),
                pix_fmt="yuyv422",
            )
            .output("pipe:", format="rawvideo", pix_fmt="bgr24")
            .global_args("-fflags", "nobuffer")
            .global_args("-flags", "low_delay")
            .global_args("-fflags", "+genpts")
            .global_args("-rtbufsize", "0")
            .compile()
        )
        self.process = subprocess.Popen(
            self.ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        # Warm-up phase: discard the first few frames with a timeout to
        # prevent indefinite hang if ffmpeg can't access the device.
        if verbose:
            print("Warming up the camera...")
        warm_up_frames = 100
        deadline = time.time() + 15.0  # 15s timeout for camera warm-up
        try:
            for _ in range(warm_up_frames):
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Camera {getattr(self, 'serial', 'unknown')} warm-up "
                        f"timed out after 15s."
                    )
                # Check if ffmpeg exited early
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Camera {getattr(self, 'serial', 'unknown')} ffmpeg "
                        f"exited with code {self.process.returncode} during warm-up."
                    )
                self._read_exact_frame_bytes()
        except (RuntimeError, TimeoutError):
            # Always kill ffmpeg on any failure during warm-up
            try:
                self.process.kill()
                self.process.wait(timeout=2)
            except Exception:
                pass
            raise
        if verbose:
            print("Camera ready for use!")
            
        # If threaded, start the thread for image acquisition
        if self._threaded:
            self._running = True
            self._thread = threading.Thread(target=self._thread_loop, daemon=True)
            self._thread.start()

    def _thread_loop(self):
        """
        Thread loop for image acquisition.
        This runs in a separate thread if _threaded is True.
        """
        while self._running:
            try:
                frame = self.get_image_internal()
                with self._lock:
                    self._latest_frame = frame
                    self._latest_frame_available = True
            except RuntimeError as e:
                if not self._running:
                    break
                # ffmpeg died — attempt reconnect
                print(f"ffmpeg error: {e}. Reconnecting...")
                try:
                    self.process.kill()
                    self.process.wait(timeout=2)
                except Exception:
                    pass
                try:
                    if not self._running:
                        break
                    self.connect(verbose=False)
                    print("ffmpeg reconnected.")
                except Exception as e2:
                    if not self._running:
                        break
                    print(f"Reconnect failed: {e2}. Retrying in 1s...")
                    time.sleep(1.0)
            except Exception as e:
                if not self._running:
                    break
                print(f"Error in thread loop: {e}")
                time.sleep(0.001)

    def get_image_internal(self, flush=False):
        """
        Get the image from the camera from raw data stream.
        If flush=True, discard all but the latest frame.
        """
        if flush:
            import select
            flushed = 0
            while True:
                rlist, _, _ = select.select([self.process.stdout], [], [], 0)
                if rlist:
                    self._read_exact_frame_bytes()
                    flushed += 1
                else:
                    break
            if flushed > 0:
                print(f"Flushed {flushed} stale frames from buffer.")
        try:
            raw_frame = self._read_exact_frame_bytes()
        except RuntimeError:
            # Pipe stalled but process alive — use last good frame once.
            if self.last_good_frame is not None:
                return self.last_good_frame
            raise
        # ffmpeg outputs raw_imgw columns × raw_imgh rows.
        frame = np.frombuffer(raw_frame, np.uint8).reshape(
            (self.raw_imgh, self.raw_imgw, 3)
        )
        self.last_good_frame = frame
        return frame

    def _read_exact_frame_bytes(self):
        """
        Read one raw frame from the ffmpeg pipe.
        """
        raw_frame = self.process.stdout.read(self.raw_size)
        if len(raw_frame) != self.raw_size:
            # Frame-boundary straddle detected: the pipe read returned
            # a partial segment, which means data may cross frame
            # boundaries. Log a warning (rate-limited to the first 3
            # occurrences) so diagnostics are visible without flooding
            # the terminal at 60 fps.
            if not hasattr(self, '_straddle_warn_count'):
                self._straddle_warn_count = 0
            if self._straddle_warn_count < 3:
                import warnings
                warnings.warn(
                    f"Incomplete frame read: got {len(raw_frame)} bytes, "
                    f"expected {self.raw_size}. This may indicate a "
                    f"frame-boundary straddle."
                )
                self._straddle_warn_count += 1
            missing = self.raw_size - len(raw_frame)
            remaining = self.process.stdout.read(missing)
            if len(remaining) < missing:
                # ffmpeg pipe broken — likely process died
                rc = self.process.poll()
                if rc is not None:
                    raise RuntimeError(
                        f"ffmpeg process died with code {rc}. Cannot read frame."
                    )
                # Pipe stalled but process alive — use last good frame once
                if self.last_good_frame is not None:
                    return self.last_good_frame.tobytes()
                raise RuntimeError("ffmpeg pipe stalled and no good frame available.")
            raw_frame += remaining
        return raw_frame
    
    def get_image(self, flush=False):
        """
        Get the image from the camera.

        :param flush: bool; Whether to flush the first few frames.
        :return: np.ndarray; The image from the camera.
        """
        if self._threaded:
            with self._lock:
                if self._latest_frame_available and self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    self._latest_frame_available = False
                    return frame
                return None
        else:
            return self.get_image_internal(flush=flush)

    def release(self):
        """
        Release the camera resource.
        Kill ffmpeg before joining thread to unblock stuck reads.
        """
        self._running = False
        # Kill ffmpeg first — closes the stdout pipe, which unblocks
        # any read() in the thread, allowing it to exit cleanly.
        if hasattr(self, 'process') and self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            except ProcessLookupError:
                pass
        # Now join thread with timeout.
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


def get_camera_id(camera_name, verbose=True):
    """
    Find the camera ID that has the corresponding camera name.

    :param camera_name: str; The name of the camera.
    :param verbose: bool; Whether to print the camera information.
    :return: int; The camera ID.
    """
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
    """
    Resize and crop the image to the desired size.

    :param img: np.ndarray; The image to resize and crop.
    :param imgw: int; The width of the desired image.
    :param imgh: int; The height of the desired image.
    :return: np.ndarray; The resized and cropped image.
    """
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
        np.floor(img.shape[1] * (1 / 7))
    )
    cropped_imgh = img.shape[0] - 2 * border_size_x
    cropped_imgw = img.shape[1] - 2 * border_size_y
    # Extra cropping to maintain aspect ratio
    extra_border_h = 0
    extra_border_w = 0
    if cropped_imgh * imgw / imgh > cropped_imgw + 1e-8:
        extra_border_h = int(cropped_imgh - cropped_imgw * imgh / imgw)
    elif cropped_imgh * imgw / imgh < cropped_imgw - 1e-8:
        extra_border_w = int(cropped_imgw - cropped_imgh * imgw / imgh)
    # keep the ratio the same as the original image size
    img = img[
        border_size_x + extra_border_h : img.shape[0] - border_size_x,
        border_size_y + extra_border_w : img.shape[1] - border_size_y,
    ]
    # final resize for the desired image size
    img = cv2.resize(img, (imgw, imgh))
    return img


if __name__ == "__main__":
    cam = Camera(serial="D21275", sensors_root="sensors")
    cam.connect()
    print("Camera connected.")
    while True:
        # Capture an image
        print("Capturing image...")
        image = cam.get_image()
        cv2.imshow("Captured Image", image)
        print("Press any key to capture another image or 'q' to quit.")
        key = cv2.waitKey(0)
        if key == ord('q'):
            print("Quitting...")
            break
        else:
            print("Captured another image.")
    # Release the camera
    print("Releasing camera resources...")
    cam.release()
    cv2.destroyAllWindows()


class DigitHandler:
    """
    Utility class for DIGIT sensor discovery by serial number using pyudev.
    Copied from https://github.com/facebookresearch/digit-interface/blob/main/digit_interface/digit_handler.py
    """

    @staticmethod
    def _parse(digit_dev):
        digit_info = {
            "dev_name": digit_dev["DEVNAME"],
            "manufacturer": digit_dev.get("ID_VENDOR", ""),
            "model": digit_dev.get("ID_MODEL", ""),
            "revision": digit_dev.get("ID_REVISION", ""),
            "serial": digit_dev.get("ID_SERIAL_SHORT", ""),
        }
        return digit_info

    @staticmethod
    def list_digits():
        if pyudev is None:
            raise ImportError("pyudev is required for DigitHandler. Install with 'pip install pyudev'.")
        context = pyudev.Context()
        digits = context.list_devices(subsystem="video4linux", ID_MODEL="DIGIT")
        digits = [DigitHandler._parse(device) for device in digits]
        return digits

    @staticmethod
    def find_digit(serial):
        digits = DigitHandler.list_digits()
        for digit in digits:
            if digit["serial"] == serial:
                return digit
        return None