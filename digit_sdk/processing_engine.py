"""
ProcessingEngine: pure-Python tactile sensor processing pipeline.

Reads raw frames from shared memory (written by camera_shm), runs
background collection, and processes depth/force asynchronously via
worker threads.  One worker thread per sensor decouples GPU processing
from the ROS publish loop, so tail GPU latency never blocks the timer.

Typical usage::

    engine = ProcessingEngine(
        serials=["D21275"],
        sensors_root="/path/to/sensors",
        model_device="cuda",
        outputs=["depth"],
    )
    engine.collect_background("D21275")
    engine.start_workers()

    # In timer callback:
    frame = engine.read_frame("D21275")
    if frame is not None:
        engine.submit_frame("D21275", frame)
    result = engine.get_latest_result("D21275")
    if result:
        publish(result)
"""

import logging
import pathlib
import threading
import time
from multiprocessing import shared_memory
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from .tactile_processor import TactileProcessor
from .utils import load_config
from .viz_utils import force_field_to_rgb

logger = logging.getLogger(__name__)

BG_COLLECTION_FRAMES = 10


class ProcessingEngine:
    """Manages SHM reading, background collection, and asynchronous
    depth/force processing via worker threads.  Decouples GPU processing
    from the publish loop."""

    def __init__(
        self,
        serials: List[str],
        sensors_root: str,
        model_device: str = "cuda",
        enable_force: bool = False,
        temporal_stride: int = 5,
        outputs: Optional[List[str]] = None,
        use_mask: bool = True,
        refine_mask: bool = True,
        relative: bool = True,
        relative_scale: float = 0.5,
        mask_only_pointcloud: bool = False,
        point_sample_mm: float = 0.0,
        contact_mode: str = "standard",
        force_field_scale: float = 1.0,
        force_field_baseline: bool = False,
    ):
        self._serials = list(serials)
        self._model_device = model_device
        self._force_field_scale = force_field_scale
        self._force_field_baseline = force_field_baseline
        sensors_root = str(sensors_root)
        root = pathlib.Path(sensors_root)

        # Resolve outputs
        if outputs:
            self._outputs = list(outputs)
        else:
            self._outputs = ["depth"]

        self._depth_kwargs = {
            "use_mask": use_mask,
            "refine_mask": refine_mask,
            "relative": relative,
            "relative_scale": relative_scale,
            "mask_only_pointcloud": mask_only_pointcloud,
            "point_sample_mm": point_sample_mm,
        }

        # Per-sensor state
        self._shms: Dict[str, shared_memory.SharedMemory] = {}
        self._last_seqs: Dict[str, int] = {}
        self._processors: Dict[str, TactileProcessor] = {}
        self._ppmm: Dict[str, float] = {}
        self._streams: Dict[str, torch.cuda.Stream] = {}

        # Worker thread state
        self._running = False
        self._worker_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._latest_results: Dict[str, dict] = {}
        self._queued_frames: Dict[str, Optional[np.ndarray]] = {}
        self._queued_seqs: Dict[str, int] = {}
        self._processed_seqs: Dict[str, int] = {}

        # Connect to each sensor and initialize
        for serial in self._serials:
            self._connect_sensor(serial, root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def serials(self) -> List[str]:
        return list(self._serials)

    @property
    def outputs(self) -> List[str]:
        return list(self._outputs)

    def read_frame(self, serial: str) -> Optional[np.ndarray]:
        """Read latest frame from shared memory for *serial*.
        Returns BGR np.ndarray (h, w, 3) uint8, or None if no new frame."""
        shm = self._shms.get(serial)
        if shm is None:
            return None
        buf = shm.buf
        valid = int(buf[24])
        if not valid:
            return None
        seq = int.from_bytes(buf[0:8], "little")
        if seq == self._last_seqs.get(serial, -1):
            return None
        self._last_seqs[serial] = seq
        h = int.from_bytes(buf[16:20], "little")
        w = int.from_bytes(buf[20:24], "little")
        if h == 0 or w == 0:
            return None
        bgr = np.frombuffer(
            buf[32:32 + h * w * 3], dtype=np.uint8).reshape(h, w, 3)
        return bgr.copy()

    def collect_background(self, serial: str, timeout: float = 15.0,
                           bg_frames: int = BG_COLLECTION_FRAMES) -> bool:
        """Collect background frames for *serial* from SHM and load them
        into the TactileProcessor.  Call *before* start_workers().

        Returns True on success."""
        collected: List[np.ndarray] = []
        deadline = time.monotonic() + timeout
        while len(collected) < bg_frames:
            if time.monotonic() > deadline:
                logger.warning(
                    f"{serial}: Background collection timeout "
                    f"({len(collected)}/{bg_frames})")
                break
            frame = self.read_frame(serial)
            if frame is None:
                time.sleep(0.01)
                continue
            collected.append(frame)
        if not collected:
            logger.error(f"{serial}: Failed to collect background")
            return False
        bg_image = np.mean(collected, axis=0).astype(np.uint8)
        self._processors[serial].load_background(bg_image)
        logger.info(
            f"{serial}: Background collected "
            f"({len(collected)} clean frames)")
        return True

    def start_workers(self) -> None:
        """Start one worker thread per active sensor.
        Workers process frames asynchronously via processor.process().
        GPU work is queued on per-sensor CUDA streams for concurrency."""
        with self._lock:
            if self._running:
                return
            self._running = True
            for serial in self._serials:
                if serial not in self._processors:
                    continue
                self._queued_frames[serial] = None
                self._queued_seqs[serial] = 0
                self._processed_seqs[serial] = 0
                self._latest_results[serial] = {}
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(serial,),
                    daemon=True,
                    name=f"engine-{serial}",
                )
                self._worker_threads[serial] = t
                t.start()
        logger.info(f"Started {len(self._worker_threads)} worker threads")

    def submit_frame(self, serial: str, frame: np.ndarray) -> None:
        """Submit a frame for asynchronous processing.
        Non-blocking — the worker thread picks it up."""
        with self._lock:
            self._queued_frames[serial] = frame
            self._queued_seqs[serial] += 1

    def get_latest_result(self, serial: str) -> Optional[dict]:
        """Return the latest processed result for *serial*, or None.
        Non-blocking."""
        with self._lock:
            result = self._latest_results.get(serial)
        return result if result else None

    def process_frame_sync(self, serial: str, frame: np.ndarray) -> dict:
        """Process a single frame synchronously (bypasses worker threads).
        Returns dict with processed outputs, or empty dict on failure."""
        processor = self._processors.get(serial)
        if processor is None:
            return {}
        stream = self._streams.get(serial)
        with torch.cuda.stream(stream):
            result = processor.process(
                image=frame,
                outputs=self._outputs,
                ppmm=self._ppmm.get(serial, 0.0),
                **self._depth_kwargs,
            )
        return result if result else {}

    def canonicalize_force_field(
        self,
        result: dict,
        frame: Optional[np.ndarray] = None,
        serial: str = "",
    ) -> dict:
        """Apply force_field scaling and recompute pointcloud colors/forces.
        Works in-place and returns the modified result dict."""
        if (result is None or "force_field" not in result
                or result["force_field"] is None):
            return result
        ff = result["force_field"]
        try:
            normal_arr = np.asarray(ff["normal"]).astype(np.float32)
            shear_arr = np.asarray(ff["shear"]).astype(np.float32)
        except (KeyError, TypeError, ValueError):
            return result

        normal_vis = normal_arr.astype(np.float64)
        shear_vis = shear_arr.astype(np.float64)

        if self._force_field_scale != 1.0:
            s = float(self._force_field_scale)
            normal_vis = normal_vis * s
            shear_vis = shear_vis * s

        normal_vis = np.clip(normal_vis, 0.0, 1.0).astype(np.float32)
        shear_vis = np.clip(shear_vis, -1.0, 1.0).astype(np.float32)

        ff["normal"] = normal_vis
        ff["shear"] = shear_vis
        result["force_field"] = ff

        # Recompute pointcloud colors / forces
        if frame is not None:
            try:
                pc = result.get("pointcloud")
                if pc is not None:
                    force_rgb = force_field_to_rgb(
                        normal_vis, shear_vis
                    )
                    th, tw = frame.shape[0], frame.shape[1]
                    fh, fw = force_rgb.shape[:2]
                    if (fh, fw) != (th, tw):
                        force_rgb = cv2.resize(
                            force_rgb, (tw, th),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    colors_flat = (
                        force_rgb.reshape(-1, 3) / 255.0
                    )
                    mask = result.get("mask")
                    if (
                        mask is not None
                        and pc.shape[0] != (th * tw)
                    ):
                        mask_flat = mask.ravel()
                        if mask_flat.shape[0] == th * tw:
                            colors_flat = colors_flat[mask_flat]
                    result["pointcloud_colors"] = colors_flat

                    fx_img = shear_vis[..., 0]
                    fy_img = shear_vis[..., 1]
                    fz_img = normal_vis
                    if (fx_img.shape[0], fx_img.shape[1]) != (th, tw):
                        fx_img = cv2.resize(
                            fx_img, (tw, th),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        fy_img = cv2.resize(
                            fy_img, (tw, th),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        fz_img = cv2.resize(
                            fz_img, (tw, th),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    fx_flat = fx_img.reshape(-1)
                    fy_flat = fy_img.reshape(-1)
                    fz_flat = fz_img.reshape(-1)
                    if (
                        mask is not None
                        and pc.shape[0] != (th * tw)
                    ):
                        mask_flat = mask.ravel()
                        if mask_flat.shape[0] == th * tw:
                            fx_flat = fx_flat[mask_flat]
                            fy_flat = fy_flat[mask_flat]
                            fz_flat = fz_flat[mask_flat]
                    result["pointcloud_forces"] = np.stack(
                        [fx_flat, fy_flat, fz_flat], axis=1
                    )
            except Exception:
                logger.warning(
                    f"{serial}: Failed to recompute "
                    "pointcloud colors/forces",
                    exc_info=True,
                )
        return result

    def shutdown(self) -> None:
        """Stop all worker threads and close SHM handles."""
        with self._lock:
            self._running = False
        for t in self._worker_threads.values():
            t.join(timeout=2.0)
        self._worker_threads.clear()
        for shm in self._shms.values():
            shm.close()
        self._processors.clear()
        self._shms.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _worker_loop(self, serial: str) -> None:
        """Background processing loop for one sensor.
        Waits for new frames, processes via processor.process(),
        stores results for get_latest_result()."""
        processor = self._processors[serial]
        stream = self._streams.get(serial)
        outputs = self._outputs
        ppmm = self._ppmm.get(serial, 0.0)
        depth_kwargs = self._depth_kwargs.copy()

        while True:
            with self._lock:
                if not self._running:
                    break
                frame = self._queued_frames.get(serial)
                qseq = self._queued_seqs.get(serial, 0)
                pseq = self._processed_seqs.get(serial, 0)
            if frame is not None and qseq > pseq:
                try:
                    with torch.cuda.stream(stream):
                        result = processor.process(
                            image=frame,
                            outputs=outputs,
                            ppmm=ppmm,
                            **depth_kwargs,
                        )
                    with self._lock:
                        if result:
                            self._latest_results[serial] = result
                        self._processed_seqs[serial] = qseq
                except Exception:
                    logger.warning(
                        f"{serial}: Worker processing error",
                        exc_info=True,
                    )
            else:
                time.sleep(0.001)

    def _connect_sensor(self, serial: str, root: pathlib.Path) -> None:
        """Connect to SHM, load config, create TactileProcessor."""
        shm_name = f"tactile_{serial}"
        shm = None
        for _ in range(100):
            try:
                shm = shared_memory.SharedMemory(
                    name=shm_name, create=False)
                break
            except FileNotFoundError:
                time.sleep(0.1)
        if shm is None:
            logger.error(f"SHM '{shm_name}' not found after 10s -- skipping")
            return
        self._shms[serial] = shm
        self._last_seqs[serial] = -1

        # Config
        config_path = str(root / serial / f"{serial}.yaml")
        config = load_config(config_path=config_path)
        ppmm = config.get("ppmm", 0.0)
        self._ppmm[serial] = ppmm

        # Model paths
        model_path = str(root / serial / "model" / "nnmodel.pth")
        force_encoder_path = str(
            root.parent / "models" / "sparsh_dino_base_encoder.ckpt")
        force_decoder_path = str(
            root.parent / "models" / "sparsh_digit_forcefield_decoder.pth")

        force_cfg = config.get("force", {}) or {}
        force_vector_scale_cfg = force_cfg.get(
            "force_vector_scale", [1.0, 1.0, 1.0])
        force_enabled_yaml = force_cfg.get("enable_force", False)

        depth_outputs = {"depth", "gradient", "pointcloud", "mask"}
        force_outputs = {"force_field", "force_vector"}
        enable_depth = any(o in depth_outputs for o in self._outputs)
        enable_force_est = force_enabled_yaml or any(o in force_outputs for o in self._outputs)

        self._processors[serial] = TactileProcessor(
            model_path=model_path if enable_depth else None,
            enable_depth=enable_depth,
            enable_force=enable_force_est,
            force_encoder_path=force_encoder_path,
            force_decoder_path=force_decoder_path,
            temporal_stride=5,
            bg_offset=0.5,
            device=self._model_device,
            ppmm=ppmm,
            contact_mode="standard",
            force_field_baseline=self._force_field_baseline,
            force_vector_scale=force_vector_scale_cfg,
        )

        logger.info(f"Sensor {serial} initialized")
        self._streams[serial] = torch.cuda.Stream()

    def __repr__(self) -> str:
        return (
            f"ProcessingEngine(serials={self._serials}, "
            f"outputs={self._outputs})"
        )
