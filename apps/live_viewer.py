"""
Standalone DIGIT viewer using Camera + TactileProcessor directly.
Supports depth, gradient, pointcloud, and force visualization modes.

Usage:
    python3 apps/live_viewer.py --serial D21275 --mode depth
    python3 apps/live_viewer.py --serial D21275 --mode pointcloud
    python3 apps/live_viewer.py --serial D21275 --mode force_field --enable_force
"""
import argparse
import os
import sys
import time
import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# Ensure local workspace package is imported when running as a script:
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from vistac_sdk.vistac_device import Camera
from vistac_sdk.tactile_processor import TactileProcessor
from vistac_sdk.utils import load_config
from vistac_sdk.viz_utils import force_field_to_rgb, visualize_force_field, visualize_force_vector

DEFAULT_SENSORS_ROOT = os.path.join(os.path.dirname(__file__), "../sensors")

# Background collection constants
BG_COLLECTION_FRAMES = 10
BG_COLLECTION_DELAY_SEC = 0.2
CAMERA_POLL_INTERVAL_SEC = 0.01


class PointCloudApp:
    def __init__(self, width=640, height=480):
        self.window = gui.Application.instance.create_window("PointCloud", width, height)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)
        self.pcd = o3d.geometry.PointCloud()
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.axis.translate([-0.01, -0.01, 0.0])
        self.scene.scene.add_geometry("axis", self.axis, rendering.MaterialRecord())
        self.scene.scene.set_background([1, 1, 1, 1])
        center = np.array([0.0, 0.0, 0.0])
        eye = center + np.array([0, 0, 0.1])
        up = [0, 1, 0]
        self.scene.setup_camera(
            60,
            o3d.geometry.AxisAlignedBoundingBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            center
        )
        self.scene.scene.camera.look_at(center, eye, up)
        self.first = True

    def update(self, pc, colors):
        self.pcd.points = o3d.utility.Vector3dVector(pc)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        try:
            self.scene.scene.remove_geometry("pcd")
        except Exception:
            pass
        self.scene.scene.add_geometry("pcd", self.pcd, rendering.MaterialRecord())
        if self.first and len(pc) > 0:
            bounds = self.pcd.get_axis_aligned_bounding_box()
            center = bounds.get_center()
            eye = center + np.array([0, 0, 10])
            up = [0, 1, 0]
            self.scene.setup_camera(60, bounds, center)
            self.scene.scene.camera.look_at(center, eye, up)
            self.first = False


def run_live_viewer(
    serial,
    sensors_root=DEFAULT_SENSORS_ROOT,
    use_mask=True,
    mode="depth",
    device_type="cuda",
    refine_mask=False,
    relative=False,
    relative_scale=0.5,
    mask_only_pointcloud=False,
    point_sample_mm=0.0,
    color_dist_threshold=15,
    height_threshold=0.2,
    enable_force=False,
    temporal_stride=5,
    force_field_baseline: bool = False,
    force_field_scale: float = 1.0,
    outputs=None,
):
    # Determine outputs
    if outputs is None:
        mode_map = {
            "depth": ["depth"],
            "gradient": ["gradient"],
            "pointcloud": ["pointcloud"],
            "pointcloud_force": ["pointcloud", "force_field", "mask"],
            "force_field": ["force_field"],
            "force_vector": ["force_vector"],
        }
        outputs = mode_map.get(mode, ["depth"])

    depth_outputs = {"depth", "gradient", "pointcloud", "mask"}
    force_outputs = {"force_field", "force_vector"}
    enable_depth_est = any(x in outputs for x in depth_outputs)
    enable_force_est = enable_force or any(x in outputs for x in force_outputs)

    # --- Camera init ---
    camera = Camera(serial=serial, sensors_root=sensors_root)
    camera.connect(verbose=True)
    print(f"Camera ready for {serial}")

    # Load config for ppmm + model paths
    config = load_config(config_path=os.path.join(sensors_root, serial, f"{serial}.yaml"))
    ppmm = config["ppmm"]
    sdk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    model_path = os.path.join(sdk_root, "models", "nnmodel.pth")
    force_encoder_path = os.path.join(sdk_root, "models", "sparsh_dino_base_encoder.ckpt")
    force_decoder_path = os.path.join(sdk_root, "models", "sparsh_digit_forcefield_decoder.pth")

    force_cfg = config.get("force", {}) or {}
    force_vector_scale_cfg = force_cfg.get("force_vector_scale", [1.0, 1.0, 1.0])

    # --- TactileProcessor init ---
    processor = TactileProcessor(
        model_path=model_path if enable_depth_est else None,
        enable_depth=enable_depth_est,
        enable_force=enable_force_est,
        force_encoder_path=force_encoder_path,
        force_decoder_path=force_decoder_path,
        temporal_stride=temporal_stride,
        bg_offset=0.5,
        device=device_type,
        ppmm=ppmm,
        contact_mode="standard",
        force_field_baseline=force_field_baseline,
        force_vector_scale=force_vector_scale_cfg,
    )

    # --- Background collection ---
    print("Collecting background...")
    bg_images = []
    for _ in range(BG_COLLECTION_FRAMES):
        time.sleep(BG_COLLECTION_DELAY_SEC)
        frame = camera.get_image()
        while frame is None:
            time.sleep(CAMERA_POLL_INTERVAL_SEC)
            frame = camera.get_image()
        bg_images.append(frame)
    bg_image = np.mean(bg_images, axis=0).astype(np.uint8)

    depth_kwargs = {
        "use_mask": use_mask,
        "refine_mask": refine_mask,
        "relative": relative,
        "relative_scale": relative_scale,
        "mask_only_pointcloud": mask_only_pointcloud,
        "point_sample_mm": point_sample_mm,
        "color_dist_threshold": color_dist_threshold,
        "height_threshold": height_threshold,
    }
    processor.load_background(bg_image)
    processor.start_thread(outputs=outputs, ppmm=ppmm, **depth_kwargs)
    print("Background collected. Starting viewer...")

    def get_output():
        """Get latest camera frame and processor result (replaces old get_latest_output)."""
        frame = camera.get_image()
        if frame is None:
            return None, {}
        ts = time.time()
        processor.set_input_frame(frame, ts)
        result = processor.get_latest_result()
        # Canonicalize force_field (same as process_node._canonicalize_force_field)
        if result and "force_field" in result and result["force_field"] is not None:
            ff = result["force_field"]
            try:
                na = np.asarray(ff["normal"]).astype(np.float32)
                sa = np.asarray(ff["shear"]).astype(np.float32)
            except Exception:
                pass
            else:
                nv = np.clip(na, 0.0, 1.0)
                sv = np.clip(sa, -1.0, 1.0)
                if force_field_scale != 1.0:
                    sc = float(force_field_scale)
                    nv = (nv.astype(np.float64) * sc).astype(np.float32)
                    sv = (sv.astype(np.float64) * sc).astype(np.float32)
                ff["normal"] = nv
                ff["shear"] = sv
                result["force_field"] = ff
                # Recompute pointcloud colors/forces
                try:
                    pc = result.get("pointcloud")
                    if pc is not None:
                        frgb = force_field_to_rgb(nv, sv)
                        th, tw = frame.shape[0], frame.shape[1]
                        fh, fw = frgb.shape[:2]
                        if (fh, fw) != (th, tw):
                            frgb = cv2.resize(frgb, (tw, th), interpolation=cv2.INTER_NEAREST)
                        cf = frgb.reshape(-1, 3) / 255.0
                        mask = result.get("mask")
                        if mask is not None and pc.shape[0] != (th * tw):
                            mf = mask.ravel()
                            if mf.shape[0] == th * tw:
                                cf = cf[mf]
                        result["pointcloud_colors"] = cf
                except Exception:
                    pass
                # Inject raw
                result["raw"] = frame.copy()
        return frame, result

    device_name = device_type

    # Multi-panel mode: display multiple outputs side by side
    if len(outputs) > 1 and "pointcloud" not in outputs:
        print("\nMulti-panel mode. Press any key to quit.\n")
        while True:
            frame, result_dict = get_output()
            if frame is None or not result_dict:
                continue

            vis_panels = []

            # Always show raw frame first
            if frame.ndim == 2:
                raw_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                raw_vis = frame
            vis_panels.append(raw_vis)

            for output_name in outputs:
                output_data = result_dict.get(output_name)

                if output_name == "depth" and output_data is not None:
                    dv = cv2.resize(output_data, (frame.shape[1], frame.shape[0]))
                    dv = cv2.cvtColor(dv, cv2.COLOR_GRAY2BGR)
                    vis_panels.append(dv)

                elif output_name == "gradient" and output_data is not None:
                    G = output_data
                    red = G[:, :, 0] * 255 / 3.0 + 127
                    red = np.clip(red, 0, 255)
                    blue = G[:, :, 1] * 255 / 3.0 + 127
                    blue = np.clip(blue, 0, 255)
                    gimg = np.stack((blue, np.zeros_like(blue), red), axis=-1).astype(np.uint8)
                    gv = cv2.resize(gimg, (frame.shape[1], frame.shape[0]))
                    vis_panels.append(gv)

                elif output_name == "force_field":
                    if output_data is None:
                        wp = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                        cv2.putText(wp, "Buffering...", (10, frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        vis_panels.append(wp)
                    else:
                        nf = output_data["normal"].astype(np.float32)
                        sf = output_data["shear"].astype(np.float32)
                        fv = visualize_force_field(nf, sf)
                        if fv.shape[:2] != (frame.shape[0], frame.shape[1]):
                            fv = cv2.resize(fv, (frame.shape[1], frame.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                        vis_panels.append(fv)

                elif output_name == "force_vector":
                    if output_data is None:
                        wp = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                        cv2.putText(wp, "Buffering...", (10, frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        vis_panels.append(wp)
                    else:
                        phys = result_dict.get("force_vector_physical")
                        if phys is not None:
                            fx, fy, fz = phys["fx"], phys["fy"], phys["fz"]
                        else:
                            fx, fy, fz = output_data["fx"], output_data["fy"], output_data["fz"]
                        fv = visualize_force_vector(fx, fy, fz, frame)
                        vis_panels.append(fv)

            if vis_panels:
                combined = np.hstack(vis_panels)
                cv2.imshow(device_name, combined)

            key = cv2.waitKey(1)
            if key != -1:
                break

        processor.stop_thread()
        camera.release()
        cv2.destroyAllWindows()
        return

    # Single output modes
    if mode == "depth":
        print("\nPress any key to quit.\n")
        while True:
            frame, result_dict = get_output()
            if frame is None or not result_dict:
                continue
            depth_image = result_dict.get("depth")
            if depth_image is None:
                continue
            dv = cv2.resize(depth_image, (frame.shape[1], frame.shape[0]))
            dv = cv2.cvtColor(dv, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 2:
                rv = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                rv = frame
            cv2.imshow(device_name, np.hstack([rv, dv]))
            key = cv2.waitKey(1)
            if key != -1:
                break
        processor.stop_thread()
        camera.release()
        cv2.destroyAllWindows()

    elif mode == "gradient":
        print("\nPress any key to quit.\n")
        while True:
            frame, result_dict = get_output()
            if frame is None or not result_dict:
                continue
            G = result_dict.get("gradient")
            if G is None:
                continue
            red = G[:, :, 0] * 255 / 3.0 + 127
            red = np.clip(red, 0, 255)
            blue = G[:, :, 1] * 255 / 3.0 + 127
            blue = np.clip(blue, 0, 255)
            gimg = np.stack((blue, np.zeros_like(blue), red), axis=-1).astype(np.uint8)
            gv = cv2.resize(gimg, (frame.shape[1], frame.shape[0]))
            if frame.ndim == 2:
                rv = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                rv = frame
            cv2.imshow(device_name, np.hstack([rv, gv]))
            key = cv2.waitKey(1)
            if key != -1:
                break
        processor.stop_thread()
        camera.release()
        cv2.destroyAllWindows()

    elif mode == "pointcloud":
        gui.Application.instance.initialize()
        app = PointCloudApp()
        running = True

        def on_close():
            nonlocal running
            running = False
        app.window.set_on_close(on_close)

        def update_loop():
            if not running:
                return
            frame, result_dict = get_output()
            if not result_dict:
                gui.Application.instance.post_to_main_thread(app.window, update_loop)
                return
            pc = result_dict.get("pointcloud")
            if pc is not None and frame is not None:
                pc = pc * 1000.0  # to mm
                colors = frame.reshape(-1, 3)[:, ::-1] / 255.0
                app.update(pc, colors)
            gui.Application.instance.post_to_main_thread(app.window, update_loop)

        gui.Application.instance.post_to_main_thread(app.window, update_loop)
        print("\nClose the window to quit.\n")
        gui.Application.instance.run()
        processor.stop_thread()
        camera.release()
        gui.Application.instance.quit()

    elif mode == "pointcloud_force":
        gui.Application.instance.initialize()
        app = PointCloudApp()
        running = True

        def on_close():
            nonlocal running
            running = False
        app.window.set_on_close(on_close)

        def update_loop_force():
            if not running:
                return
            frame, result_dict = get_output()
            if not result_dict:
                gui.Application.instance.post_to_main_thread(app.window, update_loop_force)
                return
            pc = result_dict.get("pointcloud")
            force_data = result_dict.get("force_field")
            colors = result_dict.get("pointcloud_colors")
            if pc is not None and frame is not None:
                if colors is None and force_data is not None:
                    normal = force_data["normal"]
                    shear = force_data["shear"]
                    normal_n = np.clip(normal, 0.0, 1.0)
                    sx_n = np.clip((shear[..., 0] + 1.0) / 2.0, 0.0, 1.0)
                    sy_n = np.clip((shear[..., 1] + 1.0) / 2.0, 0.0, 1.0)
                    frgb = np.stack([sx_n * 255.0, sy_n * 255.0, normal_n * 255.0], axis=-1).astype(np.uint8)
                    fh, fw = frgb.shape[:2]
                    th, tw = frame.shape[0], frame.shape[1]
                    if (fh, fw) != (th, tw):
                        frgb = cv2.resize(frgb, (tw, th), interpolation=cv2.INTER_NEAREST)
                    colors = frgb.reshape(-1, 3) / 255.0
                    mask = result_dict.get("mask")
                    if mask is not None and pc.shape[0] != (th * tw):
                        mf = mask.ravel()
                        if mf.shape[0] == th * tw:
                            colors = colors[mf]
                if colors is None:
                    colors = frame.reshape(-1, 3)[:, ::-1] / 255.0
                pc = pc * 1000.0
                app.update(pc, colors)
            gui.Application.instance.post_to_main_thread(app.window, update_loop_force)

        gui.Application.instance.post_to_main_thread(app.window, update_loop_force)
        print("\nClose the window to quit.\n")
        gui.Application.instance.run()
        processor.stop_thread()
        camera.release()
        gui.Application.instance.quit()

    elif mode == "force_field":
        print("\nForce field visualization. Press any key to quit.\n")
        while True:
            frame, result_dict = get_output()
            if frame is None or not result_dict:
                continue
            force_data = result_dict.get("force_field")
            if force_data is None:
                if frame.ndim == 2:
                    rv = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    rv = frame
                wp = np.zeros_like(rv)
                cv2.putText(wp, "Buffering...", (10, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow(device_name, np.hstack([rv, wp]))
            else:
                nf = force_data["normal"].astype(np.float32)
                sf = force_data["shear"].astype(np.float32)
                fv = visualize_force_field(nf, sf)
                if fv.shape[:2] != (frame.shape[0], frame.shape[1]):
                    fv = cv2.resize(fv, (frame.shape[1], frame.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                combined = np.hstack([frame, fv]) if frame is not None else fv
                cv2.imshow(device_name, combined)
            key = cv2.waitKey(1)
            if key != -1:
                break
        processor.stop_thread()
        camera.release()
        cv2.destroyAllWindows()

    elif mode == "force_vector":
        print("\nForce vector visualization. Press any key to quit.\n")
        while True:
            frame, result_dict = get_output()
            if frame is None or not result_dict:
                continue
            force_data = result_dict.get("force_vector")
            if force_data is None:
                if frame.ndim == 2:
                    rv = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    rv = frame
                wp = np.zeros_like(rv)
                cv2.putText(wp, "Buffering...", (10, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow(device_name, np.hstack([rv, wp]))
            else:
                phys = result_dict.get("force_vector_physical")
                if phys is not None:
                    fx, fy, fz = phys["fx"], phys["fy"], phys["fz"]
                else:
                    fx, fy, fz = force_data["fx"], force_data["fy"], force_data["fz"]
                fv = visualize_force_vector(fx, fy, fz, frame,
                                            arrow_scale=50.0, arrow_color=(0, 255, 0), arrow_thickness=3)
                if frame.ndim == 2:
                    rv = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    rv = frame
                cv2.imshow(device_name, np.hstack([rv, fv]))
            key = cv2.waitKey(1)
            if key != -1:
                break
        processor.stop_thread()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test model: 2D or pointcloud visualization.")
    parser.add_argument("--serial", type=str, required=True,
                        help="sensor serial number")
    parser.add_argument("--sensors_root", type=str,
                        default=DEFAULT_SENSORS_ROOT)
    parser.add_argument("--use_mask", action="store_true", default=False)
    parser.add_argument("--mode", type=str,
                        choices=["depth", "gradient", "pointcloud",
                                 "pointcloud_force", "force_field", "force_vector"],
                        default="depth")
    parser.add_argument("--refine_mask", action="store_true", default=False)
    parser.add_argument("--relative", action="store_true", default=False)
    parser.add_argument("--relative_scale", type=float, default=0.5)
    parser.add_argument("--mask_only_pointcloud",
                        action="store_true", default=False)
    parser.add_argument("--point_sample_mm", type=float, default=0.0)
    parser.add_argument("--device_type", type=str,
                        choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--color_dist_threshold", type=float, default=15)
    parser.add_argument("--height_threshold", type=float, default=0.2)
    parser.add_argument("--enable_force", action="store_true", default=False)
    parser.add_argument("--temporal_stride", type=int, default=5)
    parser.add_argument("--force_field_baseline",
                        action="store_true", default=False)
    parser.add_argument("--force_field_scale", type=float, default=1.0)
    parser.add_argument("--outputs", type=str, nargs="+", default=None)
    args = parser.parse_args()

    run_live_viewer(
        serial=args.serial,
        sensors_root=args.sensors_root,
        use_mask=args.use_mask,
        mode=args.mode,
        device_type=args.device_type,
        refine_mask=args.refine_mask,
        relative=args.relative,
        relative_scale=args.relative_scale,
        mask_only_pointcloud=args.mask_only_pointcloud,
        point_sample_mm=args.point_sample_mm,
        color_dist_threshold=args.color_dist_threshold,
        height_threshold=args.height_threshold,
        enable_force=args.enable_force,
        temporal_stride=args.temporal_stride,
        force_field_baseline=args.force_field_baseline,
        force_field_scale=args.force_field_scale,
        outputs=args.outputs,
    )
