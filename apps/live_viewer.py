import argparse
import os
import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from apps.live_core import LiveReconstructor

DEFAULT_SENSORS_ROOT = os.path.join(os.path.dirname(__file__), "../sensors")

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
        except:
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
    relative_scale=0.5
):
    recon = LiveReconstructor(
        serial=serial,
        sensors_root=sensors_root,
        model_device=device_type,
        mode=mode,
        use_mask=use_mask,
        refine_mask=refine_mask,
        relative=relative,
        relative_scale=relative_scale
    )
    device_type = recon.device_type
    ppmm = recon.ppmm

    if mode == "depth":
        print("\nPress any key to quit.\n")
        while True:
            frame, result = recon.get_latest_output()
            if frame is None or result is None:
                continue
            depth_image = result
            depth_vis = cv2.resize(depth_image, (frame.shape[1], frame.shape[0]))
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 2:
                raw_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                raw_vis = frame
            vis_list = [raw_vis, depth_vis]
            combined = np.hstack(vis_list)
            cv2.imshow(device_type, combined)
            key = cv2.waitKey(1)
            if key != -1:
                break
        recon.release()
        cv2.destroyAllWindows()
    elif mode == "gradient":
        print("\nPress any key to quit.\n")
        while True:
            frame, result = recon.get_latest_output()
            if frame is None or result is None:
                continue
            G = recon.get_gradient(frame)
            red = G[:, :, 0] * 255 / 3.0 + 127
            red = np.clip(red, 0, 255)
            blue = G[:, :, 1] * 255 / 3.0 + 127
            blue = np.clip(blue, 0, 255)
            grad_image = np.stack((blue, np.zeros_like(blue), red), axis=-1).astype(np.uint8)
            grad_vis = cv2.resize(grad_image, (frame.shape[1], frame.shape[0]))
            if frame.ndim == 2:
                raw_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                raw_vis = frame
            vis_list = [raw_vis, grad_vis]
            combined = np.hstack(vis_list)
            cv2.imshow(device_type, combined)
            key = cv2.waitKey(1)
            if key != -1:
                break
        recon.release()
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
            frame, pc = recon.get_latest_output()
            if pc is not None and frame is not None:
                pc = pc * 1000.0  # to mm
                colors = frame.reshape(-1, 3)[:, ::-1] / 255.0
                app.update(pc, colors)
            gui.Application.instance.post_to_main_thread(app.window, update_loop)
        gui.Application.instance.post_to_main_thread(app.window, update_loop)
        print("\nClose the window to quit.\n")
        gui.Application.instance.run()
        recon.release()
        gui.Application.instance.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test model: 2D or pointcloud visualization."
    )
    parser.add_argument(
        "--serial",
        type=str,
        required=True,
        help="sensor serial number (directory name under sensors/)",
    )
    parser.add_argument(
        "--sensors_root",
        type=str,
        default=DEFAULT_SENSORS_ROOT,
        help="root directory containing sensors/<serial>/",
    )
    parser.add_argument(
        "--use_mask", action="store_true", default=False,
        help="If set, only show masked (valid) area in the point cloud"
    )
    parser.add_argument(
        "--mode", type=str, choices=["depth", "gradient", "pointcloud"], default="depth",
        help="Visualization mode: depth, gradient or pointcloud"
    )
    parser.add_argument(
        "--refine_mask", action="store_true", default=False,
        help="If set, refine the mask using morphological operations"
    )
    parser.add_argument(
        "--relative", action="store_true", default=False,
        help="If set, use relative depth for point cloud"
    )
    parser.add_argument(
        "--relative_scale", type=float, default=0.5,
        help="Scale factor for relative depth"
    )
    parser.add_argument(
        "--device_type", type=str, choices=["cuda", "cpu"], default="cuda",
        help="Device type for model inference"
    )
    args = parser.parse_args()
    run_live_viewer(
        serial=args.serial,
        sensors_root=args.sensors_root,
        use_mask=args.use_mask,
        mode=args.mode,
        device_type=args.device_type,
        refine_mask=args.refine_mask,
        relative=args.relative,
        relative_scale=args.relative_scale
    )