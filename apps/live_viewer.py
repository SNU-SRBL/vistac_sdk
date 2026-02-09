import argparse
import os
import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from vistac_sdk.live_core import LiveTactileProcessor
from vistac_sdk.viz_utils import visualize_force_field, visualize_force_vector

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
    relative_scale=0.5,
    mask_only_pointcloud=False,
    color_dist_threshold=15,
    height_threshold=0.2,
    enable_force=False,
    temporal_stride=5,
    outputs=None
):
    # Determine outputs based on mode or explicit outputs parameter
    if outputs is None:
        # Legacy mode-based selection
        if mode == "depth":
            outputs = ['depth']
        elif mode == "gradient":
            outputs = ['gradient']
        elif mode == "pointcloud":
            outputs = ['pointcloud']
        elif mode == "force_field":
            outputs = ['force_field']
        elif mode == "force_vector":
            outputs = ['force_vector']
        else:
            outputs = ['depth']
    
    # Determine which estimators to enable based on requested outputs
    enable_depth_estimator = any(x in outputs for x in ['depth', 'gradient', 'pointcloud', 'mask'])
    enable_force_estimator = enable_force or any(x in outputs for x in ['force_field', 'force_vector'])
    
    processor = LiveTactileProcessor(
        serial=serial,
        sensors_root=sensors_root,
        model_device=device_type,
        enable_depth=enable_depth_estimator,
        enable_force=enable_force_estimator,
        temporal_stride=temporal_stride,
        outputs=outputs,
        use_mask=use_mask,
        refine_mask=refine_mask,
        relative=relative,
        relative_scale=relative_scale,
        mask_only_pointcloud=mask_only_pointcloud,
        color_dist_threshold=color_dist_threshold,
        height_threshold=height_threshold
    )
    device_type = processor.device_type
    ppmm = processor.ppmm

    # Multi-panel mode: display multiple outputs side by side
    if len(outputs) > 1 and 'pointcloud' not in outputs:
        print("\nMulti-panel mode. Press any key to quit.\n")
        while True:
            frame, result_dict = processor.get_latest_output()
            if frame is None or not result_dict:
                continue
            
            vis_panels = []
            
            # Always show raw frame first
            if frame.ndim == 2:
                raw_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                raw_vis = frame
            vis_panels.append(raw_vis)
            
            # Add visualization for each requested output
            for output_name in outputs:
                output_data = result_dict.get(output_name)
                
                if output_name == 'depth' and output_data is not None:
                    depth_vis = cv2.resize(output_data, (frame.shape[1], frame.shape[0]))
                    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                    vis_panels.append(depth_vis)
                
                elif output_name == 'gradient' and output_data is not None:
                    G = output_data
                    red = G[:, :, 0] * 255 / 3.0 + 127
                    red = np.clip(red, 0, 255)
                    blue = G[:, :, 1] * 255 / 3.0 + 127
                    blue = np.clip(blue, 0, 255)
                    grad_image = np.stack((blue, np.zeros_like(blue), red), axis=-1).astype(np.uint8)
                    grad_vis = cv2.resize(grad_image, (frame.shape[1], frame.shape[0]))
                    vis_panels.append(grad_vis)
                
                elif output_name == 'force_field':
                    if output_data is None:
                        # Warmup message
                        warmup_panel = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                        cv2.putText(warmup_panel, "Buffering...", (10, frame.shape[0]//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        vis_panels.append(warmup_panel)
                    else:
                        force_vis = visualize_force_field(
                            output_data['normal'],
                            output_data['shear'],
                            overlay_image=frame
                        )
                        vis_panels.append(force_vis)
                
                elif output_name == 'force_vector':
                    if output_data is None:
                        # Warmup message
                        warmup_panel = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                        cv2.putText(warmup_panel, "Buffering...", (10, frame.shape[0]//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        vis_panels.append(warmup_panel)
                    else:
                        force_vis = visualize_force_vector(
                            output_data['fx'],
                            output_data['fy'],
                            output_data['fz'],
                            overlay_image=frame
                        )
                        vis_panels.append(force_vis)
            
            # Combine all panels horizontally
            if len(vis_panels) > 0:
                combined = np.hstack(vis_panels)
                cv2.imshow(device_type, combined)
            
            key = cv2.waitKey(1)
            if key != -1:
                break
        
        processor.release()
        cv2.destroyAllWindows()
        return

    # Single output modes below
    if mode == "depth":
        print("\nPress any key to quit.\n")
        while True:
            frame, result_dict = processor.get_latest_output()
            if frame is None or not result_dict:
                continue
            depth_image = result_dict.get('depth')
            if depth_image is None:
                continue
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
        processor.release()
        cv2.destroyAllWindows()
    elif mode == "gradient":
        print("\nPress any key to quit.\n")
        while True:
            frame, result_dict = processor.get_latest_output()
            if frame is None or not result_dict:
                continue
            G = result_dict.get('gradient')
            if G is None:
                continue
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
        processor.release()
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
            frame, result_dict = processor.get_latest_output()
            if not result_dict:
                gui.Application.instance.post_to_main_thread(app.window, update_loop)
                return
            pc = result_dict.get('pointcloud')
            if pc is not None and frame is not None:
                pc = pc * 1000.0  # to mm
                colors = frame.reshape(-1, 3)[:, ::-1] / 255.0
                app.update(pc, colors)
            gui.Application.instance.post_to_main_thread(app.window, update_loop)
        gui.Application.instance.post_to_main_thread(app.window, update_loop)
        print("\nClose the window to quit.\n")
        gui.Application.instance.run()
        processor.release()
        gui.Application.instance.quit()
    elif mode == "force_field":
        print("\nForce field visualization. Press any key to quit.\n")
        while True:
            frame, result_dict = processor.get_latest_output()
            if frame is None or not result_dict:
                continue
            
            force_data = result_dict.get('force_field')
            
            if force_data is None:
                # Display warmup message
                if frame.ndim == 2:
                    raw_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    raw_vis = frame
                warmup_panel = np.zeros_like(raw_vis)
                cv2.putText(warmup_panel, "Buffering...", (10, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                combined = np.hstack([raw_vis, warmup_panel])
                cv2.imshow(device_type, combined)
            else:
                # Visualize force field as RGB heatmap overlaid on raw image
                force_vis = visualize_force_field(
                    force_data['normal'],
                    force_data['shear'],
                    overlay_image=frame,
                    alpha=0.6
                )
                if frame.ndim == 2:
                    raw_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    raw_vis = frame
                combined = np.hstack([raw_vis, force_vis])
                cv2.imshow(device_type, combined)
            
            key = cv2.waitKey(1)
            if key != -1:
                break
        processor.release()
        cv2.destroyAllWindows()
    elif mode == "force_vector":
        print("\nForce vector visualization. Press any key to quit.\n")
        while True:
            frame, result_dict = processor.get_latest_output()
            if frame is None or not result_dict:
                continue
            
            force_data = result_dict.get('force_vector')
            
            if force_data is None:
                # Display warmup message
                if frame.ndim == 2:
                    raw_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    raw_vis = frame
                warmup_panel = np.zeros_like(raw_vis)
                cv2.putText(warmup_panel, "Buffering...", (10, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                combined = np.hstack([raw_vis, warmup_panel])
                cv2.imshow(device_type, combined)
            else:
                # Visualize force vector as arrow/circle overlay on raw image
                force_vis = visualize_force_vector(
                    force_data['fx'],
                    force_data['fy'],
                    force_data['fz'],
                    overlay_image=frame,
                    arrow_scale=50.0,
                    arrow_color=(0, 255, 0),
                    arrow_thickness=3
                )
                if frame.ndim == 2:
                    raw_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    raw_vis = frame
                combined = np.hstack([raw_vis, force_vis])
                cv2.imshow(device_type, combined)
            
            key = cv2.waitKey(1)
            if key != -1:
                break
        processor.release()
        cv2.destroyAllWindows()

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
        "--mode", type=str, 
        choices=["depth", "gradient", "pointcloud", "force_field", "force_vector"], 
        default="depth",
        help="Visualization mode: depth, gradient, pointcloud, force_field, or force_vector"
    )
    parser.add_argument(
        "--refine_mask", action="store_true", default=False,
        help="If set, refine the mask using morphological operations"
    )
    parser.add_argument(
        "--relative", action="store_true", default=False,
        help="If set, use relative depth for depth image"
    )
    parser.add_argument(
        "--relative_scale", type=float, default=0.5,
        help="Scale factor for relative depth"
    )
    parser.add_argument(
        "--mask_only_pointcloud", action="store_true", default=False,
        help="If set, use only masked area for point cloud"
    )
    parser.add_argument(
        "--device_type", type=str, choices=["cuda", "cpu"], default="cuda",
        help="Device type for model inference"
    )
    parser.add_argument(
        "--color_dist_threshold", type=float, default=15,
        help="Color distance threshold for contact mask"
    )
    parser.add_argument(
        "--height_threshold", type=float, default=0.2,
        help="Height threshold for contact mask (in mm)"
    )
    parser.add_argument(
        "--enable_force", action="store_true", default=False,
        help="If set, enable force estimation (requires Sparsh models)"
    )
    parser.add_argument(
        "--temporal_stride", type=int, default=5,
        help="Temporal stride for force estimation (frames between temporal pair)"
    )
    parser.add_argument(
        "--outputs", type=str, nargs='+', default=None,
        help="Explicit list of outputs to compute (e.g., depth force_field force_vector)"
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
        relative_scale=args.relative_scale,
        mask_only_pointcloud=args.mask_only_pointcloud,
        color_dist_threshold=args.color_dist_threshold,
        height_threshold=args.height_threshold,
        enable_force=args.enable_force,
        temporal_stride=args.temporal_stride,
        outputs=args.outputs
    )