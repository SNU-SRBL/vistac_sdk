import os
import sys

# Add parent directory to path so we can import apps.live_viewer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from apps.live_viewer import run_live_viewer

'''
This script runs a live viewer for real-time model testing with visualization.
It allows users to connect to a camera, visualize depth, gradient, or point cloud data,
and optionally apply a mask to the visualizations with various customization options.

Usage:
    python test_model.py --serial SERIAL [--sensors_root SENSORS_ROOT] [--device_type DEVICE_TYPE] [--mode MODE] [--use_mask] [--refine_mask] [--relative] [--relative_scale SCALE] [--mask_only_pointcloud] [--color_dist_threshold THRESHOLD] [--height_threshold THRESHOLD]
'''

DEFAULT_SENSORS_ROOT = os.path.join(os.path.dirname(__file__), "../sensors")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time model test with visualization")
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
        "--device_type",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device type for model inference"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["depth", "gradient", "pointcloud"],
        default="depth",
        help="Visualization mode: depth, gradient or pointcloud"
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="If set, only show masked (valid) area in the visualization"
    )
    parser.add_argument(
        "--refine_mask",
        action="store_true",
        default=False,
        help="If set, refine the mask using morphological operations"
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        default=False,
        help="If set, use relative depth for depth image"
    )
    parser.add_argument(
        "--relative_scale",
        type=float,
        default=0.5,
        help="Scale factor for relative depth"
    )
    parser.add_argument(
        "--mask_only_pointcloud",
        action="store_true",
        default=False,
        help="If set, use only masked area for point cloud"
    )
    parser.add_argument(
        "--color_dist_threshold",
        type=float,
        default=15,
        help="Color distance threshold for contact mask"
    )
    parser.add_argument(
        "--height_threshold",
        type=float,
        default=0.2,
        help="Height threshold for contact mask (in mm)"
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
        height_threshold=args.height_threshold
    )