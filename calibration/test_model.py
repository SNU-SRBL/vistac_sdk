from apps.live_viewer import run_live_viewer

'''
This script runs a live viewer for real-time model testing with visualization.
It allows users to connect to a camera, visualize depth, gradient, or point cloud data,
and optionally apply a mask to the visualizations.

Usage:
    python live_viewer.py --serial SERIAL --sensors_root SENSORS_ROOT [--device_type DEVICE_TYPE] [--mode MODE] [--use_mask]
'''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time model test with visualization")
    parser.add_argument("--serial", type=str, required=True)
    parser.add_argument("--sensors_root", type=str, default="sensors")
    parser.add_argument("--device_type", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--mode", type=str, choices=["depth", "gradient", "pointcloud"], default="depth")
    parser.add_argument("--use_mask", action="store_true", default=False)
    args = parser.parse_args()
    run_live_viewer(
        serial=args.serial,
        sensors_root=args.sensors_root,
        use_mask=args.use_mask,
        mode=args.mode,
        device_type=args.device_type
    )