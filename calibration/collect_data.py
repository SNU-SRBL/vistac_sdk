import argparse
import os

import cv2
import numpy as np

from vistac_sdk.vistac_device import Camera
from calibration.utils import load_csv_as_dict
from vistac_sdk.utils import load_config

"""
This script collects tactile data using ball indenters for sensor calibration.

Usage:
    python collect_data.py --serial SERIAL --ball_diameter DIAMETER [--sensors_root SENSORS_ROOT]

Arguments:
    --serial: Sensor serial number (directory name under sensors/)
    --ball_diameter: Diameter of the ball indenter in mm
    --sensors_root: (Optional) Root directory for sensors/<serial>/ (default: gs_sdk/sensors)
"""

DEFAULT_SENSORS_ROOT = os.path.join(os.path.dirname(__file__), "../sensors")

def overlay_grid(img, rows=5, cols=5, color=(0, 255, 0), thickness=1):
    """Draw a rows×cols grid overlay on img."""
    h, w = img.shape[:2]
    for i in range(1, rows):
        y = int(i * h / rows)
        cv2.line(img, (0, y), (w, y), color, thickness)
    for j in range(1, cols):
        x = int(j * w / cols)
        cv2.line(img, (x, 0), (x, h), color, thickness)
    return img

def collect_data():
    parser = argparse.ArgumentParser(
        description="Collect calibration data with ball indenters to calibrate the sensor."
    )
    parser.add_argument(
        "--serial",
        type=str,
        required=True,
        help="sensor serial number (directory name under sensors/)",
    )
    parser.add_argument(
        "-d", "--ball_diameter", type=float, required=True, help="diameter of the indenter in mm"
    )
    parser.add_argument(
        "--sensors_root",
        type=str,
        default=DEFAULT_SENSORS_ROOT,
        help="root directory containing sensors/<serial>/",
    )
    args = parser.parse_args()

    # Set up paths
    sensor_dir = os.path.join(args.sensors_root, args.serial)
    config_path = os.path.join(sensor_dir, f"{args.serial}.yaml")
    calib_dir = os.path.join(sensor_dir, "calibration")
    if not os.path.isdir(calib_dir):
        os.makedirs(calib_dir)
    ball_diameter = args.ball_diameter
    indenter_subdir = "%.3fmm" % (ball_diameter)
    indenter_dir = os.path.join(calib_dir, indenter_subdir)
    if not os.path.isdir(indenter_dir):
        os.makedirs(indenter_dir)

    # Load config
    config = load_config(config_path=config_path)
    device_type = config["device_type"]
    imgh = config["imgh"]
    imgw = config["imgw"]
    raw_imgh = config["raw_imgh"]
    raw_imgw = config["raw_imgw"]
    framerate = config["framerate"]

    # Create the data saving catalog
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    if not os.path.isfile(catalog_path):
        with open(catalog_path, "w") as f:
            f.write("experiment_reldir,diameter(mm)\n")

    # Find last data_count collected with this diameter
    data_dict = load_csv_as_dict(catalog_path)
    diameters = np.array([float(diameter) for diameter in data_dict["diameter(mm)"]]) if data_dict["diameter(mm)"] else np.array([])
    data_idxs = np.where(np.abs(diameters - ball_diameter) < 1e-3)[0] if diameters.size > 0 else []
    data_counts = np.array(
        [int(os.path.basename(reldir)) for reldir in data_dict["experiment_reldir"]]
    ) if data_dict["experiment_reldir"] else np.array([])
    if len(data_idxs) == 0:
        data_count = 0
    else:
        data_count = max(data_counts[data_idxs]) + 1

    # Connect to the device and collect data until quit
    device = Camera(serial=args.serial, sensors_root=args.sensors_root)
    device.connect()
    print("Press key to collect data, collect background, or quit (w/b/q)")
    while True:
        image = device.get_image()
        vis = overlay_grid(image.copy(), rows=5, cols=5)
        cv2.imshow("frame", vis)
        key = cv2.waitKey(1)
        if key == ord("w"):
            experiment_reldir = os.path.join(indenter_subdir, str(data_count))
            experiment_dir = os.path.join(calib_dir, experiment_reldir)
            if not os.path.isdir(experiment_dir):
                os.makedirs(experiment_dir)
            save_path = os.path.join(experiment_dir, "gelsight.png")
            cv2.imwrite(save_path, image)
            print("Save data to new path: %s" % save_path)
            with open(catalog_path, "a") as f:
                f.write(experiment_reldir + "," + str(ball_diameter))
                f.write("\n")
            data_count += 1
        elif key == ord("b"):
            print("Collecting 10 background images, please wait ...")
            images = []
            for _ in range(10):
                img = device.get_image()
                images.append(img)
                vis_bg = overlay_grid(img.copy(), rows=5, cols=5)
                cv2.imshow("frame", vis_bg)
                cv2.waitKey(1)
            image = np.mean(images, axis=0).astype(np.uint8)
            save_path = os.path.join(calib_dir, "background.png")
            cv2.imwrite(save_path, image)
            print("Save background image to %s" % save_path)
        elif key == ord("q"):
            break
        elif key == -1:
            continue
        else:
            print("Unrecognized key %s" % key)

    device.release()
    cv2.destroyAllWindows()
    print("%d images collected in total." % data_count)

if __name__ == "__main__":
    collect_data()