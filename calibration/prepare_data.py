import argparse
import json
import os

import cv2
import numpy as np

from calibration.utils import load_csv_as_dict
from vistac_sdk.vistac_reconstruct import image2bgrxys
from vistac_sdk.utils import load_config

"""
This script prepares dataset for the tactile sensor calibration.
It is based on the collected and labeled data.

Prerequisite:
    - Tactile images collected using ball indenters with known diameters are collected.
    - Collected tactile images are labeled.

Usage:
    python prepare_data.py --serial SERIAL [--sensors_root SENSORS_ROOT] [--radius_reduction RADIUS_REDUCTION]

Arguments:
    --serial: Sensor serial number (directory name under sensors/)
    --sensors_root: (Optional) Root directory for sensors/<serial>/ (default: gs_sdk/sensors)
    --radius_reduction: (Optional) Reduce the radius of the labeled circle. This helps guarantee all labeled pixels are indented.
                        If not provided, 4 pixels will be reduced.
"""

DEFAULT_SENSORS_ROOT = os.path.join(os.path.dirname(__file__), "../sensors")

def prepare_data():
    # Argument Parsers
    parser = argparse.ArgumentParser(
        description="Use the labeled collected data to prepare the dataset files (npz)."
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
        "--radius_reduction",
        type=float,
        help="reduce the radius of the labeled circle. When not considering shadows, this helps guarantee all labeled pixels are indented.",
        default=4.0,
    )
    args = parser.parse_args()

    # Set up paths
    sensor_dir = os.path.join(args.sensors_root, args.serial)
    config_path = os.path.join(sensor_dir, f"{args.serial}.yaml")
    calib_dir = os.path.join(sensor_dir, "calibration")

    # Load the data_dict
    catalog_path = os.path.join(calib_dir, "catalog.csv")
    data_dict = load_csv_as_dict(catalog_path)
    diameters = np.array([float(diameter) for diameter in data_dict["diameter(mm)"]])
    experiment_reldirs = np.array(data_dict["experiment_reldir"])

    # Split data into train and test and save the split information
    perm = np.random.permutation(len(experiment_reldirs))
    n_train = 4 * len(experiment_reldirs) // 5
    data_path = os.path.join(calib_dir, "train_test_split.json")
    dict_to_save = {
        "train": experiment_reldirs[perm[:n_train]].tolist(),
        "test": experiment_reldirs[perm[n_train:]].tolist(),
    }
    with open(data_path, "w") as f:
        json.dump(dict_to_save, f, indent=4)

    # Read the configuration
    config = load_config(config_path=config_path)
    ppmm = config["ppmm"]

    # Extract the pixel data from each tactile image and calculate the gradients
    for experiment_reldir, diameter in zip(experiment_reldirs, diameters):
        experiment_dir = os.path.join(calib_dir, experiment_reldir)
        image_path = os.path.join(experiment_dir, "gelsight.png")
        image = cv2.imread(image_path)

        # Filter the non-indented pixels
        label_path = os.path.join(experiment_dir, "label.npz")
        label_data = np.load(label_path)
        center = label_data["center"]
        radius = label_data["radius"] - args.radius_reduction
        xys = np.dstack(
            np.meshgrid(
                np.arange(image.shape[1]), np.arange(image.shape[0]), indexing="xy"
            )
        )
        dists = np.linalg.norm(xys - center, axis=2)
        mask = dists < radius

        # Find the gradient angles, prepare the data, and save the data
        ball_radius = diameter * ppmm / 2.0
        if ball_radius < radius:
            print(experiment_reldir)
            print("Press too deep, deeper than the ball radius")
            continue
        dxys = xys - center
        dists[np.logical_not(mask)] = 0.0
        dzs = np.sqrt(ball_radius**2 - np.square(dists))
        gxangles = np.arctan2(dxys[:, :, 0], dzs)
        gyangles = np.arctan2(dxys[:, :, 1], dzs)
        gxyangles = np.stack([gxangles, gyangles], axis=-1)
        gxyangles[np.logical_not(mask)] = np.array([0.0, 0.0])
        bgrxys = image2bgrxys(image)
        save_path = os.path.join(experiment_dir, "data.npz")
        np.savez(save_path, bgrxys=bgrxys, gxyangles=gxyangles, mask=mask)

    # Save the background data
    bg_path = os.path.join(calib_dir, "background.png")
    bg_image = cv2.imread(bg_path)
    bgrxys = image2bgrxys(bg_image)
    gxyangles = np.zeros((bg_image.shape[0], bg_image.shape[1], 2))
    mask = np.ones((bg_image.shape[0], bg_image.shape[1]), dtype=np.bool_)
    save_path = os.path.join(calib_dir, "background_data.npz")
    np.savez(save_path, bgrxys=bgrxys, gxyangles=gxyangles, mask=mask)


if __name__ == "__main__":
    prepare_data()