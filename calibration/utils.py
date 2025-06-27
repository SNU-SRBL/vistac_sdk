import csv
import numpy as np
import os

def load_csv_as_dict(csv_path):
    """
    Load the csv file entries as dictionaries.

    :params csv_path: str; the path of the csv file.
    :returns: dict; the dictionary of the csv file.
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        data = list(reader)
        keys = reader.fieldnames
        data_dict = {}
        for key in keys:
            data_dict[key] = []
        for line in data:
            for key in keys:
                data_dict[key].append(line[key])
    return data_dict

def list_sensors(sensors_root):
    """
    List all sensors in the sensors_root directory that have a matching <serial>.yaml config file.
    """
    sensors = []
    for entry in os.listdir(sensors_root):
        sensor_dir = os.path.join(sensors_root, entry)
        config_path = os.path.join(sensor_dir, f"{entry}.yaml")
        if os.path.isdir(sensor_dir) and os.path.isfile(config_path):
            sensors.append(entry)
    return sensors

if __name__ == "__main__":
    # Default sensors_root is ../sensors relative to this script
    sensors_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sensors"))
    sensors = list_sensors(sensors_root)
    if not sensors:
        print("No sensors found in", sensors_root)
    else:
        print("Available sensors:")
        for s in sensors:
            print(f"  {s}")