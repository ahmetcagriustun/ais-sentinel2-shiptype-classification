import yaml
import os

def load_config(file_path="config.yaml"):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML parsing error in {file_path}: {e}")

    return config
