import yaml
import os

def load_config(file_path="config.yaml"):
    """
    Loads a YAML configuration file and returns it as a Python dictionary.

    Args:
        file_path (str): Path to the YAML configuration file (default: 'config.yaml').

    Returns:
        dict: The configuration as a nested Python dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML parsing error in {file_path}: {e}")

    return config
