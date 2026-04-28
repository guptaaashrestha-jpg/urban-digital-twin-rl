"""
Configuration loader for the Urban Digital Twin.
Reads YAML config files and provides dot-notation access.
"""

import yaml
import os


class Config:
    """Recursive attribute-access wrapper around a dictionary."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"Config({vars(self)})"


def load_config(path: str = None) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to YAML config file. Defaults to configs/default.yaml
              relative to project root.

    Returns:
        Config object with attribute access to all settings.
    """
    if path is None:
        # Find project root (where configs/ directory lives)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(project_root, "configs", "default.yaml")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return Config(data)
