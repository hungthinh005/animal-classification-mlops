"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration file
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        config: Original configuration dictionary
        updates: Updates to apply
        
    Returns:
        Updated configuration dictionary
    """
    import copy
    updated_config = copy.deepcopy(config)
    
    def deep_update(original, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and key in original:
                original[key] = deep_update(original.get(key, {}), value)
            else:
                original[key] = value
        return original
    
    return deep_update(updated_config, updates)

