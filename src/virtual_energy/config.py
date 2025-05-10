"""
Configuration module for virtual-energy package.

This module loads configuration from pyproject.toml and provides it to the rest of the application.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings

from virtual_energy.models.config import BatteryConfig

# Try to import tomllib (Python 3.11+) or tomli as a fallback
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        warnings.warn(
            "Neither tomllib nor tomli is installed. Configuration loading from pyproject.toml will be disabled.",
            ImportWarning
        )
        tomllib = None

# Default configuration values
DEFAULT_CONFIG = {
    "default_iso": "ercot",
    "battery": {
        "e_max_mwh": 200,
        "p_max_mw": 25,
        "eta_chg": 0.95,
        "delta_t": 0.25,
        "initial_soc_pct": 0.5,
    },
    "optimisers": {
        "enabled": ["oracle_lp", "online_mpc", "online_quartile"],
    },
    "forecasters": {
        "ridge": {
            "lags": 4,
            "alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
            "cv": 5,
        },
        "ewma": {
            "spans": [12, 24, 48, 96],
        },
        "quartile": {
            "percentiles": [10, 25, 45],
            "window_sizes": [672],
        },
    },
    "benchmark": {
        "output_dir": "benchmark_results",
        "max_nodes": 100,
        "n_jobs": -1,
    },
}

def find_pyproject_toml() -> Optional[Path]:
    """Find the pyproject.toml file."""
    # Start with the current directory and go up
    current_dir = Path.cwd()
    
    # Try the current directory first
    pyproject_path = current_dir / "pyproject.toml"
    if pyproject_path.exists():
        return pyproject_path
    
    # Then go up the directory tree
    for parent in current_dir.parents:
        pyproject_path = parent / "pyproject.toml"
        if pyproject_path.exists():
            return pyproject_path
    
    # If we can't find it, return None
    return None

def load_config() -> Dict[str, Any]:
    """
    Load configuration from pyproject.toml or use defaults.
    
    Returns:
        Dict containing the configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    # Check if tomllib/tomli is available
    if tomllib is None:
        warnings.warn(
            "No TOML parser available. Using default configuration.",
            UserWarning
        )
        return config
    
    # Try to find and load the pyproject.toml file
    pyproject_path = find_pyproject_toml()
    if pyproject_path is None:
        warnings.warn(
            "Could not find pyproject.toml. Using default configuration.",
            UserWarning
        )
        return config
    
    try:
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        # Extract the virtual-energy configuration
        if "tool" in pyproject_data and "virtual-energy" in pyproject_data["tool"]:
            ve_config = pyproject_data["tool"]["virtual-energy"]
            
            # Update the config with values from pyproject.toml
            # Top-level keys
            for key in config:
                if key in ve_config:
                    if isinstance(config[key], dict) and isinstance(ve_config[key], dict):
                        # For nested dictionaries, update recursively
                        config[key].update(ve_config[key])
                    else:
                        # For simple values, replace
                        config[key] = ve_config[key]
        else:
            warnings.warn(
                "No [tool.virtual-energy] section found in pyproject.toml. Using default configuration.",
                UserWarning
            )
    except Exception as e:
        warnings.warn(
            f"Error loading configuration from pyproject.toml: {e}. Using default configuration.",
            UserWarning
        )
    
    return config

# Global configuration object
CONFIG = load_config()

def get_config() -> Dict[str, Any]:
    """Get the configuration."""
    return CONFIG

def get_battery_config() -> BatteryConfig:
    """
    Get the battery configuration as a BatteryConfig object.
    
    Returns:
        BatteryConfig: A pydantic model with the battery configuration
    """
    # Create a BatteryConfig instance with the values from the config
    return BatteryConfig(**CONFIG["battery"])

def get_optimisers() -> List[str]:
    """Get the list of enabled optimisers."""
    return CONFIG["optimisers"]["enabled"]

def get_forecaster_config(forecaster: str) -> Dict[str, Any]:
    """
    Get the configuration for a specific forecaster.
    
    Args:
        forecaster: The name of the forecaster (e.g., 'ridge', 'ewma', 'quartile')
        
    Returns:
        Dict containing the forecaster configuration
    """
    if forecaster not in CONFIG["forecasters"]:
        warnings.warn(
            f"No configuration found for forecaster '{forecaster}'. Using default.",
            UserWarning
        )
        # Return an empty dict to avoid errors
        return {}
    
    return CONFIG["forecasters"][forecaster]

def get_benchmark_config() -> Dict[str, Any]:
    """Get the benchmark configuration."""
    return CONFIG["benchmark"] 