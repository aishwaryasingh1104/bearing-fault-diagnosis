"""
src/utils/config.py
Configuration loader and validator for YAML-based experiment configs.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_update(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base* and return the result."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a nested dict.

    Parameters
    ----------
    config_path:
        Path to a YAML config file.  When *None*, the ``configs/default.yaml``
        relative to the project root is used.

    Returns
    -------
    dict
        Fully resolved configuration dictionary.
    """
    if config_path is None:
        # Locate default config relative to this file's location
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    return cfg


def merge_configs(base_path: str | Path, override_path: str | Path) -> Dict[str, Any]:
    """Load two YAML files and merge the override on top of the base."""
    base = load_config(base_path)
    override = load_config(override_path)
    return _deep_update(base, override)


def get_nested(cfg: Dict, *keys: str, default: Any = None) -> Any:
    """Safely fetch a nested value from *cfg* using a sequence of keys."""
    node = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
    return node


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create all configured output directories if they do not already exist."""
    path_keys = ("checkpoint_dir", "export_dir", "log_dir")
    sections  = ("training", "evaluation")

    for section in sections:
        for path_key in path_keys:
            path = get_nested(cfg, section, path_key)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)

    # Dedicated logging section
    log_path = get_nested(cfg, "logging", "log_dir")
    if log_path:
        Path(log_path).mkdir(parents=True, exist_ok=True)
