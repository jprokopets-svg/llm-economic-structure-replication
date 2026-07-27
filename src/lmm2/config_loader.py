"""
Load and validate project configuration files.

Reads worlds.yaml, parameters.yaml, and models.yaml, and resolves
paths to v1 simulator configs.
"""

import os
from pathlib import Path
from typing import Any

import yaml


# Project root is two levels up from this file (src/lmm2/ -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# v1 repo root — resolved relative to project root
V1_ROOT = PROJECT_ROOT.parent / "LLM-Matrix"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_worlds(enabled_only: bool = True) -> dict[str, dict]:
    """
    Load world definitions from config/worlds.yaml.

    Args:
        enabled_only: If True, only return worlds with enabled=True.

    Returns:
        Dict mapping world_id -> world config dict, with v1_config resolved
        to an absolute path.
    """
    raw = load_yaml(CONFIG_DIR / "worlds.yaml")
    worlds = {}
    for world_id, world_config in raw["worlds"].items():
        if enabled_only and not world_config.get("enabled", False):
            continue
        # Resolve v1 config path relative to project root
        v1_config_path = PROJECT_ROOT / world_config["v1_config"]
        world_config["v1_config_resolved"] = str(v1_config_path.resolve())
        worlds[world_id] = world_config
    return worlds


def load_parameters() -> dict[str, dict]:
    """
    Load structural parameter definitions from config/parameters.yaml.

    Returns:
        Dict mapping parameter_name -> parameter config dict.
    """
    raw = load_yaml(CONFIG_DIR / "parameters.yaml")
    return raw["parameters"]


def load_models() -> list[dict]:
    """
    Load model definitions from config/models.yaml.

    Returns:
        List of model config dicts, each with keys: id, provider,
        two_stage_extraction.
    """
    raw = load_yaml(CONFIG_DIR / "models.yaml")
    return raw["models"]


def resolve_v1_config(relative_path: str) -> str:
    """
    Resolve a v1 config path (relative to project root) to absolute path.

    Raises FileNotFoundError if the resolved path doesn't exist.
    """
    resolved = (PROJECT_ROOT / relative_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"v1 config not found: {resolved}. "
            f"Make sure the LLM-Matrix repo is at {V1_ROOT}"
        )
    return str(resolved)
