"""Tests for config loading and path resolution."""

import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lmm2.config_loader import load_worlds, load_parameters, load_models


def test_load_worlds_returns_enabled():
    """Only enabled worlds are returned by default."""
    worlds = load_worlds(enabled_only=True)
    for world_id, config in worlds.items():
        assert config["enabled"] is True


def test_load_worlds_all():
    """All worlds are returned when enabled_only=False."""
    worlds = load_worlds(enabled_only=False)
    assert "world1" in worlds
    assert "world2" in worlds
    assert "world3" in worlds
    assert "world4" in worlds


def test_load_parameters():
    """Parameters load with expected keys."""
    params = load_parameters()
    assert "phillips_slope" in params
    assert "taylor_inflation_coefficient" in params

    phillips = params["phillips_slope"]
    assert phillips["baseline"] == 0.4
    assert len(phillips["settings"]) == 5
    assert "direction_hints" in phillips


def test_load_models():
    """Models load as a list with expected fields."""
    models = load_models()
    assert len(models) >= 1
    for model in models:
        assert "id" in model
        assert "provider" in model
