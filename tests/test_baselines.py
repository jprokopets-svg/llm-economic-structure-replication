"""Tests for statistical baselines (Naive, AR(1), VAR)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lmm2.config_loader import load_worlds
from lmm2.world_runner import get_simulator_class, generate_history
from lmm2.baselines import naive_baseline, ar1_baseline, var_baseline


@pytest.fixture
def world1_history():
    """Generate a 60-quarter World 1 history for baseline testing."""
    worlds = load_worlds(enabled_only=False)
    w1 = worlds["world1"]
    sim_class = get_simulator_class(w1["v1_simulator"])
    config_path = w1["v1_config_resolved"]
    history = generate_history(config_path, sim_class, n_periods=60, seed=42)
    return history


@pytest.fixture
def world2_history():
    """Generate a 60-quarter World 2 history for baseline testing."""
    worlds = load_worlds(enabled_only=False)
    w2 = worlds["world2"]
    sim_class = get_simulator_class(w2["v1_simulator"])
    config_path = w2["v1_config_resolved"]
    history = generate_history(config_path, sim_class, n_periods=60, seed=42)
    return history


WORLD1_VARS = ["y", "pi", "r", "e", "u"]
WORLD2_VARS = ["y", "pi", "r", "u"]


def test_naive_baseline_format(world1_history):
    """Naive baseline returns forecasts in the expected format."""
    forecast = naive_baseline(world1_history, WORLD1_VARS, n_horizons=4)

    # Check all expected keys are present
    for var in WORLD1_VARS:
        for h in range(1, 5):
            key = f"{var}_{h}"
            assert key in forecast, f"Missing key: {key}"
            assert "point" in forecast[key]
            assert "ci_low" in forecast[key]
            assert "ci_high" in forecast[key]


def test_naive_baseline_constant(world1_history):
    """Naive baseline predicts the same value at all horizons."""
    forecast = naive_baseline(world1_history, WORLD1_VARS, n_horizons=4)

    for var in WORLD1_VARS:
        values = [forecast[f"{var}_{h}"]["point"] for h in range(1, 5)]
        # All horizons should have the same point estimate
        assert all(v == values[0] for v in values)


def test_ar1_baseline_format(world1_history):
    """AR(1) baseline returns forecasts in the expected format."""
    forecast = ar1_baseline(world1_history, WORLD1_VARS, n_horizons=4)

    for var in WORLD1_VARS:
        for h in range(1, 5):
            key = f"{var}_{h}"
            assert key in forecast
            # CI should widen: ci_low <= point <= ci_high
            assert forecast[key]["ci_low"] <= forecast[key]["point"]
            assert forecast[key]["point"] <= forecast[key]["ci_high"]


def test_var_baseline_format(world1_history):
    """VAR baseline returns forecasts in the expected format."""
    forecast = var_baseline(world1_history, WORLD1_VARS, n_horizons=4)

    for var in WORLD1_VARS:
        for h in range(1, 5):
            key = f"{var}_{h}"
            assert key in forecast
            assert forecast[key]["ci_low"] <= forecast[key]["point"]
            assert forecast[key]["point"] <= forecast[key]["ci_high"]


def test_var_baseline_ci_widens(world1_history):
    """VAR forecast uncertainty should generally increase with horizon."""
    forecast = var_baseline(world1_history, WORLD1_VARS, n_horizons=4)

    for var in WORLD1_VARS:
        widths = []
        for h in range(1, 5):
            key = f"{var}_{h}"
            width = forecast[key]["ci_high"] - forecast[key]["ci_low"]
            widths.append(width)
        # h=4 CI should be at least as wide as h=1 CI
        assert widths[3] >= widths[0], (
            f"VAR CI for {var} did not widen: h1={widths[0]:.4f}, h4={widths[3]:.4f}"
        )


def test_var_baseline_world2(world2_history):
    """VAR works on World 2 (closed economy, 4 variables)."""
    forecast = var_baseline(world2_history, WORLD2_VARS, n_horizons=4)

    for var in WORLD2_VARS:
        for h in range(1, 5):
            key = f"{var}_{h}"
            assert key in forecast


def test_var_baseline_fails_loud_with_one_variable(world1_history):
    """VAR must raise ValueError if given fewer than 2 variables."""
    with pytest.raises(ValueError, match="at least 2 variables"):
        var_baseline(world1_history, ["y"], n_horizons=4)


def test_all_baselines_same_format(world1_history):
    """All three baselines produce the same set of keys."""
    naive = naive_baseline(world1_history, WORLD1_VARS, n_horizons=4)
    ar1 = ar1_baseline(world1_history, WORLD1_VARS, n_horizons=4)
    var = var_baseline(world1_history, WORLD1_VARS, n_horizons=4)

    assert set(naive.keys()) == set(ar1.keys()) == set(var.keys())
