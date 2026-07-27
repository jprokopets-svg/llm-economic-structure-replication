"""Tests for scoring module, including clustered bootstrap."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lmm2.scoring import (
    run_sign_and_slope_regression,
    run_sign_and_slope_bootstrap,
    _summarize_bootstrap_betas,
)


def test_sign_and_slope_perfect_tracking():
    """When model perfectly tracks true response, beta should be ~1.0."""
    true = [1.0, 2.0, 3.0, 4.0, 5.0]
    model = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = run_sign_and_slope_regression(true, model)

    assert abs(result["beta"] - 1.0) < 0.01
    assert result["significant"] is True


def test_sign_and_slope_no_relationship():
    """When model response is constant, beta should be ~0."""
    true = [1.0, 2.0, 3.0, 4.0, 5.0]
    model = [3.0, 3.0, 3.0, 3.0, 3.0]
    result = run_sign_and_slope_regression(true, model)

    assert abs(result["beta"]) < 0.01
    assert result["significant"] is False


def test_sign_and_slope_too_few_points():
    """With fewer than 3 points, result should be NaN."""
    result = run_sign_and_slope_regression([1.0, 2.0], [1.0, 2.0])
    assert result["significant"] is False
    assert np.isnan(result["beta"])


def test_bootstrap_returns_both_methods():
    """Bootstrap function returns both i.i.d. and clustered results."""
    observations = []
    for setting in ["s1", "s2", "s3", "s4"]:
        for horizon in [1, 2, 3, 4]:
            true_delta = float(hash(setting) % 10) / 10.0
            model_delta = true_delta * 0.8 + 0.1
            observations.append({
                "true_delta": true_delta,
                "model_delta": model_delta,
                "cluster_id": f"world1__{setting}",
            })

    result = run_sign_and_slope_bootstrap(
        observations, n_resamples=500, seed=42,
    )

    assert "ols" in result
    assert "bootstrap_iid" in result
    assert "bootstrap_clustered" in result
    assert result["n_observations"] == 16
    assert result["n_clusters"] == 4


def test_clustered_ci_wider_than_iid():
    """
    Clustered bootstrap CI should generally be wider than i.i.d. CI
    when there is within-cluster correlation.

    We create data where observations within a cluster are identical
    (maximum within-cluster correlation), so the clustered CI should
    be noticeably wider.
    """
    observations = []
    rng = np.random.default_rng(123)
    for setting_idx in range(6):
        true_delta = float(setting_idx) * 0.5
        model_delta = true_delta * 0.7 + rng.normal(0, 0.1)
        # All horizons within a cluster get the SAME deltas
        # (maximum within-cluster correlation)
        for horizon in [1, 2, 3, 4]:
            observations.append({
                "true_delta": true_delta,
                "model_delta": model_delta,
                "cluster_id": f"world1__s{setting_idx}",
            })

    result = run_sign_and_slope_bootstrap(
        observations, n_resamples=5000, seed=42,
    )

    iid_width = (
        result["bootstrap_iid"]["beta_ci_high"]
        - result["bootstrap_iid"]["beta_ci_low"]
    )
    clustered_width = (
        result["bootstrap_clustered"]["beta_ci_high"]
        - result["bootstrap_clustered"]["beta_ci_low"]
    )

    # With maximum within-cluster correlation, clustered CI should be wider
    assert clustered_width > iid_width, (
        f"Expected clustered CI ({clustered_width:.4f}) to be wider "
        f"than i.i.d. CI ({iid_width:.4f})"
    )


def test_bootstrap_too_few_observations():
    """Bootstrap with <3 observations returns NaN results."""
    observations = [
        {"true_delta": 1.0, "model_delta": 1.0, "cluster_id": "c1"},
        {"true_delta": 2.0, "model_delta": 2.0, "cluster_id": "c2"},
    ]
    result = run_sign_and_slope_bootstrap(observations, n_resamples=100)

    assert np.isnan(result["bootstrap_iid"]["beta_mean"])
    assert np.isnan(result["bootstrap_clustered"]["beta_mean"])


def test_summarize_bootstrap_betas_significance():
    """Beta is significant only when CI excludes zero and mean is positive."""
    # All positive betas — should be significant
    positive_betas = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    result = _summarize_bootstrap_betas(positive_betas, 6)
    assert result["significant"] is True
    assert result["beta_ci_low"] > 0

    # Betas spanning zero — should NOT be significant
    mixed_betas = np.array([-0.5, -0.2, 0.1, 0.3, 0.5, 0.8])
    result = _summarize_bootstrap_betas(mixed_betas, 6)
    assert result["significant"] is False
