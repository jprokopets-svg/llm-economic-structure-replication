"""
Scoring module: compare model forecasts against Monte Carlo ground truth.

Computes:
  1. Directional accuracy (sign test): did the forecast move in the right
     direction when the structural parameter changed?
  2. Sign-and-slope regression: does the model response scale with the
     true structural effect?
  3. TOLD-minus-INFER gap: difference in the above between conditions.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_directional_accuracy(
    baseline_mc_mean: dict[str, list[float]],
    modified_mc_mean: dict[str, list[float]],
    baseline_forecast: dict[str, dict],
    modified_forecast: dict[str, dict],
    direction_hints: dict[str, int],
    parameter_value: float,
    baseline_value: float,
) -> dict[str, dict]:
    """
    Check if the model's forecast moved in the correct direction.

    For each variable, compare:
      - True direction: sign of (modified_mc_mean - baseline_mc_mean)
      - Model direction: sign of (modified_forecast - baseline_forecast)

    Args:
        baseline_mc_mean: MC mean forecasts under baseline parameter.
        modified_mc_mean: MC mean forecasts under modified parameter.
        baseline_forecast: Model's forecast under baseline parameter.
        modified_forecast: Model's forecast under modified parameter.
        direction_hints: Expected direction per variable (+1, -1, 0=skip).
        parameter_value: The modified parameter value.
        baseline_value: The baseline parameter value.

    Returns:
        Dict mapping variable -> {
            "correct": bool,
            "true_direction": int (+1 or -1),
            "model_direction": int (+1 or -1),
            "true_delta": float,
            "model_delta": float,
            "horizon": int,
        }
        One entry per (variable, horizon) pair.
    """
    results = {}
    param_direction = 1 if parameter_value > baseline_value else -1

    for variable, expected_sign_per_unit in direction_hints.items():
        if expected_sign_per_unit == 0:
            # Ambiguous direction — skip this variable
            continue

        # Expected direction for this specific parameter change
        expected_direction = expected_sign_per_unit * param_direction

        # Check each forecast horizon
        for horizon_idx, mc_baseline_val in enumerate(baseline_mc_mean.get(variable, [])):
            horizon = horizon_idx + 1
            mc_modified_val = modified_mc_mean.get(variable, [None])[horizon_idx]
            if mc_modified_val is None:
                continue

            # True direction from MC
            true_delta = mc_modified_val - mc_baseline_val
            true_direction = 1 if true_delta > 0 else -1

            # Model direction from forecasts
            forecast_key = f"{variable}_{horizon}"
            baseline_point = baseline_forecast.get(forecast_key, {}).get("point")
            modified_point = modified_forecast.get(forecast_key, {}).get("point")

            if baseline_point is None or modified_point is None:
                continue

            model_delta = modified_point - baseline_point
            model_direction = 1 if model_delta > 0 else -1

            correct = (model_direction == true_direction)

            results[f"{variable}_h{horizon}"] = {
                "correct": correct,
                "true_direction": true_direction,
                "model_direction": model_direction,
                "true_delta": true_delta,
                "model_delta": model_delta,
                "horizon": horizon,
                "variable": variable,
            }

    return results


def run_sign_and_slope_regression(
    true_responses: list[float],
    model_responses: list[float],
) -> dict[str, float]:
    """
    Regress model response on true response across parameter settings.

    model_response_i = alpha + beta * true_response_i + epsilon_i

    Args:
        true_responses: List of true (MC) response values, one per setting.
        model_responses: List of model response values, one per setting.

    Returns:
        Dict with keys: alpha, beta, beta_se, t_stat, p_value, r_squared,
        n_observations.
    """
    from scipy import stats

    true_arr = np.array(true_responses)
    model_arr = np.array(model_responses)

    # Need at least 3 points for a meaningful regression
    n = len(true_arr)
    if n < 3:
        return {
            "alpha": float("nan"),
            "beta": float("nan"),
            "beta_se": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "r_squared": float("nan"),
            "n_observations": n,
            "significant": False,
        }

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        true_arr, model_arr
    )

    return {
        "alpha": intercept,
        "beta": slope,
        "beta_se": std_err,
        "t_stat": slope / std_err if std_err > 0 else float("nan"),
        "p_value": p_value,
        "r_squared": r_value ** 2,
        "n_observations": n,
        "significant": bool(p_value < 0.05 and slope > 0),
    }


def run_sign_and_slope_bootstrap(
    observations: list[dict],
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run both i.i.d. and clustered bootstrap on the sign-and-slope regression.

    Each observation dict must have keys:
      - "true_delta": float (MC ground-truth response)
      - "model_delta": float (model forecast response)
      - "cluster_id": str (identifies the cluster, typically
        "{world_id}__{parameter_setting}" — all horizons and variables from the
        same parameter-setting within the same world share a cluster)

    The i.i.d. bootstrap resamples individual observations. The clustered
    bootstrap resamples at the cluster level: when a cluster is drawn, ALL of
    its observations are included. This preserves within-cluster dependence
    (multiple horizons and variables from the same model call / parameter
    setting).

    Both are reported side by side so the difference in CI width is visible.

    Args:
        observations: List of observation dicts, each with true_delta,
                      model_delta, and cluster_id.
        n_resamples: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
          - "ols": naive OLS regression result (from run_sign_and_slope_regression)
          - "bootstrap_iid": {beta_mean, beta_ci_low, beta_ci_high, beta_se,
                              significant, n_resamples}
          - "bootstrap_clustered": same keys, resampled at cluster level
          - "n_observations": total observation count
          - "n_clusters": number of unique clusters
    """
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(seed)

    true_deltas = [obs["true_delta"] for obs in observations]
    model_deltas = [obs["model_delta"] for obs in observations]

    # Naive OLS result for reference
    ols_result = run_sign_and_slope_regression(true_deltas, model_deltas)

    n_obs = len(observations)
    if n_obs < 3:
        empty_bootstrap = {
            "beta_mean": float("nan"),
            "beta_ci_low": float("nan"),
            "beta_ci_high": float("nan"),
            "beta_se": float("nan"),
            "significant": False,
            "n_resamples": n_resamples,
        }
        return {
            "ols": ols_result,
            "bootstrap_iid": empty_bootstrap,
            "bootstrap_clustered": empty_bootstrap,
            "n_observations": n_obs,
            "n_clusters": 0,
        }

    true_arr = np.array(true_deltas)
    model_arr = np.array(model_deltas)

    # --- i.i.d. bootstrap ---
    iid_betas = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n_obs, size=n_obs)
        sampled_true = true_arr[indices]
        sampled_model = model_arr[indices]
        # Skip degenerate resamples (all same x value)
        if np.std(sampled_true) < 1e-12:
            continue
        slope, _, _, _, _ = scipy_stats.linregress(sampled_true, sampled_model)
        iid_betas.append(slope)

    iid_betas = np.array(iid_betas)
    iid_result = _summarize_bootstrap_betas(iid_betas, n_resamples)

    # --- Clustered bootstrap ---
    # Group observations by cluster_id
    cluster_ids = list(set(obs["cluster_id"] for obs in observations))
    n_clusters = len(cluster_ids)

    # Build index arrays per cluster for fast resampling
    cluster_indices = {}
    for i, obs in enumerate(observations):
        cid = obs["cluster_id"]
        if cid not in cluster_indices:
            cluster_indices[cid] = []
        cluster_indices[cid].append(i)

    cluster_id_array = np.array(cluster_ids)

    clustered_betas = []
    for _ in range(n_resamples):
        # Resample clusters with replacement
        sampled_cluster_ids = rng.choice(cluster_id_array, size=n_clusters, replace=True)
        # Collect all observation indices from the sampled clusters
        sampled_indices = []
        for cid in sampled_cluster_ids:
            sampled_indices.extend(cluster_indices[cid])
        sampled_indices = np.array(sampled_indices)

        sampled_true = true_arr[sampled_indices]
        sampled_model = model_arr[sampled_indices]
        if np.std(sampled_true) < 1e-12:
            continue
        slope, _, _, _, _ = scipy_stats.linregress(sampled_true, sampled_model)
        clustered_betas.append(slope)

    clustered_betas = np.array(clustered_betas)
    clustered_result = _summarize_bootstrap_betas(clustered_betas, n_resamples)

    return {
        "ols": ols_result,
        "bootstrap_iid": iid_result,
        "bootstrap_clustered": clustered_result,
        "n_observations": n_obs,
        "n_clusters": n_clusters,
    }


def _summarize_bootstrap_betas(
    betas: np.ndarray,
    n_resamples: int,
) -> dict[str, Any]:
    """
    Summarize a bootstrap distribution of beta estimates.

    Returns mean, 95% CI (percentile method), SE, and significance
    (CI excludes zero AND mean is positive).
    """
    if len(betas) == 0:
        return {
            "beta_mean": float("nan"),
            "beta_ci_low": float("nan"),
            "beta_ci_high": float("nan"),
            "beta_se": float("nan"),
            "significant": False,
            "n_resamples": n_resamples,
        }

    beta_mean = float(np.mean(betas))
    beta_se = float(np.std(betas))
    ci_low = float(np.percentile(betas, 2.5))
    ci_high = float(np.percentile(betas, 97.5))
    # Significant if the 95% CI excludes zero and beta is positive
    significant = ci_low > 0 and beta_mean > 0

    return {
        "beta_mean": beta_mean,
        "beta_ci_low": ci_low,
        "beta_ci_high": ci_high,
        "beta_se": beta_se,
        "significant": significant,
        "n_resamples": n_resamples,
    }


def compute_told_infer_gap(
    told_results: dict[str, Any],
    infer_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute the TOLD-minus-INFER gap for directional accuracy and slope metrics.

    Args:
        told_results: Scoring results from the TOLD condition.
        infer_results: Scoring results from the INFER condition.

    Returns:
        Dict with gap metrics.
    """
    gap = {}

    # Directional accuracy gap
    if "directional_accuracy" in told_results and "directional_accuracy" in infer_results:
        told_correct = sum(
            1 for v in told_results["directional_accuracy"].values()
            if v.get("correct", False)
        )
        told_total = len(told_results["directional_accuracy"])

        infer_correct = sum(
            1 for v in infer_results["directional_accuracy"].values()
            if v.get("correct", False)
        )
        infer_total = len(infer_results["directional_accuracy"])

        told_rate = told_correct / told_total if told_total > 0 else 0
        infer_rate = infer_correct / infer_total if infer_total > 0 else 0

        gap["directional_accuracy_told"] = told_rate
        gap["directional_accuracy_infer"] = infer_rate
        gap["directional_accuracy_gap"] = told_rate - infer_rate

    # Slope gap (if regression results exist)
    if "regression" in told_results and "regression" in infer_results:
        gap["beta_told"] = told_results["regression"].get("beta", float("nan"))
        gap["beta_infer"] = infer_results["regression"].get("beta", float("nan"))
        gap["beta_gap"] = gap["beta_told"] - gap["beta_infer"]
        gap["significant_told"] = told_results["regression"].get("significant", False)
        gap["significant_infer"] = infer_results["regression"].get("significant", False)

    return gap
