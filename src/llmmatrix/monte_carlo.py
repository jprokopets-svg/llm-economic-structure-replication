"""
Monte Carlo forward simulation for counterfactual analysis.

Given a deterministic history and a shock scenario, this module branches
from the end of history and runs many forward paths with fresh random
shocks. The resulting distribution is the ground truth that LLM predictions
will be scored against.
"""

from typing import Callable

import numpy as np
import pandas as pd

from llmmatrix.simulator import Sim


def run_counterfactual(
    config_path: str,
    history_df: pd.DataFrame,
    shock_fn: Callable[[], dict[tuple[int, str], float]],
    n_paths: int = 1000,
    n_forward_periods: int = 12,
    base_seed: int = 1000,
    sim_class=None,
    variable_names=None,
) -> np.ndarray:
    """
    Run Monte Carlo forward simulation from the end of a historical trajectory.

    For each path:
      1. Start from the last state in history_df.
      2. Apply the counterfactual shock overrides from shock_fn.
      3. Simulate n_forward_periods quarters with fresh random shocks.

    Args:
        config_path: Path to YAML config file.
        history_df: DataFrame of historical trajectory (output of Sim.to_dataframe).
        shock_fn: Callable that returns a shock override dict.
        n_paths: Number of Monte Carlo paths to simulate.
        n_forward_periods: Number of quarters to simulate forward.
        base_seed: Starting seed; each path uses base_seed + path_index.
        sim_class: Simulator class to use. Defaults to Sim (world 1).
                   Pass ClosedSim for world 2.
        variable_names: Variables to extract from each state dict.
                        Defaults to ["y", "pi", "r", "e", "u"].

    Returns:
        3D numpy array of shape (n_paths, n_forward_periods, n_variables).
    """
    if variable_names is None:
        variable_names = ["y", "pi", "r", "e", "u"]

    # Extract the final state from history to use as the branching point
    last_row = history_df.iloc[-1]
    last_period = int(last_row["period"])

    # Build the initial state from all available columns in history_df.
    # This handles worlds with different state variables (w, u_natural, etc.)
    branch_state = {"period": last_period}
    for col in history_df.columns:
        if col == "period":
            continue
        branch_state[col] = float(last_row[col])

    # Add e_prev if present (needed by world 1 / world 3 Phillips curve)
    if "e" in history_df.columns and len(history_df) >= 2:
        second_to_last_row = history_df.iloc[-2]
        branch_state["e_prev"] = float(second_to_last_row["e"])

    # Zero out epsilon fields that aren't in the DataFrame
    for eps_field in ["eps_y", "eps_pi", "eps_e", "eps_w",
                      "eps_productivity", "eps_u_natural"]:
        if eps_field not in branch_state:
            branch_state[eps_field] = 0.0

    # Get the shock overrides
    shock_overrides = shock_fn()

    # Use the provided sim class, or default to world 1 Sim
    if sim_class is None:
        sim_class = Sim

    # Allocate output array
    results = np.zeros((n_paths, n_forward_periods, len(variable_names)))

    for path_index in range(n_paths):
        # Each path gets its own seed for reproducibility
        sim = sim_class(config_path, seed=base_seed + path_index)

        # Run forward from the branch point
        trajectory = sim.run(
            n_periods=n_forward_periods,
            initial_state=branch_state,
            shock_overrides=shock_overrides,
        )

        # Skip the initial state (period 0 of the forward sim = end of history)
        # Take only the n_forward_periods new states
        for t_index in range(n_forward_periods):
            forward_state = trajectory[t_index + 1]
            for v_index, var_name in enumerate(variable_names):
                results[path_index, t_index, v_index] = forward_state[var_name]

    return results


def summarize_paths(
    paths: np.ndarray,
    start_period: int,
) -> pd.DataFrame:
    """
    Summarize Monte Carlo paths into mean and percentile forecasts.

    Args:
        paths: 3D array of shape (n_paths, n_periods, 5) from run_counterfactual.
        start_period: The first period number of the forward simulation.

    Returns:
        DataFrame with columns: period, variable, mean, std, p10, p25, p50, p75, p90.
    """
    variable_names = ["y", "pi", "r", "e", "u"]
    n_periods = paths.shape[1]

    rows = []
    for t_index in range(n_periods):
        period = start_period + t_index + 1
        for v_index, var_name in enumerate(variable_names):
            values = paths[:, t_index, v_index]
            rows.append({
                "period": period,
                "variable": var_name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p10": float(np.percentile(values, 10)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
            })

    return pd.DataFrame(rows)
