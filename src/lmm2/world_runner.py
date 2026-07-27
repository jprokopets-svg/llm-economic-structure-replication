"""
World runner utilities for generating simulation histories.

Provides helper functions used by tests to instantiate simulators
and generate synthetic histories from world configurations.
"""

import importlib

import pandas as pd


def get_simulator_class(simulator_path: str):
    """
    Resolve a simulator module path to its simulator class.

    Args:
        simulator_path: Dotted path like 'llmmatrix.simulator.Sim'
            or 'lmm2.simulator.Sim'.

    Returns:
        The simulator class.
    """
    parts = simulator_path.rsplit(".", 1)
    if len(parts) == 2:
        mod_path, class_name = parts
    else:
        mod_path = simulator_path
        class_name = "Sim"

    mod = importlib.import_module(mod_path)
    return getattr(mod, class_name)


def generate_history(config_path, sim_class, n_periods=60, seed=42):
    """
    Generate a synthetic history from a simulator configuration.

    Args:
        config_path: Path to the YAML config file.
        sim_class: Simulator class to instantiate.
        n_periods: Number of periods to simulate.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with columns: period, y, pi, r, e, u, ...
    """
    sim = sim_class(str(config_path), seed=seed)
    initial_state = sim.get_initial_state()
    history = sim.run(n_periods, initial_state)
    df = sim.to_dataframe(history)
    return df
