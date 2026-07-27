"""
Core simulator for the world 2 closed-economy New Keynesian model.

Implements a four-equation backward-looking system:
  - IS curve (output gap): Ball (1999) without exchange rate channel
  - Phillips curve (inflation): Ball (1999) without exchange rate passthrough
  - Taylor rule with smoothing (interest rate): Taylor (1993), Dotsey & Sill (2015)
  - Okun's law (unemployment): Okun (1962)

This is a closed-economy variant of the world 1 open-economy model.
Exchange rate (e) is absent — no UIP condition, no passthrough.

State dicts include e=0.0 and e_prev=0.0 as dummy fields for pipeline
compatibility with monte_carlo.py and question_templating.py, which
expect a 5-variable [y, pi, r, e, u] layout.
"""

from typing import Optional

import numpy as np
import pandas as pd
import yaml


class ClosedSim:
    """
    Simulator for the world 2 closed-economy NK model.

    Same interface as world 1 Sim: step(), run(), to_dataframe().
    """

    def __init__(self, config_path: str, seed: int = 42) -> None:
        """
        Load config and initialize the random number generator.

        Args:
            config_path: Path to YAML config file (world2_closed_economy.yaml).
            seed: Explicit seed for reproducibility.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Unpack targets
        targets = self.config["targets"]
        self.r_star = targets["r_star"]
        self.pi_star = targets["pi_star"]
        self.u_star = targets["u_star"]

        # Unpack IS curve parameters (Ball 1999, closed-economy version)
        is_params = self.config["is_curve"]
        self.output_persistence = is_params["output_persistence"]
        self.interest_sensitivity = is_params["interest_sensitivity"]

        # Unpack Phillips curve parameters (Ball 1999, no exchange rate terms)
        pc_params = self.config["phillips_curve"]
        self.inflation_persistence = pc_params["inflation_persistence"]
        self.output_slope = pc_params["output_slope"]

        # Unpack Taylor rule parameters
        tr_params = self.config["taylor_rule"]
        self.inflation_coefficient = tr_params["inflation_coefficient"]
        self.output_coefficient = tr_params["output_coefficient"]
        self.smoothing_rho = tr_params["smoothing_rho"]

        # Unpack Okun parameter
        self.okun_coefficient = self.config["okun"]["coefficient"]

        # Unpack shock standard deviations
        shocks = self.config["shocks"]
        self.sigma_y = shocks["sigma_y"]
        self.sigma_pi = shocks["sigma_pi"]

        # Dummy fields for pipeline compatibility
        self.e_star = 0.0
        self.r_world = 0.0
        self.exchange_sensitivity = 0.0
        self.exchange_passthrough = 0.0
        self.interest_differential = 0.0
        self.sigma_e = 0.0

        # Initialize RNG with explicit seed
        self.rng = np.random.default_rng(seed)

    def get_initial_state(self) -> dict:
        """
        Build the initial state dict from config initial conditions.

        Returns:
            State dict with all standard keys (e and e_prev are dummy zeros).
        """
        ic = self.config["initial_conditions"]
        return {
            "period": 0,
            "y": ic["y_0"],
            "pi": ic["pi_0"],
            "r": ic["r_0"],
            "e": 0.0,        # dummy for pipeline compat
            "e_prev": 0.0,   # dummy for pipeline compat
            "u": self.u_star - self.okun_coefficient * ic["y_0"],
            "eps_y": 0.0,
            "eps_pi": 0.0,
            "eps_e": 0.0,    # dummy for pipeline compat
        }

    def step(
        self,
        state: dict,
        shock_overrides: Optional[dict] = None,
    ) -> dict:
        """
        Advance the economy by one quarter.

        Order of computation:
          1. Draw shocks (or apply overrides)
          2. IS curve: y_t from y_{t-1}, r_{t-1}  (no exchange rate term)
          3. Phillips curve: pi_t from pi_{t-1}, y_{t-1}  (no delta-e term)
          4. Taylor rule: r_t from contemporaneous pi_t, y_t
          5. Okun: u_t from y_t

        Args:
            state: Current state dict (period t-1 values).
            shock_overrides: Optional dict mapping (period, variable) -> value.

        Returns:
            New state dict for period t.
        """
        new_period = state["period"] + 1
        overrides = shock_overrides or {}

        # Step 1: Draw shocks or use overrides
        eps_y = overrides.get((new_period, "eps_y"), self.rng.normal(0, self.sigma_y))
        eps_pi = overrides.get((new_period, "eps_pi"), self.rng.normal(0, self.sigma_pi))

        # Step 2: IS curve — Ball (1999), closed-economy version
        # y_t = persistence * y_{t-1} - interest_sens * (r_{t-1} - r_star) + eps_y
        y_t = (
            self.output_persistence * state["y"]
            - self.interest_sensitivity * (state["r"] - self.r_star)
            + eps_y
        )

        # Step 3: Phillips curve — Ball (1999), no exchange rate passthrough
        # (pi_t - pi_star) = persistence * (pi_{t-1} - pi_star) + slope * y_{t-1} + eps_pi
        pi_t = (
            self.pi_star
            + self.inflation_persistence * (state["pi"] - self.pi_star)
            + self.output_slope * state["y"]
            + eps_pi
        )

        # Step 4: Taylor rule with interest-rate smoothing
        r_target = (
            self.r_star
            + self.inflation_coefficient * (pi_t - self.pi_star)
            + self.output_coefficient * y_t
        )
        rho = self.smoothing_rho
        r_t = rho * state["r"] + (1 - rho) * r_target

        # Apply monetary policy override if specified
        r_override = overrides.get((new_period, "r_override"), 0.0)
        r_t = r_t + r_override

        # Step 5: Okun's law
        u_t = self.u_star - self.okun_coefficient * y_t

        return {
            "period": new_period,
            "y": y_t,
            "pi": pi_t,
            "r": r_t,
            "e": 0.0,          # no exchange rate in closed economy
            "e_prev": 0.0,     # no exchange rate in closed economy
            "u": u_t,
            "eps_y": eps_y,
            "eps_pi": eps_pi,
            "eps_e": 0.0,      # no exchange rate shock
        }

    def run(
        self,
        n_periods: int,
        initial_state: Optional[dict] = None,
        shock_overrides: Optional[dict] = None,
    ) -> list[dict]:
        """
        Produce a trajectory of n_periods quarters.

        Args:
            n_periods: Number of quarters to simulate.
            initial_state: Starting state. If None, uses config initial conditions.
            shock_overrides: Optional dict mapping (period, variable) -> shock value.

        Returns:
            List of state dicts, one per period (including the initial state).
        """
        if initial_state is None:
            state = self.get_initial_state()
        else:
            state = initial_state.copy()

        trajectory = [state]

        for _ in range(n_periods):
            state = self.step(state, shock_overrides=shock_overrides)
            trajectory.append(state)

        return trajectory

    def to_dataframe(self, trajectory: list[dict]) -> pd.DataFrame:
        """
        Convert a trajectory (list of state dicts) to a tidy DataFrame.

        Includes the dummy 'e' column for pipeline compatibility.

        Args:
            trajectory: Output from run().

        Returns:
            DataFrame with columns: period, y, pi, r, e, u, eps_y, eps_pi, eps_e.
        """
        output_columns = ["period", "y", "pi", "r", "e", "u", "eps_y", "eps_pi", "eps_e"]
        rows = []
        for state in trajectory:
            row = {col: state[col] for col in output_columns}
            rows.append(row)

        return pd.DataFrame(rows)
