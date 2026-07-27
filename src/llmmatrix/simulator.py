"""
Core simulator for the Ball (1999) open-economy New Keynesian model.

Implements a five-equation backward-looking system:
  - IS curve (output gap): Ball (1999)
  - Phillips curve (inflation): Ball (1999) with exchange rate passthrough
  - Taylor rule with smoothing (interest rate): Taylor (1993), Dotsey & Sill (2015)
  - UIP condition (exchange rate): standard open-economy calibration
  - Okun's law (unemployment): Okun (1962)

All parameters are loaded from a YAML config file. No magic numbers in code.
"""

from typing import Optional

import numpy as np
import pandas as pd
import yaml


class Sim:
    """
    Simulator for the Ball (1999) open-economy NK model.

    Each call to step() advances the economy by one quarter.
    run() produces a full trajectory as a list of state dicts.
    """

    def __init__(self, config_path: str, seed: int = 42) -> None:
        """
        Load config and initialize the random number generator.

        Args:
            config_path: Path to YAML config file with all model parameters.
            seed: Explicit seed for reproducibility.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Unpack targets
        targets = self.config["targets"]
        self.r_star = targets["r_star"]
        self.pi_star = targets["pi_star"]
        self.e_star = targets["e_star"]
        self.u_star = targets["u_star"]
        self.r_world = targets["r_world"]

        # Unpack IS curve parameters (Ball 1999)
        is_params = self.config["is_curve"]
        self.output_persistence = is_params["output_persistence"]
        self.interest_sensitivity = is_params["interest_sensitivity"]
        self.exchange_sensitivity = is_params["exchange_sensitivity"]

        # Unpack Phillips curve parameters (Ball 1999, McCarthy 1999)
        pc_params = self.config["phillips_curve"]
        self.inflation_persistence = pc_params["inflation_persistence"]
        self.output_slope = pc_params["output_slope"]
        self.exchange_passthrough = pc_params["exchange_passthrough"]

        # Unpack Taylor rule parameters (Taylor 1993, Dotsey & Sill 2015)
        tr_params = self.config["taylor_rule"]
        self.inflation_coefficient = tr_params["inflation_coefficient"]
        self.output_coefficient = tr_params["output_coefficient"]
        self.smoothing_rho = tr_params["smoothing_rho"]

        # Unpack UIP parameter
        self.interest_differential = self.config["uip"]["interest_differential"]

        # Unpack Okun parameter (Okun 1962)
        self.okun_coefficient = self.config["okun"]["coefficient"]

        # Unpack shock standard deviations
        shocks = self.config["shocks"]
        self.sigma_y = shocks["sigma_y"]
        self.sigma_pi = shocks["sigma_pi"]
        self.sigma_e = shocks["sigma_e"]

        # Initialize RNG with explicit seed
        self.rng = np.random.default_rng(seed)

    def get_initial_state(self) -> dict:
        """
        Build the initial state dict from config initial conditions.

        Returns:
            State dict with keys: period, y, pi, r, e, e_prev, u, eps_y, eps_pi, eps_e.
        """
        ic = self.config["initial_conditions"]
        return {
            "period": 0,
            "y": ic["y_0"],
            "pi": ic["pi_0"],
            "r": ic["r_0"],
            "e": ic["e_0"],
            "e_prev": ic["e_minus1"],
            "u": self.u_star - self.okun_coefficient * ic["y_0"],
            "eps_y": 0.0,
            "eps_pi": 0.0,
            "eps_e": 0.0,
        }

    def step(
        self,
        state: dict,
        shock_overrides: Optional[dict] = None,
    ) -> dict:
        """
        Advance the economy by one quarter.

        Order of computation matters:
          1. Draw shocks (or apply overrides)
          2. IS curve: y_t from y_{t-1}, r_{t-1}, e_{t-1}
          3. Phillips curve: pi_t from pi_{t-1}, y_{t-1}, delta-e
          4. Taylor rule: r_t from contemporaneous pi_t, y_t
          5. UIP: e_t from r_t
          6. Okun: u_t from y_t

        Args:
            state: Current state dict (period t-1 values).
            shock_overrides: Optional dict mapping (period, variable) -> value.
                             Variables are 'eps_y', 'eps_pi', 'eps_e', or 'r_override'.

        Returns:
            New state dict for period t.
        """
        new_period = state["period"] + 1
        overrides = shock_overrides or {}

        # Step 1: Draw shocks or use overrides
        eps_y = overrides.get((new_period, "eps_y"), self.rng.normal(0, self.sigma_y))
        eps_pi = overrides.get((new_period, "eps_pi"), self.rng.normal(0, self.sigma_pi))
        eps_e = overrides.get((new_period, "eps_e"), self.rng.normal(0, self.sigma_e))

        # Step 2: IS curve — Ball (1999) equation 1
        # y_t = persistence * y_{t-1} - interest_sens * (r_{t-1} - r_star)
        #        - exchange_sens * (e_{t-1} - e_star) + eps_y
        y_t = (
            self.output_persistence * state["y"]
            - self.interest_sensitivity * (state["r"] - self.r_star)
            - self.exchange_sensitivity * (state["e"] - self.e_star)
            + eps_y
        )

        # Step 3: Phillips curve — Ball (1999), deviation-from-target form
        # With persistence < 1 (Gali 2008; Stock & Watson 1999), inflation
        # must be formulated in deviations from pi_star to maintain steady state:
        #   (pi_t - pi_star) = persistence * (pi_{t-1} - pi_star)
        #                      + slope * y_{t-1}
        #                      - passthrough * (e_{t-1} - e_{t-2}) + eps_pi
        delta_e_prev = state["e"] - state["e_prev"]
        pi_t = (
            self.pi_star
            + self.inflation_persistence * (state["pi"] - self.pi_star)
            + self.output_slope * state["y"]
            - self.exchange_passthrough * delta_e_prev
            + eps_pi
        )

        # Step 4: Taylor rule with interest-rate smoothing
        # Dotsey & Sill (2015); FRB/US model calibration.
        # r_target = r_star + inflation_coeff * (pi_t - pi_star) + output_coeff * y_t
        # r_t = rho * r_{t-1} + (1 - rho) * r_target
        r_target = (
            self.r_star
            + self.inflation_coefficient * (pi_t - self.pi_star)
            + self.output_coefficient * y_t
        )
        rho = self.smoothing_rho
        r_t = rho * state["r"] + (1 - rho) * r_target

        # Apply monetary policy override if specified
        # r_override adds to the Taylor-rule rate (a discretionary tightening/loosening)
        r_override = overrides.get((new_period, "r_override"), 0.0)
        r_t = r_t + r_override

        # Step 5: UIP — exchange rate responds to interest differential
        # e_t = e_star + coeff * (r_t - r_world) + eps_e
        e_t = (
            self.e_star
            + self.interest_differential * (r_t - self.r_world)
            + eps_e
        )

        # Step 6: Okun's law — Okun (1962)
        # u_t = u_star - coeff * y_t
        u_t = self.u_star - self.okun_coefficient * y_t

        return {
            "period": new_period,
            "y": y_t,
            "pi": pi_t,
            "r": r_t,
            "e": e_t,
            "e_prev": state["e"],  # carry forward for next Phillips curve delta-e
            "u": u_t,
            "eps_y": eps_y,
            "eps_pi": eps_pi,
            "eps_e": eps_e,
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
            shock_overrides: Optional dict mapping (period, variable) -> shock value
                             for counterfactual injection.

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

        df = pd.DataFrame(rows)
        return df
