"""
Core simulator for the world 3 emerging-market high-inflation NK model.

Extends the world 1 open-economy model with:
  - Augmented Taylor rule including exchange rate response (Castro et al. 2017)
  - Higher volatility (BIS Hofmann & Bogdanova 2012 emerging market estimates)
  - Steeper Phillips curve and higher FX passthrough
  - Support for regime-shift shocks (pi_star changes) and external-pressure
    shocks (r_world changes)

Five-equation backward-looking system:
  - IS curve: Ball (1999) with higher exchange rate sensitivity
  - Phillips curve: Ball (1999) with higher FX passthrough
  - Augmented Taylor rule: Castro et al. (2017) with FX term + smoothing
  - UIP condition: higher interest differential coefficient
  - Okun's law: unchanged

Shock overrides support two additional types beyond world 1:
  - ("regime_shift", period): changes pi_star from that period onward
  - ("external_pressure", period): changes r_world from that period onward
"""

from typing import Optional

import numpy as np
import pandas as pd
import yaml


class EmergingSim:
    """
    Simulator for the world 3 emerging-market NK model.

    Same interface as world 1 Sim: step(), run(), to_dataframe().
    Adds support for regime-shift and external-pressure shocks.
    """

    def __init__(self, config_path: str, seed: int = 42) -> None:
        """
        Load config and initialize the random number generator.

        Args:
            config_path: Path to YAML config file (world3_emerging_market.yaml).
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

        # Unpack IS curve parameters
        is_params = self.config["is_curve"]
        self.output_persistence = is_params["output_persistence"]
        self.interest_sensitivity = is_params["interest_sensitivity"]
        self.exchange_sensitivity = is_params["exchange_sensitivity"]

        # Unpack Phillips curve parameters
        pc_params = self.config["phillips_curve"]
        self.inflation_persistence = pc_params["inflation_persistence"]
        self.output_slope = pc_params["output_slope"]
        self.exchange_passthrough = pc_params["exchange_passthrough"]

        # Unpack Taylor rule parameters (augmented with FX)
        tr_params = self.config["taylor_rule"]
        self.inflation_coefficient = tr_params["inflation_coefficient"]
        self.output_coefficient = tr_params["output_coefficient"]
        self.smoothing_rho = tr_params["smoothing_rho"]
        self.exchange_coefficient = tr_params["exchange_coefficient"]

        # Unpack UIP parameter
        self.interest_differential = self.config["uip"]["interest_differential"]

        # Unpack Okun parameter
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
            State dict with all standard keys.
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

        Equation order:
          1. Draw shocks (or apply overrides)
          2. Apply regime shifts (pi_star, r_world changes)
          3. IS curve: y_t from y_{t-1}, r_{t-1}, e_{t-1}
          4. Phillips curve: pi_t from pi_{t-1}, y_{t-1}, delta-e
          5. UIP: e_t from previous-period dynamics + eps_e (temporary, for Taylor input)
          6. Augmented Taylor rule: r_t from pi_t, y_t, e_t (contemporaneous)
          7. UIP: e_t recomputed from final r_t
          8. Okun: u_t from y_t

        Note on the FX-augmented Taylor rule:
          The exchange rate enters the Taylor rule contemporaneously (Castro et al.
          2017). This creates a simultaneity between r_t and e_t (via UIP). We
          resolve this by substitution: e_t = e_star + beta*(r_t - r_world) + eps_e,
          and r_t depends on e_t. Solving the system algebraically gives a closed-form
          for r_t that avoids iteration.

        Args:
            state: Current state dict (period t-1 values).
            shock_overrides: Optional dict mapping (period, variable) -> value.
                             Supports standard shocks plus:
                             - (period, "pi_star_shift"): change in inflation target
                             - (period, "r_world_shift"): change in foreign rate

        Returns:
            New state dict for period t.
        """
        new_period = state["period"] + 1
        overrides = shock_overrides or {}

        # Step 1: Draw shocks or use overrides
        eps_y = overrides.get((new_period, "eps_y"), self.rng.normal(0, self.sigma_y))
        eps_pi = overrides.get((new_period, "eps_pi"), self.rng.normal(0, self.sigma_pi))
        eps_e = overrides.get((new_period, "eps_e"), self.rng.normal(0, self.sigma_e))

        # Step 2: Apply regime shifts (cumulative — check all periods up to now)
        # These modify the targets going forward, not just for one period.
        current_pi_star = self.pi_star
        current_r_world = self.r_world
        for t in range(1, new_period + 1):
            if (t, "pi_star_shift") in overrides:
                current_pi_star = current_pi_star + overrides[(t, "pi_star_shift")]
            if (t, "r_world_shift") in overrides:
                current_r_world = current_r_world + overrides[(t, "r_world_shift")]

        # Step 3: IS curve — Ball (1999) with higher FX sensitivity
        y_t = (
            self.output_persistence * state["y"]
            - self.interest_sensitivity * (state["r"] - self.r_star)
            - self.exchange_sensitivity * (state["e"] - self.e_star)
            + eps_y
        )

        # Step 4: Phillips curve — steeper, higher FX passthrough
        delta_e_prev = state["e"] - state["e_prev"]
        pi_t = (
            current_pi_star
            + self.inflation_persistence * (state["pi"] - current_pi_star)
            + self.output_slope * state["y"]
            - self.exchange_passthrough * delta_e_prev
            + eps_pi
        )

        # Steps 5-7: Solve the simultaneous r_t / e_t system.
        #
        # Taylor rule (augmented):
        #   r_target = r_star + alpha_pi*(pi_t - pi_star) + alpha_y*y_t + alpha_e*(e_t - e_star)
        #   r_t = rho*r_{t-1} + (1-rho)*r_target + r_override
        #
        # UIP (deviation form — ensures SS consistency when r_star != r_world):
        #   e_t = e_star + beta * ((r_t - r_star) - (r_world_t - r_world_base)) + eps_e
        #
        # At SS: r=r_star, r_world=r_world_base → e = e_star. This avoids the
        # permanent offset that occurs with the world 1 UIP (e = e_star + beta*(r - r_world))
        # when r_star != r_world.
        #
        # Substituting UIP into Taylor rule and solving for r_t:
        #   e_t - e_star = beta*(r_t - r_star) - beta*(r_world_t - r_world_base) + eps_e
        #   Let delta_rw = current_r_world - self.r_world  (external pressure shift)
        #
        #   r_target = r_star + alpha_pi*(pi_t - pi_star) + alpha_y*y_t
        #              + alpha_e*(beta*(r_t - r_star) - beta*delta_rw + eps_e)
        #   r_t = rho*r_{t-1} + w*r_target + r_override
        #   r_t = rho*r_{t-1} + w*[r_star + alpha_pi*(pi_t-pi_star) + alpha_y*y_t
        #          + alpha_e*(beta*(r_t-r_star) - beta*delta_rw + eps_e)] + r_override
        #   r_t*(1 - w*alpha_e*beta) = rho*r_{t-1} + w*[r_star*(1-alpha_e*beta)
        #          + alpha_pi*(pi_t-pi_star) + alpha_y*y_t
        #          + alpha_e*(eps_e - beta*delta_rw)] + r_override

        rho = self.smoothing_rho
        w = 1.0 - rho
        alpha_pi = self.inflation_coefficient
        alpha_y = self.output_coefficient
        alpha_e = self.exchange_coefficient
        beta = self.interest_differential

        r_override = overrides.get((new_period, "r_override"), 0.0)

        # External pressure shift (deviation from config r_world)
        delta_r_world = current_r_world - self.r_world

        denominator = 1.0 - w * alpha_e * beta
        numerator = (
            rho * state["r"]
            + w * (
                self.r_star * (1.0 - alpha_e * beta)
                + alpha_pi * (pi_t - current_pi_star)
                + alpha_y * y_t
                + alpha_e * (eps_e - beta * delta_r_world)
            )
            + r_override
        )
        r_t = numerator / denominator

        # Step 7: Compute e_t from the solved r_t via UIP (deviation form)
        e_t = (
            self.e_star
            + beta * (r_t - self.r_star)
            - beta * delta_r_world
            + eps_e
        )

        # Step 8: Okun's law
        u_t = self.u_star - self.okun_coefficient * y_t

        return {
            "period": new_period,
            "y": y_t,
            "pi": pi_t,
            "r": r_t,
            "e": e_t,
            "e_prev": state["e"],
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
