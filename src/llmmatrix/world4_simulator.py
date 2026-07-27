"""
Core simulator for the world 4 labor-market-with-hysteresis NK model.

Closed-economy model with six endogenous variables:
  - y: output gap (IS curve)
  - pi: price inflation (price Phillips curve, wage-driven)
  - r: real interest rate (Taylor rule)
  - u: unemployment rate (Okun's law with time-varying NAIRU)
  - u_natural: time-varying NAIRU (hysteresis dynamics)
  - w: wage inflation (wage Phillips curve)

One exogenous state:
  - productivity: deviation from trend (persistent, can be shocked)

Key features:
  - Wage Phillips curve: wages respond to unemployment gap (u - u_natural)
    and productivity (Layard, Nickell & Jackman 1991)
  - Price Phillips curve: prices follow wages (wage passthrough)
  - NAIRU hysteresis: prolonged unemployment raises NAIRU
    (Blanchard & Summers 1986; Ball 2009)
  - Anchored drift: NAIRU pulled toward long-run anchor to prevent
    unbounded drift

State dicts include dummy e=0 fields for pipeline compatibility with
monte_carlo.py (which expects [y, pi, r, e, u] in the 5-column array).
"""

from typing import Optional

import numpy as np
import pandas as pd
import yaml


class HysteresisSim:
    """
    Simulator for the world 4 labor-market-with-hysteresis NK model.

    Same run/step/to_dataframe interface as Sim (world 1).
    """

    def __init__(self, config_path: str, seed: int = 42) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Targets
        targets = self.config["targets"]
        self.r_star = targets["r_star"]
        self.pi_star = targets["pi_star"]
        self.u_natural_initial = targets["u_natural_initial"]
        self.w_star = targets["w_star"]

        # IS curve
        is_params = self.config["is_curve"]
        self.output_persistence = is_params["output_persistence"]
        self.interest_sensitivity = is_params["interest_sensitivity"]

        # Price Phillips curve (wage-based)
        pp = self.config["price_phillips"]
        self.inflation_persistence = pp["inflation_persistence"]
        self.wage_passthrough = pp["wage_passthrough"]
        self.output_slope = pp["output_slope"]

        # Wage Phillips curve
        wp = self.config["wage_phillips"]
        self.wage_persistence = wp["wage_persistence"]
        self.unemployment_gap_slope = wp["unemployment_gap_slope"]
        self.productivity_passthrough = wp["productivity_passthrough"]

        # Productivity dynamics
        prod_params = self.config.get("productivity", {})
        self.productivity_persistence = prod_params.get("persistence", 1.0)

        # NAIRU dynamics
        nd = self.config["nairu_dynamics"]
        self.hysteresis_coefficient = nd["hysteresis_coefficient"]
        self.u_natural_anchor = nd["u_natural_anchor"]
        self.anchor_weight = nd["anchor_weight"]

        # Okun
        self.okun_coefficient = self.config["okun"]["coefficient"]

        # Taylor rule
        tr = self.config["taylor_rule"]
        self.smoothing_rho = tr["smoothing_rho"]
        self.inflation_coefficient = tr["inflation_coefficient"]
        self.output_coefficient = tr["output_coefficient"]

        # Shocks
        shocks = self.config["shocks"]
        self.sigma_y = shocks["sigma_y"]
        self.sigma_pi = shocks["sigma_pi"]
        self.sigma_w = shocks["sigma_w"]
        self.sigma_productivity = shocks["sigma_productivity"]
        self.sigma_u_natural = shocks["sigma_u_natural"]

        # Dummy fields for pipeline compatibility
        self.e_star = 0.0
        self.r_world = 0.0
        self.u_star = self.u_natural_initial
        self.exchange_sensitivity = 0.0
        self.exchange_passthrough = 0.0
        self.interest_differential = 0.0
        self.sigma_e = 0.0

        # RNG
        self.rng = np.random.default_rng(seed)

    def get_initial_state(self) -> dict:
        ic = self.config["initial_conditions"]
        return {
            "period": 0,
            "y": ic["y_0"],
            "pi": ic["pi_0"],
            "r": ic["r_0"],
            "u": ic["u_0"],
            "u_natural": ic["u_natural_0"],
            "w": ic["w_0"],
            "productivity": ic["productivity_0"],
            "e": 0.0,          # dummy
            "e_prev": 0.0,     # dummy
            "eps_y": 0.0,
            "eps_pi": 0.0,
            "eps_w": 0.0,
            "eps_productivity": 0.0,
            "eps_u_natural": 0.0,
            "eps_e": 0.0,      # dummy
        }

    def step(
        self,
        state: dict,
        shock_overrides: Optional[dict] = None,
    ) -> dict:
        """
        Advance the economy by one quarter.

        Order of computation:
          1. Draw shocks
          2. NAIRU dynamics (hysteresis + anchor)
          3. Productivity evolution
          4. IS curve: y_t
          5. Okun's law: u_t (depends on time-varying NAIRU)
          6. Wage Phillips curve: w_t (depends on u - u_natural gap)
          7. Price Phillips curve: pi_t (depends on w_t)
          8. Taylor rule: r_t

        Args:
            state: Current state dict.
            shock_overrides: Optional (period, variable) -> value mapping.
                Supports: eps_y, eps_pi, eps_w, eps_productivity,
                eps_u_natural, r_override, prolonged_demand (multi-period).
        """
        new_period = state["period"] + 1
        overrides = shock_overrides or {}

        # Step 1: Draw shocks
        eps_y = overrides.get((new_period, "eps_y"), self.rng.normal(0, self.sigma_y))
        eps_pi = overrides.get((new_period, "eps_pi"), self.rng.normal(0, self.sigma_pi))
        eps_w = overrides.get((new_period, "eps_w"), self.rng.normal(0, self.sigma_w))
        eps_prod = overrides.get(
            (new_period, "eps_productivity"),
            self.rng.normal(0, self.sigma_productivity),
        )
        eps_u_nat = overrides.get(
            (new_period, "eps_u_natural"),
            self.rng.normal(0, self.sigma_u_natural),
        )

        # Step 2: NAIRU dynamics — Blanchard-Summers hysteresis + anchor
        # u_natural_t = (1 - kappa - alpha)*u_natural_{t-1}
        #               + kappa*u_{t-1}
        #               + alpha*u_natural_anchor
        #               + eps_u_natural
        kappa = self.hysteresis_coefficient
        alpha = self.anchor_weight
        u_natural_t = (
            (1.0 - kappa - alpha) * state["u_natural"]
            + kappa * state["u"]
            + alpha * self.u_natural_anchor
            + eps_u_nat
        )

        # Step 3: Productivity evolution (mean-reverting AR(1) process)
        productivity_t = self.productivity_persistence * state["productivity"] + eps_prod

        # Step 4: IS curve (closed economy)
        y_t = (
            self.output_persistence * state["y"]
            - self.interest_sensitivity * (state["r"] - self.r_star)
            + eps_y
        )

        # Step 5: Okun's law with time-varying NAIRU
        u_t = u_natural_t - self.okun_coefficient * y_t

        # Step 6: Wage Phillips curve (Layard-Nickell-Jackman)
        # Wage inflation responds to unemployment gap and productivity
        unemployment_gap = state["u"] - state["u_natural"]
        w_t = (
            self.w_star
            + self.wage_persistence * (state["w"] - self.w_star)
            - self.unemployment_gap_slope * unemployment_gap
            + self.productivity_passthrough * state["productivity"]
            + eps_w
        )

        # Step 7: Price Phillips curve (wage-driven)
        pi_t = (
            self.pi_star
            + self.inflation_persistence * (state["pi"] - self.pi_star)
            + self.wage_passthrough * (w_t - self.w_star)
            + self.output_slope * state["y"]
            + eps_pi
        )

        # Step 8: Taylor rule
        r_target = (
            self.r_star
            + self.inflation_coefficient * (pi_t - self.pi_star)
            + self.output_coefficient * y_t
        )
        rho = self.smoothing_rho
        r_t = rho * state["r"] + (1.0 - rho) * r_target

        # Apply monetary override
        r_override = overrides.get((new_period, "r_override"), 0.0)
        r_t = r_t + r_override

        return {
            "period": new_period,
            "y": y_t,
            "pi": pi_t,
            "r": r_t,
            "u": u_t,
            "u_natural": u_natural_t,
            "w": w_t,
            "productivity": productivity_t,
            "e": 0.0,          # dummy
            "e_prev": 0.0,     # dummy
            "eps_y": eps_y,
            "eps_pi": eps_pi,
            "eps_w": eps_w,
            "eps_productivity": eps_prod,
            "eps_u_natural": eps_u_nat,
            "eps_e": 0.0,      # dummy
        }

    def run(
        self,
        n_periods: int,
        initial_state: Optional[dict] = None,
        shock_overrides: Optional[dict] = None,
    ) -> list[dict]:
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
        Convert trajectory to DataFrame.

        Includes standard 5-column layout (y, pi, r, e=0, u) for MC
        pipeline compatibility, plus world-4-specific columns
        (u_natural, w, productivity).
        """
        output_columns = [
            "period", "y", "pi", "r", "e", "u",
            "u_natural", "w", "productivity",
            "eps_y", "eps_pi", "eps_w", "eps_productivity", "eps_u_natural", "eps_e",
        ]
        rows = []
        for state in trajectory:
            row = {col: state[col] for col in output_columns}
            rows.append(row)
        return pd.DataFrame(rows)
