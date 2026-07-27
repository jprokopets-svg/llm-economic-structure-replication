"""
Counterfactual shock scenarios for the LLMMatrix simulator.

Each function returns a dict mapping (period, variable) -> shock_value,
which can be passed directly to Sim.run(shock_overrides=...).

Four scenarios for the prototype:
  1. Monetary tightening (discretionary rate hike above Taylor rule)
  2. Demand shock (positive eps_y)
  3. Cost-push shock (positive eps_pi)
  4. Exchange rate shock (positive eps_e = depreciation pressure)
"""


def monetary_tightening_shock(
    magnitude: float = 2.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Discretionary monetary tightening: the central bank raises the real rate
    by `magnitude` percentage points above what the Taylor rule prescribes.

    This overrides r_t at the shock period. The Taylor rule still operates
    in subsequent periods, so the tightening is a one-time deviation.

    Args:
        magnitude: Percentage points above Taylor-rule rate.
        period: Quarter in which the shock hits.

    Returns:
        Shock override dict for Sim.run().
    """
    return {(period, "r_override"): magnitude}


def demand_shock(
    magnitude: float = 2.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Positive demand shock: an unexpected boost to output gap.

    Sets eps_y to `magnitude` at the shock period. Could represent
    a fiscal stimulus, a surge in consumer confidence, etc.

    Args:
        magnitude: Size of the demand shock in percentage points of output gap.
        period: Quarter in which the shock hits.

    Returns:
        Shock override dict for Sim.run().
    """
    return {(period, "eps_y"): magnitude}


def cost_push_shock(
    magnitude: float = 2.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Cost-push shock: an unexpected increase in inflation not driven by demand.

    Sets eps_pi to `magnitude` at the shock period. Could represent
    an oil price spike, supply chain disruption, etc.

    Args:
        magnitude: Size of the cost-push shock in percentage points of inflation.
        period: Quarter in which the shock hits.

    Returns:
        Shock override dict for Sim.run().
    """
    return {(period, "eps_pi"): magnitude}


def exchange_rate_shock(
    magnitude: float = 5.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Exchange rate shock: an unexpected appreciation of the real exchange rate.

    Sets eps_e to `magnitude` at the shock period. Could represent
    a capital flow surge, terms-of-trade shift, etc. Positive = appreciation.

    Args:
        magnitude: Size of the exchange rate shock in log-index units.
        period: Quarter in which the shock hits.

    Returns:
        Shock override dict for Sim.run().
    """
    return {(period, "eps_e"): magnitude}


def regime_shift_shock(
    magnitude: float = -2.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Regime-shift shock: the central bank announces a new inflation target.

    Shifts pi_star by `magnitude` percentage points starting at `period`.
    The simulator applies this cumulatively, so pi_star changes permanently.

    Negative magnitude = target reduction (e.g., -2.0 moves pi_star from 6% to 4%).
    Positive magnitude = target increase (e.g., +1.0 moves pi_star up).

    Only meaningful for world 3 (EmergingSim), which supports pi_star_shift
    in shock overrides. World 1 Sim ignores this shock type.

    Args:
        magnitude: Change in inflation target in percentage points.
        period: Quarter in which the regime shift takes effect.

    Returns:
        Shock override dict for EmergingSim.run().
    """
    return {(period, "pi_star_shift"): magnitude}


def external_pressure_shock(
    magnitude: float = 3.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    External pressure shock: global financial conditions shift.

    Changes r_world by `magnitude` percentage points starting at `period`.
    Positive magnitude = global tightening (e.g., +3.0 raises r_world from 2% to 5%).

    In an open economy this forces currency depreciation via UIP, which feeds
    inflation via passthrough, triggering a domestic policy response.

    Only meaningful for world 3 (EmergingSim), which supports r_world_shift
    in shock overrides.

    Args:
        magnitude: Change in foreign interest rate in percentage points.
        period: Quarter in which the external pressure hits.

    Returns:
        Shock override dict for EmergingSim.run().
    """
    return {(period, "r_world_shift"): magnitude}


def combined_shock(
    r_world_magnitude: float = 3.0,
    cost_push_magnitude: float = 2.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Combined external pressure + cost-push shock (emerging-market crisis pattern).

    Simultaneously applies a global rate increase (r_world shift) and a
    domestic cost-push inflation shock (eps_pi). This mimics a common
    emerging-market crisis: global tightening + commodity price spike.

    Args:
        r_world_magnitude: Change in foreign interest rate (pp).
        cost_push_magnitude: Size of the domestic cost-push shock (pp).
        period: Quarter in which both shocks hit.

    Returns:
        Shock override dict for EmergingSim.run().
    """
    return {
        (period, "r_world_shift"): r_world_magnitude,
        (period, "eps_pi"): cost_push_magnitude,
    }


# ---------------------------------------------------------------------------
# World 4 shock types (labor market)
# ---------------------------------------------------------------------------

def wage_shock(
    magnitude: float = 1.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Wage inflation shock: direct injection to wage dynamics.

    Represents minimum wage increases, union bargaining wins, or other
    direct wage pressures. Positive = wage inflation increase.

    Only meaningful for world 4 (HysteresisSim).

    Args:
        magnitude: Wage shock in percentage points of wage inflation.
        period: Quarter in which the shock hits.
    """
    return {(period, "eps_w"): magnitude}


def productivity_shock(
    magnitude: float = 1.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Productivity shock: shift in productivity level.

    Persistent shock — modifies the productivity state variable, which
    feeds into wages via passthrough. Positive = productivity increase.

    Only meaningful for world 4 (HysteresisSim).

    Args:
        magnitude: Productivity change in percentage points.
        period: Quarter in which the shock hits.
    """
    return {(period, "eps_productivity"): magnitude}


def labor_supply_shock(
    magnitude: float = 0.5,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Labor supply shock: direct NAIRU shift.

    Represents immigration, demographic change, labor force participation
    shifts. Positive = NAIRU increases (structural unemployment rises).

    Only meaningful for world 4 (HysteresisSim).

    Args:
        magnitude: NAIRU change in percentage points.
        period: Quarter in which the shock hits.
    """
    return {(period, "eps_u_natural"): magnitude}


def prolonged_demand_shock(
    magnitude: float = -1.5,
    period: int = 60,
    duration: int = 4,
) -> dict[tuple[int, str], float]:
    """
    Prolonged demand shock: sustained negative demand over multiple quarters.

    Tests hysteresis — prolonged unemployment should raise the NAIRU.
    Applies eps_y = magnitude for `duration` consecutive quarters.

    Only meaningful for world 4 (HysteresisSim).

    Args:
        magnitude: Demand shock per quarter in percentage points.
        period: First quarter of the shock sequence.
        duration: Number of consecutive quarters.
    """
    overrides = {}
    for t in range(duration):
        overrides[(period + t, "eps_y")] = magnitude
    return overrides


# ---------------------------------------------------------------------------
# World 5 shock types (multi-country trade network)
# ---------------------------------------------------------------------------

def tariff_shock(
    magnitude: float = 0.10,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Unilateral tariff shock: Country A raises tariff on imports from B.

    Magnitude is in decimal form (0.10 = 10% tariff). Cumulative —
    the tariff level shifts permanently from that period.

    Args:
        magnitude: Tariff rate change (decimal, e.g. 0.10 for 10%).
        period: Quarter the tariff takes effect.
    """
    return {(period, "A_tariff_shift"): magnitude}


def reciprocal_tariff_shock(
    magnitude: float = 0.10,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Reciprocal tariff shock: both countries raise tariffs simultaneously.

    Trade war pattern. Both A and B raise tariffs by the same amount.

    Args:
        magnitude: Tariff rate change per country (decimal).
        period: Quarter both tariffs take effect.
    """
    return {
        (period, "A_tariff_shift"): magnitude,
        (period, "B_tariff_shift"): magnitude,
    }


def supply_chain_break_shock(
    magnitude: float = -0.5,
    period: int = 60,
    duration: int = 4,
) -> dict[tuple[int, str], float]:
    """
    Supply chain break: Company Y output reduced for N periods.

    Simulates a production disruption at the upstream supplier.
    Magnitude is the output level override (e.g. -0.5 means 50% below
    normal; output is in deviation terms, so -0.5 = halved from baseline).

    Args:
        magnitude: Y output level during disruption (deviation units).
        period: First quarter of disruption.
        duration: Number of quarters disrupted.
    """
    overrides = {}
    for t in range(duration):
        overrides[(period + t, "Y_output_override")] = magnitude
    return overrides


def asymmetric_demand_shock(
    magnitude: float = -2.0,
    period: int = 60,
    country: str = "A",
) -> dict[tuple[int, str], float]:
    """
    Asymmetric demand shock: hits one country only.

    The other country's demand is not directly shocked (but may be
    affected indirectly via trade linkages).

    Args:
        magnitude: Demand shock in pp of output gap.
        period: Quarter the shock hits.
        country: "A" or "B".
    """
    eps_key = f"eps_{country}y"
    return {(period, eps_key): magnitude}


def cross_border_contagion_shock(
    magnitude: float = 1.0,
    period: int = 60,
    country: str = "A",
) -> dict[tuple[int, str], float]:
    """
    Cross-border monetary contagion: one country tightens, observe transmission.

    Applies a monetary override to the specified country. The question
    tests whether the model traces the transmission to the other country
    via trade flows.

    Args:
        magnitude: Rate override in percentage points.
        period: Quarter the monetary action takes effect.
        country: "A" or "B".
    """
    return {(period, f"{country}_r_override"): magnitude}
