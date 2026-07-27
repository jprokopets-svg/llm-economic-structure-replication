"""
Statistical baselines for the structural-understanding benchmark.

Three baselines, all fit on the same 60-quarter synthetic history that the
LLMs receive. They answer: "could a standard time-series method, given only
the history and no economic knowledge, match the LLMs' structural sensitivity?"

All baselines return forecasts as a dict mapping variable name to a list of
per-horizon dicts with keys "point", "ci_low", "ci_high". This matches the
format that scoring.py expects from LLM forecasts.

Baselines:
  1. Naive — last-observation-carried-forward.
  2. AR(1) — univariate first-order autoregression per variable.
  3. VAR  — vector autoregression fit jointly on all variables.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR as StatsmodelsVAR

logger = logging.getLogger(__name__)


def _format_forecast(
    point_estimates: dict[str, list[float]],
    std_estimates: dict[str, list[float]],
    n_horizons: int,
    ci_coverage: float = 0.80,
) -> dict[str, dict[str, float]]:
    """
    Convert point estimates and standard deviations into the standard forecast
    format used by scoring.py.

    Args:
        point_estimates: Dict mapping variable name to list of point forecasts
                         (one per horizon).
        std_estimates: Dict mapping variable name to list of forecast std devs.
        n_horizons: Number of forecast horizons.
        ci_coverage: Coverage probability for the confidence interval.

    Returns:
        Dict mapping "{variable}_{horizon}" to {"point", "ci_low", "ci_high"}.
    """
    from scipy import stats as scipy_stats

    # z-score for the given coverage (e.g., 1.28 for 80%)
    z = scipy_stats.norm.ppf(0.5 + ci_coverage / 2)

    forecast = {}
    for variable in point_estimates:
        for h_idx in range(n_horizons):
            horizon = h_idx + 1
            point = point_estimates[variable][h_idx]
            std = std_estimates[variable][h_idx]
            forecast[f"{variable}_{horizon}"] = {
                "point": point,
                "ci_low": point - z * std,
                "ci_high": point + z * std,
            }
    return forecast


def naive_baseline(
    history_df: pd.DataFrame,
    variables: list[str],
    n_horizons: int = 4,
) -> dict[str, dict[str, float]]:
    """
    Naive (random walk) baseline: forecast = last observed value for all horizons.

    Args:
        history_df: The 60-quarter history DataFrame.
        variables: List of variable names to forecast.
        n_horizons: Number of quarterly horizons.

    Returns:
        Forecast dict in standard format.
    """
    last_row = history_df.iloc[-1]

    point_estimates = {}
    std_estimates = {}
    for variable in variables:
        last_value = float(last_row[variable])
        # Naive forecast: constant at last value, zero uncertainty
        point_estimates[variable] = [last_value] * n_horizons
        std_estimates[variable] = [0.0] * n_horizons

    return _format_forecast(point_estimates, std_estimates, n_horizons)


def ar1_baseline(
    history_df: pd.DataFrame,
    variables: list[str],
    n_horizons: int = 4,
) -> dict[str, dict[str, float]]:
    """
    AR(1) baseline: fit univariate AR(1) per variable, then forecast.

    Each variable is modeled independently. This captures persistence but
    ignores cross-variable dynamics.

    Args:
        history_df: The 60-quarter history DataFrame.
        variables: List of variable names to forecast.
        n_horizons: Number of quarterly horizons.

    Returns:
        Forecast dict in standard format.
    """
    point_estimates = {}
    std_estimates = {}

    for variable in variables:
        series = history_df[variable].values

        model = AutoReg(series, lags=1)
        fitted = model.fit()

        forecasts = fitted.predict(
            start=len(series),
            end=len(series) + n_horizons - 1,
        )

        residual_std = float(np.std(fitted.resid))

        points = []
        stds = []
        for h_idx in range(n_horizons):
            points.append(float(forecasts[h_idx]))
            # Uncertainty grows with sqrt(horizon) for AR processes
            stds.append(residual_std * np.sqrt(h_idx + 1))

        point_estimates[variable] = points
        std_estimates[variable] = stds

    return _format_forecast(point_estimates, std_estimates, n_horizons)


def var_baseline(
    history_df: pd.DataFrame,
    variables: list[str],
    n_horizons: int = 4,
    max_lags: int = 4,
) -> dict[str, dict[str, float]]:
    """
    VAR (vector autoregression) baseline: fit jointly on all forecast variables.

    Lag order is selected by AIC, capped at max_lags. With only 60 observations
    and up to 5 variables, higher lag orders risk overfitting and singular
    covariance matrices. max_lags=4 keeps the parameter count manageable:
    a VAR(4) with 5 variables estimates 5*4*5 + 5 = 105 parameters from 56
    usable observations (60 minus 4 lags), which is tight but feasible.

    If the VAR fails to fit (singular matrix, insufficient observations, or
    any other numerical issue), this function raises an exception. It does NOT
    silently fall back to AR(1) — the caller must handle the failure explicitly.

    Args:
        history_df: The 60-quarter history DataFrame.
        variables: List of variable names to forecast.
        n_horizons: Number of quarterly horizons.
        max_lags: Maximum lag order for AIC selection.

    Returns:
        Forecast dict in standard format.

    Raises:
        ValueError: If the VAR cannot be fit (too few observations, singular
                    covariance, or other numerical failure).
    """
    # Build the multivariate series matrix
    # Only include columns that are in the variables list and exist in the DataFrame
    available_variables = [v for v in variables if v in history_df.columns]
    if len(available_variables) < 2:
        raise ValueError(
            f"VAR requires at least 2 variables, but only found "
            f"{available_variables} in the history DataFrame."
        )

    # Drop near-perfectly collinear variables before fitting. In our synthetic
    # worlds, u is a deterministic linear function of y (Okun's law:
    # u = u* - okun*y), which makes the covariance matrix singular. Rather
    # than failing opaquely, we detect |correlation| > 0.999 and drop the
    # later variable, logging which variables were excluded.
    data_matrix = history_df[available_variables].values
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)
    dropped_variables = []
    keep_mask = [True] * len(available_variables)
    for i in range(len(available_variables)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(available_variables)):
            if not keep_mask[j]:
                continue
            if abs(corr_matrix[i, j]) > 0.999:
                dropped_var = available_variables[j]
                kept_var = available_variables[i]
                keep_mask[j] = False
                dropped_variables.append((dropped_var, kept_var))
                logger.warning(
                    f"VAR: dropping '{dropped_var}' — near-perfectly correlated "
                    f"with '{kept_var}' (|r|={abs(corr_matrix[i, j]):.6f}). "
                    f"Forecasts for '{dropped_var}' will be derived from the "
                    f"historical relationship."
                )

    fit_variables = [v for v, keep in zip(available_variables, keep_mask) if keep]
    if len(fit_variables) < 2:
        raise ValueError(
            f"VAR requires at least 2 non-collinear variables, but after "
            f"dropping collinear pairs only {fit_variables} remain."
        )

    data_matrix = history_df[fit_variables].values

    # Check for sufficient observations
    min_obs_needed = max_lags + 2  # need at least max_lags + 2 usable rows
    if len(data_matrix) < min_obs_needed:
        raise ValueError(
            f"VAR needs at least {min_obs_needed} observations for max_lags={max_lags}, "
            f"but history has only {len(data_matrix)} rows."
        )

    # Fit VAR with AIC-selected lag order.
    # statsmodels' select_order() tries every lag from 0 to max_lags and
    # crashes if ANY lag produces a near-singular residual covariance (common
    # in our synthetic data due to tight structural relationships like the
    # Taylor rule). We select the lag manually: try each lag order, keep the
    # ones that fit successfully, pick the best AIC among them.
    var_model = StatsmodelsVAR(data_matrix)
    best_lag = None
    best_aic = float("inf")
    lag_results = {}
    for lag in range(1, max_lags + 1):
        try:
            result = var_model.fit(lag)
            lag_results[lag] = result.aic
            if result.aic < best_aic:
                best_aic = result.aic
                best_lag = lag
        except Exception as e:
            logger.debug(
                f"VAR: lag={lag} failed to fit ({e}), skipping."
            )

    if best_lag is None:
        raise ValueError(
            f"VAR fitting failed at all lag orders 1 through {max_lags}. "
            f"The residual covariance matrix is singular at every lag. "
            f"Check that the history has sufficient variation in all variables."
        )

    logger.info(
        f"VAR lag order selected by AIC: {best_lag} "
        f"(tested lags: {sorted(lag_results.keys())}, "
        f"AICs: {lag_results})"
    )
    selected_lag = best_lag
    fitted = var_model.fit(selected_lag)

    # Forecast
    try:
        # statsmodels VAR.forecast() needs the last `lag_order` rows as input
        forecast_input = data_matrix[-selected_lag:]
        forecasts = fitted.forecast(forecast_input, steps=n_horizons)
        # forecasts shape: (n_horizons, n_variables)
    except Exception as e:
        raise ValueError(
            f"VAR forecasting failed: {e}."
        ) from e

    # Extract forecast standard errors from the model's MSE matrix
    # The forecast error covariance grows with horizon for a VAR
    mse = fitted.mse(n_horizons)
    # mse shape: (n_horizons, n_variables, n_variables)

    point_estimates = {}
    std_estimates = {}
    for v_idx, variable in enumerate(fit_variables):
        points = []
        stds = []
        for h_idx in range(n_horizons):
            points.append(float(forecasts[h_idx, v_idx]))
            # Forecast std = sqrt of diagonal of MSE matrix at this horizon
            stds.append(float(np.sqrt(mse[h_idx, v_idx, v_idx])))
        point_estimates[variable] = points
        std_estimates[variable] = stds

    # Derive forecasts for dropped (collinear) variables using OLS on history.
    # For each dropped variable, regress it on the variable it was collinear
    # with, then apply that linear relationship to the VAR forecasts.
    for dropped_var, kept_var in dropped_variables:
        kept_series = history_df[kept_var].values
        dropped_series = history_df[dropped_var].values
        # Simple OLS: dropped = a + b * kept
        from scipy import stats as scipy_stats
        slope, intercept, _, _, _ = scipy_stats.linregress(
            kept_series, dropped_series,
        )
        kept_forecasts = point_estimates[kept_var]
        kept_stds = std_estimates[kept_var]
        derived_points = [intercept + slope * pt for pt in kept_forecasts]
        # Propagate uncertainty: std(a + b*X) = |b| * std(X)
        derived_stds = [abs(slope) * s for s in kept_stds]
        point_estimates[dropped_var] = derived_points
        std_estimates[dropped_var] = derived_stds
        logger.info(
            f"VAR: derived '{dropped_var}' forecasts from '{kept_var}' "
            f"(slope={slope:.4f}, intercept={intercept:.4f})"
        )

    return _format_forecast(point_estimates, std_estimates, n_horizons)
