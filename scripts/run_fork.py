"""Fork experiment runner.

Three-arm paired structural intervention on shared baseline histories.
See docs/FORK_EXPERIMENT_DESIGN.md for full spec.

Arms:
  ARM 1 (placebo): history + "parameter reviewed, left unchanged"
  ARM 2 (change + equation): history + announcement + governing equation
  ARM 3 (baseline): history + plain forecast, no announcement

Usage:
    python scripts/run_fork.py              # full run
    python scripts/run_fork.py --gt-only    # compute GT + history only (no API calls)
    python scripts/run_fork.py --dry-run    # build schedule, print counts, stop
"""

import argparse
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

V2_ROOT = Path(__file__).resolve().parent.parent
V1_ROOT = V2_ROOT

load_dotenv(V2_ROOT / ".env")

sys.path.insert(0, str(V2_ROOT / "src"))

from lmm2.openrouter_caller import call_openrouter, call_gemini

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(V2_ROOT / "outputs" / "fork_run.log"),
    ],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

RUN_ID = "fork_run"
OUTPUT_DIR = V2_ROOT / "outputs" / RUN_ID
RAW_DIR = OUTPUT_DIR / "raw_responses"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.jsonl"
GT_FILE = OUTPUT_DIR / "fork_ground_truth.json"
HISTORY_DIR = OUTPUT_DIR / "histories"
MANIFEST_FILE = OUTPUT_DIR / "prompt_manifest.jsonl"

SEEDS = list(range(10))
HORIZONS = [1, 4, 8]
N_MC_PATHS = 100  # CRN makes 100 sufficient (paired variance is tiny)
MC_BASE_SEED = 1000

SPEND_CAPS = {"openrouter": 160.0, "google": 45.0}  # cumulative: prior ~$94 OR + ~$27 G
PARSE_RATE_THRESHOLD = 0.85

MODELS = [
    {"name": "Claude Sonnet 4.6", "caller": "openrouter",
     "slug": "anthropic/claude-sonnet-4.6",
     "max_tokens": 8192, "json_mode": False, "extra_params": {}},
    {"name": "GPT-5.5", "caller": "openrouter",
     "slug": "openai/gpt-5.5",
     "max_tokens": 8192, "json_mode": False,
     "extra_params": {"reasoning": {"effort": "minimal"}}},
    {"name": "Gemini 3.5 Flash", "caller": "gemini",
     "slug": "gemini-3.5-flash",
     "max_tokens": 16384, "json_mode": True, "extra_params": {}},
    {"name": "Claude Opus 4.8", "caller": "openrouter",
     "slug": "anthropic/claude-opus-4-8",
     "max_tokens": 8192, "json_mode": False, "extra_params": {}},
]

WORLDS = {
    "world1": {
        "config": str(V1_ROOT / "config/ball_baseline.yaml"),
        "sim_module": "llmmatrix.simulator", "sim_class": "Sim",
        "variables": ["y", "pi", "r", "e", "u"],
    },
    "world2": {
        "config": str(V1_ROOT / "config/world2_closed_economy.yaml"),
        "sim_module": "llmmatrix.world2_simulator", "sim_class": "ClosedSim",
        "variables": ["y", "pi", "r", "u"],
    },
    "world3": {
        "config": str(V1_ROOT / "config/world3_emerging_market.yaml"),
        "sim_module": "llmmatrix.world3_simulator", "sim_class": "EmergingSim",
        "variables": ["y", "pi", "r", "e", "u"],
    },
    "world4": {
        "config": str(V1_ROOT / "config/world4_labor_hysteresis.yaml"),
        "sim_module": "llmmatrix.world4_simulator", "sim_class": "HysteresisSim",
        "variables": ["y", "pi", "r", "u", "w", "u_natural"],
    },
}

PARAMS = {
    "phillips_slope": {
        "world1": {"key": "phillips_curve.output_slope", "baseline": 0.4,
                   "settings": [0.1, 0.2, 0.4, 0.6, 0.8],
                   "ood": [0.05, 1.0]},
        "world2": {"key": "phillips_curve.output_slope", "baseline": 0.4,
                   "settings": [0.1, 0.2, 0.4, 0.6, 0.8]},
        "world3": {"key": "phillips_curve.output_slope", "baseline": 0.5,
                   "settings": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "world4": {"key": "price_phillips.output_slope", "baseline": 0.2,
                   "settings": [0.05, 0.1, 0.2, 0.3, 0.5]},
    },
    "taylor_phi_pi": {
        "world1": {"key": "taylor_rule.inflation_coefficient", "baseline": 1.5,
                   "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
        "world2": {"key": "taylor_rule.inflation_coefficient", "baseline": 1.5,
                   "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
        "world3": {"key": "taylor_rule.inflation_coefficient", "baseline": 2.0,
                   "settings": [1.1, 1.5, 2.0, 2.5, 3.0]},
        "world4": {"key": "taylor_rule.inflation_coefficient", "baseline": 1.5,
                   "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
    },
    "is_sensitivity": {
        "world1": {"key": "is_curve.interest_sensitivity", "baseline": 0.6,
                   "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world2": {"key": "is_curve.interest_sensitivity", "baseline": 0.6,
                   "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world3": {"key": "is_curve.interest_sensitivity", "baseline": 0.6,
                   "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world4": {"key": "is_curve.interest_sensitivity", "baseline": 0.6,
                   "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
    },
    "wage_gap_slope": {
        "world4": {"key": "wage_phillips.unemployment_gap_slope", "baseline": 0.5,
                   "settings": [0.1, 0.3, 0.5, 0.7, 1.0]},
    },
}

EXCLUDED_GT_KEYS = {
    "world3__phillips_slope__0.7",
    "world3__phillips_slope__0.9",
}

# ═══════════════════════════════════════════════════════════════════════
# ARM 2: EQUATION TEXT (per parameter)
# ═══════════════════════════════════════════════════════════════════════

EQUATION_TEXT = {
    "phillips_curve.output_slope": (
        "This parameter enters the Phillips curve: "
        "pi_t = pi* + rho_pi*(pi_{t-1} - pi*) + b2*y_{t-1} "
        "[+ FX passthrough terms if applicable] + epsilon. "
        "A higher slope means the output gap has a stronger direct "
        "effect on inflation."
    ),
    "price_phillips.output_slope": (
        "This parameter enters the price Phillips curve: "
        "pi_t = pi* + rho_pi*(pi_{t-1} - pi*) + psi*(w_t - w*) + b2*y_{t-1} + epsilon. "
        "A higher output slope means the output gap has a stronger direct "
        "effect on inflation alongside the wage channel."
    ),
    "taylor_rule.inflation_coefficient": (
        "This parameter enters the Taylor rule: "
        "r_t = rho*r_{t-1} + (1-rho)*[r* + phi_pi*(pi_t - pi*) + phi_y*y_t]. "
        "A higher coefficient means the central bank raises rates more "
        "aggressively in response to inflation above target."
    ),
    "is_curve.interest_sensitivity": (
        "This parameter enters the IS curve: "
        "y_t = phi_y*y_{t-1} - beta_r*(r_{t-1} - r*) "
        "[+ exchange rate terms if applicable] + epsilon. "
        "A higher sensitivity means output responds more strongly to "
        "interest rate deviations from neutral."
    ),
    "wage_phillips.unemployment_gap_slope": (
        "This parameter enters the wage Phillips curve: "
        "w_t = w* + rho_w*(w_{t-1} - w*) - lambda*(u_{t-1} - u_nat_{t-1}) "
        "+ eta*a_{t-1} + epsilon. "
        "A higher slope means wages respond more strongly to the "
        "unemployment gap."
    ),
}

PARAM_PLAIN_NAMES = {
    "phillips_curve.output_slope": "the Phillips-curve slope",
    "price_phillips.output_slope": "the price Phillips-curve output slope",
    "taylor_rule.inflation_coefficient": "the Taylor-rule inflation coefficient",
    "is_curve.interest_sensitivity": "the IS-curve interest rate sensitivity",
    "wage_phillips.unemployment_gap_slope": "the wage Phillips-curve unemployment gap slope",
}


# ═══════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════

def build_fork_prompt(narrative, arm, config_key, baseline_val,
                      modified_val, variables):
    """Build prompt for one arm of the fork experiment.

    Args:
        narrative: Base 60-quarter narrative text.
        arm: 'arm1_placebo', 'arm2_change', or 'arm3_baseline'.
        config_key: e.g. 'phillips_curve.output_slope'.
        baseline_val: Current baseline value.
        modified_val: Value being changed to (ARM 2 only).
        variables: Forecast variable list.

    Returns:
        Full prompt string.
    """
    parts = [narrative, ""]

    if arm == "arm1_placebo":
        param_name = PARAM_PLAIN_NAMES[config_key]
        parts.append(
            f"At quarter 60, the Central Bank of Vantria reviewed "
            f"{param_name} and confirmed it will remain at its current "
            f"value of {baseline_val}."
        )
        parts.append("")

    elif arm == "arm2_change":
        param_name = PARAM_PLAIN_NAMES[config_key]
        equation = EQUATION_TEXT[config_key]
        parts.append(
            f"At quarter 60, the Central Bank of Vantria announced that "
            f"{param_name} will change from {baseline_val} to "
            f"{modified_val} effective quarter 61."
        )
        parts.append("")
        parts.append(equation)
        parts.append("")

    # ARM 3: no announcement at all

    # Forecast request (identical across arms)
    horizon_labels = ", ".join(f"Q+{h}" for h in HORIZONS)
    variable_list = ", ".join(variables)
    parts.append(
        f"Based on the economic history above, provide your forecast for the "
        f"following variables at horizons {horizon_labels}:\n\n"
        f"Variables: {variable_list}\n\n"
        f"For each variable and horizon, provide:\n"
        f"- point: your point estimate\n"
        f"- ci_low: lower bound of your 80% confidence interval\n"
        f"- ci_high: upper bound of your 80% confidence interval\n\n"
        f"Respond with a JSON object. Keys should be in the format "
        f"'{{variable}}_{{horizon}}' (e.g., 'pi_1' for inflation at Q+1, "
        f"'pi_4' for inflation at Q+4, 'pi_8' for inflation at Q+8). "
        f"Each value should be an object with 'point', 'ci_low', 'ci_high'.\n\n"
        f"Return ONLY the JSON object, no other text."
    )
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# PARSING (same as run_structural_tracking.py)
# ═══════════════════════════════════════════════════════════════════════

def parse_forecast(text, expected_keys):
    """Parse and validate a forecast response."""
    if not text:
        return {"success": False, "method": "empty", "forecast": None,
                "missing": expected_keys, "malformed": []}

    cleaned = text.strip()

    # Try code block
    m = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if m:
        parsed, method = _try_json(m.group(1).strip(), "code_block")
        if parsed is not None:
            return _validate(parsed, expected_keys, method)

    # Try raw
    parsed, method = _try_json(cleaned, "raw_json")
    if parsed is not None:
        return _validate(parsed, expected_keys, method)

    # Try brace extraction
    depth = 0
    start_idx = end_idx = -1
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end_idx = i + 1
    if start_idx >= 0 and end_idx > start_idx:
        parsed, method = _try_json(text[start_idx:end_idx], "brace_extract")
        if parsed is not None:
            return _validate(parsed, expected_keys, method)

    return {"success": False, "method": "failed", "forecast": None,
            "missing": expected_keys, "malformed": []}


def _try_json(text, method):
    try:
        p = json.loads(text)
        if isinstance(p, dict):
            if "predictions" in p:
                p = p["predictions"]
            return p, method
    except (json.JSONDecodeError, ValueError):
        pass
    return None, method


def _validate(parsed, expected_keys, method):
    missing = [k for k in expected_keys if k not in parsed]
    malformed = []
    for k in expected_keys:
        if k in parsed:
            entry = parsed[k]
            if not isinstance(entry, dict):
                malformed.append(k)
            elif not all(isinstance(entry.get(f), (int, float))
                         for f in ["point", "ci_low", "ci_high"]):
                malformed.append(k)
    success = len(missing) == 0 and len(malformed) == 0
    return {"success": success, "method": method,
            "forecast": parsed if success else None,
            "missing": missing, "malformed": malformed}


# ═══════════════════════════════════════════════════════════════════════
# API CALL
# ═══════════════════════════════════════════════════════════════════════

def make_call(model_cfg, prompt, call_id, spend):
    """Make one API call. Returns result dict."""
    import httpx

    platform = "google" if model_cfg["caller"] == "gemini" else "openrouter"

    if spend[platform] >= SPEND_CAPS[platform]:
        return {"error": f"SPEND CAP HIT on {platform} (${spend[platform]:.2f})",
                "call_id": call_id}

    if model_cfg["caller"] == "openrouter":
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_cfg["slug"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": model_cfg["max_tokens"],
        }
        if model_cfg["json_mode"]:
            payload["response_format"] = {"type": "json_object"}
        if model_cfg.get("extra_params"):
            payload.update(model_cfg["extra_params"])

        start = time.time()
        try:
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers, json=payload,
                )
            latency = time.time() - start
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                        "call_id": call_id}
            try:
                data = resp.json()
            except Exception:
                return {"error": f"Non-JSON response: {resp.text[:300]}",
                        "call_id": call_id}
            usage = data.get("usage", {})
            result = {
                "text": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "thinking_tokens": 0,
                "cost_usd": float(usage.get("total_cost", 0.0)),
                "model_served": data.get("model", model_cfg["slug"]),
                "latency_s": latency,
                "error": None,
            }
        except Exception as ex:
            return {"error": str(ex), "call_id": call_id}
    else:
        result = call_gemini(model_cfg["slug"], prompt,
                             model_cfg["max_tokens"], temperature=0)

    if result.get("error"):
        return {"error": result["error"], "call_id": call_id}

    # Track spend
    cost = result.get("cost_usd", 0)
    if cost == 0 and platform == "openrouter":
        tokens_in = result.get("tokens_in", 0)
        tokens_out = result.get("tokens_out", 0)
        rates = {
            "openai/gpt-5.5": (5.0, 30.0),
            "anthropic/claude-sonnet-4.6": (3.0, 15.0),
            "anthropic/claude-opus-4-8": (15.0, 75.0),
        }
        r = rates.get(model_cfg["slug"], (5.0, 30.0))
        cost = tokens_in * r[0] / 1e6 + tokens_out * r[1] / 1e6
        result["cost_usd"] = cost

    spend[platform] += cost

    # Save raw response
    raw_path = RAW_DIR / f"{call_id}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump({
            "call_id": call_id, "model": model_cfg["name"],
            "text": result.get("text", ""),
            "tokens_in": result.get("tokens_in", 0),
            "tokens_out": result.get("tokens_out", 0),
            "thinking_tokens": result.get("thinking_tokens", 0),
            "cost_usd": cost,
            "model_served": result.get("model_served"),
            "latency_s": result.get("latency_s", 0),
            "timestamp": datetime.now().isoformat(),
        }, f)

    return result


# ═══════════════════════════════════════════════════════════════════════
# GROUND TRUTH (CRN)
# ═══════════════════════════════════════════════════════════════════════

def compute_fork_gt(sim_cache):
    """Compute fork ground truth with common random numbers.

    For each (world, param, setting, seed):
    1. Generate 60-quarter baseline history
    2. Fork at Q60 via run_counterfactual under BASELINE and MODIFIED params
       using identical MC seeds (CRN)
    3. Store true_delta = E[modified] - E[baseline] per var_h
    4. Store MC_SE = std(delta_path) / sqrt(n_paths)

    Returns:
        dict: {cell_key -> {var_h -> {true_delta, mc_se, eligible}}}
    """
    import tempfile
    from llmmatrix.monte_carlo import run_counterfactual

    gt = {}
    tmp_dir = Path(tempfile.gettempdir()) / "lmm2_fork_gt"
    tmp_dir.mkdir(exist_ok=True)

    total_cells = 0
    ood_cells = 0

    # Cache baseline histories: (world, seed) -> (history_df, base_config_path)
    history_cache = {}

    for param_name, param_worlds in PARAMS.items():
        for world_name, wcfg in param_worlds.items():
            wc = WORLDS[world_name]
            sim_class = sim_cache[world_name]
            base_config_path = wc["config"]
            baseline_val = wcfg["baseline"]
            config_key = wcfg["key"]
            variables = wc["variables"]
            max_h = max(HORIZONS)

            # All settings to test (main grid + OOD if defined)
            all_settings = list(wcfg["settings"])
            ood_settings = wcfg.get("ood", [])
            all_settings = all_settings + ood_settings

            for setting in all_settings:
                if setting == baseline_val:
                    continue
                cell_stem = f"{world_name}__{param_name}__{setting}"
                if cell_stem in EXCLUDED_GT_KEYS:
                    continue

                is_ood = setting in ood_settings

                # Modified config (write once per setting, reuse across seeds)
                with open(base_config_path) as f:
                    config_mod = yaml.safe_load(f)
                keys = config_key.split(".")
                target = config_mod
                for k in keys[:-1]:
                    target = target[k]
                target[keys[-1]] = setting
                mod_tmp = tmp_dir / f"mod_{world_name}_{param_name}_{setting}.yaml"
                with open(mod_tmp, "w") as f:
                    yaml.dump(config_mod, f, default_flow_style=False)

                for seed in SEEDS:
                    cell_key = f"{cell_stem}__s{seed}"

                    # Get or generate baseline history
                    hist_key = (world_name, seed)
                    if hist_key not in history_cache:
                        sim = sim_class(base_config_path, seed=seed)
                        initial = sim.get_initial_state()
                        traj = sim.run(60, initial)
                        history_df = sim.to_dataframe(traj)
                        history_cache[hist_key] = history_df
                    history_df = history_cache[hist_key]

                    # CRN: run_counterfactual with same base_seed for both
                    no_shock = lambda: {}

                    # Baseline forward paths
                    base_paths = run_counterfactual(
                        config_path=base_config_path,
                        history_df=history_df,
                        shock_fn=no_shock,
                        n_paths=N_MC_PATHS,
                        n_forward_periods=max_h,
                        base_seed=MC_BASE_SEED,
                        sim_class=sim_class,
                        variable_names=variables,
                    )

                    # Modified forward paths (same MC seeds = CRN)
                    mod_paths = run_counterfactual(
                        config_path=str(mod_tmp),
                        history_df=history_df,
                        shock_fn=no_shock,
                        n_paths=N_MC_PATHS,
                        n_forward_periods=max_h,
                        base_seed=MC_BASE_SEED,
                        sim_class=sim_class,
                        variable_names=variables,
                    )

                    # Compute true_delta and MC_SE for requested horizons
                    cell_gt = {}
                    for vi, var in enumerate(variables):
                        for h in HORIZONS:
                            hi = h - 1
                            if hi >= max_h:
                                continue
                            delta_paths = mod_paths[:, hi, vi] - base_paths[:, hi, vi]
                            true_delta = float(np.mean(delta_paths))
                            mc_se = float(np.std(delta_paths, ddof=1) / np.sqrt(N_MC_PATHS))
                            eligible = abs(true_delta) > 2 * mc_se if mc_se > 0 else abs(true_delta) > 0
                            cell_gt[f"{var}_{h}"] = {
                                "true_delta": true_delta,
                                "mc_se": mc_se,
                                "eligible": eligible,
                                "base_mean": float(np.mean(base_paths[:, hi, vi])),
                                "mod_mean": float(np.mean(mod_paths[:, hi, vi])),
                            }

                    gt[cell_key] = {
                        "world": world_name, "param": param_name,
                        "setting": setting, "seed": seed,
                        "is_ood": is_ood,
                        "forecasts": cell_gt,
                    }

                    total_cells += 1
                    if is_ood:
                        ood_cells += 1

                    if total_cells % 50 == 0:
                        logger.info(f"GT computed: {total_cells} cells "
                                    f"({ood_cells} OOD)")

    logger.info(f"GT complete: {total_cells} cells ({ood_cells} OOD)")

    # OOD stability check
    for cell_key, cell_data in gt.items():
        if cell_data.get("is_ood"):
            max_delta = max(abs(v["true_delta"])
                           for v in cell_data["forecasts"].values())
            if max_delta > 100:
                logger.error(
                    f"OOD STABILITY FAIL: {cell_key} has max |true_delta| = "
                    f"{max_delta:.1f}. EXCLUDING from run."
                )

    return gt


# ═══════════════════════════════════════════════════════════════════════
# PRE-FORK TREND (for conflict classification)
# ═══════════════════════════════════════════════════════════════════════

def compute_prefork_trends(history_df, variables):
    """Compute pre-fork trend direction per variable from the last 8 quarters.

    Returns dict: {var -> {trend_8q, trend_4q, rolling_sd, classification}}
    """
    data = history_df[history_df["period"] > 0].sort_values("period")
    trends = {}
    for var in variables:
        if var not in data.columns:
            continue
        series = data[var].values
        n = len(series)
        if n < 8:
            continue

        # 8-quarter trend
        change_8q = series[-1] - series[-8]
        # 4-quarter trend
        change_4q = series[-1] - series[-4]
        # Rolling SD (last 20 quarters for robust estimate)
        window = series[-20:] if n >= 20 else series
        rolling_sd = float(np.std(np.diff(window)))

        trends[var] = {
            "trend_8q": float(change_8q),
            "trend_4q": float(change_4q),
            "rolling_sd": float(rolling_sd),
        }
    return trends


# ═══════════════════════════════════════════════════════════════════════
# BUILD SCHEDULE
# ═══════════════════════════════════════════════════════════════════════

def build_schedule(gt):
    """Build the full call schedule for the fork experiment.

    Call sharing:
    - ARM 3 (baseline): one per (world, seed, model). No param-specific text.
    - ARM 1 (placebo): one per (world, param, seed, model).
      Text mentions param name but not setting.
    - ARM 2 (change): one per (world, param, setting, seed, model).
    """
    schedule = []

    # ARM 3: baseline forecasts (shared across all params within a world)
    arm3_done = set()
    for cell_key, cell_data in gt.items():
        world = cell_data["world"]
        seed = cell_data["seed"]
        if cell_data["is_ood"]:
            continue  # OOD uses main-grid ARM 3
        for model_cfg in MODELS:
            arm3_key = (world, seed, model_cfg["name"])
            if arm3_key in arm3_done:
                continue
            arm3_done.add(arm3_key)
            call_id = (f"fork__{world}__arm3__s{seed}"
                       f"__{model_cfg['name']}").replace(" ", "_")
            variables = WORLDS[world]["variables"]
            schedule.append({
                "call_id": call_id,
                "arm": "arm3_baseline",
                "world": world,
                "param": None,
                "config_key": None,
                "setting": None,
                "baseline_val": None,
                "seed": seed,
                "model": model_cfg,
                "variables": variables,
                "is_ood": False,
            })

    # ARM 1: placebo (shared across settings within a world/param)
    arm1_done = set()
    for cell_key, cell_data in gt.items():
        world = cell_data["world"]
        param = cell_data["param"]
        seed = cell_data["seed"]
        if cell_data["is_ood"]:
            continue
        config_key = PARAMS[param][world]["key"]
        baseline_val = PARAMS[param][world]["baseline"]
        for model_cfg in MODELS:
            arm1_key = (world, param, seed, model_cfg["name"])
            if arm1_key in arm1_done:
                continue
            arm1_done.add(arm1_key)
            call_id = (f"fork__{world}__{param}__arm1__s{seed}"
                       f"__{model_cfg['name']}").replace(" ", "_")
            variables = WORLDS[world]["variables"]
            schedule.append({
                "call_id": call_id,
                "arm": "arm1_placebo",
                "world": world,
                "param": param,
                "config_key": config_key,
                "setting": baseline_val,
                "baseline_val": baseline_val,
                "seed": seed,
                "model": model_cfg,
                "variables": variables,
                "is_ood": False,
            })

    # ARM 2: change (one per cell per model)
    for cell_key, cell_data in gt.items():
        world = cell_data["world"]
        param = cell_data["param"]
        setting = cell_data["setting"]
        seed = cell_data["seed"]
        config_key = PARAMS[param][world]["key"]
        baseline_val = PARAMS[param][world]["baseline"]
        is_ood = cell_data["is_ood"]
        for model_cfg in MODELS:
            call_id = (f"fork__{world}__{param}__{setting}__arm2__s{seed}"
                       f"__{model_cfg['name']}").replace(" ", "_")
            variables = WORLDS[world]["variables"]
            schedule.append({
                "call_id": call_id,
                "arm": "arm2_change",
                "world": world,
                "param": param,
                "config_key": config_key,
                "setting": setting,
                "baseline_val": baseline_val,
                "seed": seed,
                "model": model_cfg,
                "variables": variables,
                "is_ood": is_ood,
            })

    return schedule


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-only", action="store_true",
                        help="Compute GT and histories only (no API calls)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build schedule, print counts, stop")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # Load simulator classes
    sim_cache = {}
    for world_name, wc in WORLDS.items():
        sim_mod = importlib.import_module(wc["sim_module"])
        sim_class = getattr(sim_mod, wc["sim_class"])
        sim_cache[world_name] = sim_class

    # ── Phase 1: Compute ground truth ──
    if GT_FILE.exists():
        logger.info(f"Loading cached GT from {GT_FILE}")
        gt = json.loads(GT_FILE.read_text())
    else:
        logger.info("Computing fork ground truth with CRN...")
        gt = compute_fork_gt(sim_cache)
        GT_FILE.write_text(json.dumps(gt, indent=2))
        logger.info(f"GT saved to {GT_FILE}")

    # OOD stability check
    ood_ok = True
    for cell_key, cell_data in gt.items():
        if cell_data.get("is_ood"):
            max_delta = max(abs(v["true_delta"])
                           for v in cell_data["forecasts"].values())
            if max_delta > 100:
                logger.error(f"OOD UNSTABLE: {cell_key} max|delta|={max_delta:.1f}")
                ood_ok = False
    if not ood_ok:
        logger.error("OOD stability check FAILED. Remove unstable OOD settings.")

    # Summary
    n_eligible = sum(
        1 for c in gt.values()
        for v in c["forecasts"].values()
        if v["eligible"]
    )
    n_total_varh = sum(len(c["forecasts"]) for c in gt.values())
    logger.info(f"GT cells: {len(gt)}, var×h pairs: {n_total_varh}, "
                f"eligible (|td|>2×SE): {n_eligible} ({n_eligible/n_total_varh*100:.1f}%)")

    if args.gt_only:
        logger.info("--gt-only: stopping after GT computation.")
        return

    # ── Phase 2: Build schedule ──
    schedule = build_schedule(gt)

    arm_counts = defaultdict(int)
    model_counts = defaultdict(int)
    for item in schedule:
        arm_counts[item["arm"]] += 1
        model_counts[item["model"]["name"]] += 1

    logger.info(f"Schedule: {len(schedule)} total calls")
    for arm, count in sorted(arm_counts.items()):
        logger.info(f"  {arm}: {count}")
    for model, count in sorted(model_counts.items()):
        logger.info(f"  {model}: {count}")

    if args.dry_run:
        logger.info("--dry-run: stopping after schedule build.")
        return

    # ── Phase 3: Generate histories + narratives ──
    logger.info("Generating baseline histories and narratives...")
    from llmmatrix.narrative import generate_narrative

    # Cache: (world, seed) -> (history_df, narrative, trends)
    history_cache = {}

    def get_baseline_history(world_name, seed):
        cache_key = (world_name, seed)
        if cache_key in history_cache:
            return history_cache[cache_key]

        wc = WORLDS[world_name]
        sim_class = sim_cache[world_name]

        sim = sim_class(wc["config"], seed=seed)
        initial = sim.get_initial_state()
        traj = sim.run(60, initial)
        history_df = sim.to_dataframe(traj)
        narrative = generate_narrative(history_df)
        trends = compute_prefork_trends(history_df, wc["variables"])

        # Save history
        hist_path = HISTORY_DIR / f"{world_name}_s{seed}.csv"
        history_df.to_csv(hist_path, index=False)

        history_cache[cache_key] = (history_df, narrative, trends)
        return history_df, narrative, trends

    # Pre-generate all histories
    for world_name in WORLDS:
        for seed in SEEDS:
            get_baseline_history(world_name, seed)
    logger.info(f"Generated {len(history_cache)} baseline histories")

    # Save trends for conflict classification
    trends_out = {}
    for (world, seed), (_, _, trends) in history_cache.items():
        trends_out[f"{world}__s{seed}"] = trends
    trends_path = OUTPUT_DIR / "prefork_trends.json"
    trends_path.write_text(json.dumps(trends_out, indent=2))

    # ── Phase 4: Run API calls ──
    logger.info("Starting API calls...")

    completed = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                r = json.loads(line)
                completed.add(r["call_id"])
        logger.info(f"Resuming: {len(completed)} calls already completed")

    remaining = [s for s in schedule if s["call_id"] not in completed]
    logger.info(f"Remaining: {len(remaining)} calls")

    # Load prior spend from checkpoint to enforce cumulative caps
    spend = {"openrouter": 0.0, "google": 0.0}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                r = json.loads(line)
                cost = r.get("cost_usd", 0)
                if cost > 0:
                    platform = "google" if r.get("model") == "Gemini 3.5 Flash" else "openrouter"
                    spend[platform] += cost
        logger.info(f"Prior spend loaded: OR=${spend['openrouter']:.2f}, G=${spend['google']:.2f}")
    stats = defaultdict(lambda: {"calls": 0, "parse_ok": 0, "parse_fail": 0,
                                  "errors": 0, "tokens_in": 0, "tokens_out": 0,
                                  "cost": 0.0})

    checkpoint_f = open(CHECKPOINT_FILE, "a")
    manifest_f = open(MANIFEST_FILE, "a")
    stop_flag = False

    for i, item in enumerate(remaining):
        if stop_flag:
            break

        call_id = item["call_id"]
        model_cfg = item["model"]
        model_name = model_cfg["name"]

        # Parse rate gate
        s = stats[model_name]
        total_attempted = s["parse_ok"] + s["parse_fail"]
        if total_attempted >= 20:
            parse_rate = s["parse_ok"] / total_attempted
            if parse_rate < PARSE_RATE_THRESHOLD:
                logger.error(
                    f"PARSE RATE GATE: {model_name} at "
                    f"{parse_rate*100:.1f}% ({s['parse_ok']}/{total_attempted}). "
                    f"Skipping further calls for this model."
                )
                continue

        # Get history + narrative
        _, narrative, _ = get_baseline_history(item["world"], item["seed"])

        # Build prompt
        prompt = build_fork_prompt(
            narrative, item["arm"], item.get("config_key"),
            item.get("baseline_val"), item.get("setting"),
            item["variables"],
        )

        # SHA manifest entry
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        manifest_f.write(json.dumps({
            "call_id": call_id,
            "prompt_sha256": prompt_sha,
            "arm": item["arm"],
            "world": item["world"],
            "param": item.get("param"),
            "setting": item.get("setting"),
            "seed": item["seed"],
            "model": model_name,
            "prompt_length": len(prompt),
            "timestamp": datetime.now().isoformat(),
        }) + "\n")
        manifest_f.flush()

        # Make call
        result = make_call(model_cfg, prompt, call_id, spend)

        if result.get("error"):
            if "SPEND CAP" in str(result.get("error", "")):
                logger.error(f"STOPPING: {result['error']}")
                stop_flag = True
            else:
                logger.warning(f"API error on {call_id}: {result['error']}")
            stats[model_name]["errors"] += 1

            record = {
                "call_id": call_id,
                "arm": item["arm"],
                "world": item["world"],
                "param": item.get("param"),
                "setting": item.get("setting"),
                "seed": item["seed"],
                "model": model_name,
                "error": result.get("error"),
                "parse_success": False,
                "forecast": None,
                "is_ood": item.get("is_ood", False),
                "timestamp": datetime.now().isoformat(),
            }
            checkpoint_f.write(json.dumps(record, default=str) + "\n")
            checkpoint_f.flush()
            completed.add(call_id)
            continue

        # Parse
        expected_keys = [f"{v}_{h}" for v in item["variables"] for h in HORIZONS]
        parse_result = parse_forecast(result.get("text", ""), expected_keys)

        # Update stats
        stats[model_name]["calls"] += 1
        stats[model_name]["tokens_in"] += result.get("tokens_in", 0)
        stats[model_name]["tokens_out"] += result.get("tokens_out", 0)
        stats[model_name]["cost"] += result.get("cost_usd", 0)

        if parse_result["success"]:
            stats[model_name]["parse_ok"] += 1
        else:
            stats[model_name]["parse_fail"] += 1
            logger.warning(
                f"PARSE FAIL: {call_id} method={parse_result['method']} "
                f"missing={parse_result['missing'][:3]}..."
            )

        # Checkpoint
        record = {
            "call_id": call_id,
            "arm": item["arm"],
            "world": item["world"],
            "param": item.get("param"),
            "setting": item.get("setting"),
            "seed": item["seed"],
            "model": model_name,
            "parse_success": parse_result["success"],
            "parse_method": parse_result["method"],
            "forecast": parse_result["forecast"],
            "missing_keys": parse_result["missing"],
            "malformed_keys": parse_result["malformed"],
            "tokens_in": result.get("tokens_in", 0),
            "tokens_out": result.get("tokens_out", 0),
            "thinking_tokens": result.get("thinking_tokens", 0),
            "cost_usd": result.get("cost_usd", 0),
            "model_served": result.get("model_served"),
            "latency_s": result.get("latency_s", 0),
            "is_ood": item.get("is_ood", False),
            "timestamp": datetime.now().isoformat(),
        }
        checkpoint_f.write(json.dumps(record, default=str) + "\n")
        checkpoint_f.flush()
        completed.add(call_id)

        # Progress report every 50 calls
        if (i + 1) % 50 == 0:
            logger.info(
                f"Progress: {i+1}/{len(remaining)} | "
                f"Spend: OR=${spend['openrouter']:.2f} G=${spend['google']:.2f}"
            )
            for mn, ms in stats.items():
                total = ms["parse_ok"] + ms["parse_fail"]
                rate = ms["parse_ok"] / total * 100 if total else 0
                logger.info(
                    f"  {mn}: {total} calls, {rate:.1f}% parse, "
                    f"${ms['cost']:.2f}"
                )

    checkpoint_f.close()
    manifest_f.close()

    # Final stats
    logger.info("=== FINAL STATS ===")
    logger.info(f"Spend: OpenRouter=${spend['openrouter']:.2f}, "
                f"Google=${spend['google']:.2f}")
    for mn, ms in sorted(stats.items()):
        total = ms["parse_ok"] + ms["parse_fail"]
        rate = ms["parse_ok"] / total * 100 if total else 0
        logger.info(
            f"  {mn}: {total} calls, {rate:.1f}% parse, "
            f"${ms['cost']:.2f}, {ms['errors']} errors"
        )

    # Save run stats
    run_stats = {
        "spend": spend,
        "stats": {k: dict(v) for k, v in stats.items()},
        "total_calls": sum(s["calls"] for s in stats.values()),
        "total_errors": sum(s["errors"] for s in stats.values()),
        "timestamp": datetime.now().isoformat(),
    }
    (OUTPUT_DIR / "run_stats.json").write_text(json.dumps(run_stats, indent=2))


if __name__ == "__main__":
    main()
