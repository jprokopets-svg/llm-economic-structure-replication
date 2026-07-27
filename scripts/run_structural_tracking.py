"""
Full structural-tracking experiment runner.

Implements the locked PREREGISTRATION.md design:
  4 worlds × 4 parameters × settings × h=[1,4,8] × conditions × 10 seeds × 4 models

Features:
  - Incremental checkpointing: results saved per batch
  - Resumable: skips already-completed calls on restart
  - All raw responses saved to disk
  - Parse failures logged explicitly, never silently dropped
  - Per-platform spend tracking with safety stops
"""

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
from lmm2.baselines import naive_baseline, ar1_baseline, var_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(V2_ROOT / "outputs" / "run.log"),
    ],
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

RUN_ID = "structural_tracking_v2"
OUTPUT_DIR = V2_ROOT / "outputs" / RUN_ID
RAW_DIR = OUTPUT_DIR / "raw_responses"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.jsonl"
BASELINE_RESULTS_FILE = OUTPUT_DIR / "baseline_results.json"

SEEDS = list(range(10))
HORIZONS = [1, 4, 8]

SPEND_CAPS = {"openrouter": 300.0, "google": 85.0}

MODELS = [
    {"name": "GPT-5.5", "caller": "openrouter", "slug": "openai/gpt-5.5",
     "max_tokens": 8192, "json_mode": False},
    {"name": "Claude Sonnet 4.6", "caller": "openrouter", "slug": "anthropic/claude-sonnet-4.6",
     "max_tokens": 8192, "json_mode": False},
    {"name": "DeepSeek V4 Pro", "caller": "openrouter", "slug": "deepseek/deepseek-v4-pro",
     "max_tokens": 4096, "json_mode": True},
    {"name": "Gemini 3.5 Flash", "caller": "gemini", "slug": "gemini-3.5-flash",
     "max_tokens": 16384, "json_mode": True},
]

WORLDS = {
    "world1": {
        "config": str(V1_ROOT / "config/ball_baseline.yaml"),
        "sim_module": "llmmatrix.simulator", "sim_class": "Sim",
        "variables": ["y", "pi", "r", "e", "u"],
        "shock_module": "llmmatrix.shocks", "shock_fn": "demand_shock", "shock_mag": 2.0,
    },
    "world2": {
        "config": str(V1_ROOT / "config/world2_closed_economy.yaml"),
        "sim_module": "llmmatrix.world2_simulator", "sim_class": "ClosedSim",
        "variables": ["y", "pi", "r", "u"],
        "shock_module": "llmmatrix.shocks", "shock_fn": "demand_shock", "shock_mag": 2.0,
    },
    "world3": {
        "config": str(V1_ROOT / "config/world3_emerging_market.yaml"),
        "sim_module": "llmmatrix.world3_simulator", "sim_class": "EmergingSim",
        "variables": ["y", "pi", "r", "e", "u"],
        "shock_module": "llmmatrix.shocks", "shock_fn": "demand_shock", "shock_mag": 3.0,
    },
    "world4": {
        "config": str(V1_ROOT / "config/world4_labor_hysteresis.yaml"),
        "sim_module": "llmmatrix.world4_simulator", "sim_class": "HysteresisSim",
        "variables": ["y", "pi", "r", "u", "w", "u_natural"],
        "shock_module": "llmmatrix.shocks", "shock_fn": "demand_shock", "shock_mag": 2.0,
    },
}

PARAMS = {
    "phillips_slope": {
        "world1": {"key": "phillips_curve.output_slope", "baseline": 0.4, "settings": [0.1, 0.2, 0.4, 0.6, 0.8]},
        "world2": {"key": "phillips_curve.output_slope", "baseline": 0.4, "settings": [0.1, 0.2, 0.4, 0.6, 0.8]},
        "world3": {"key": "phillips_curve.output_slope", "baseline": 0.5, "settings": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "world4": {"key": "price_phillips.output_slope", "baseline": 0.2, "settings": [0.05, 0.1, 0.2, 0.3, 0.5]},
    },
    "taylor_phi_pi": {
        "world1": {"key": "taylor_rule.inflation_coefficient", "baseline": 1.5, "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
        "world2": {"key": "taylor_rule.inflation_coefficient", "baseline": 1.5, "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
        "world3": {"key": "taylor_rule.inflation_coefficient", "baseline": 2.0, "settings": [1.1, 1.5, 2.0, 2.5, 3.0]},
        "world4": {"key": "taylor_rule.inflation_coefficient", "baseline": 1.5, "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
    },
    "is_sensitivity": {
        "world1": {"key": "is_curve.interest_sensitivity", "baseline": 0.6, "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world2": {"key": "is_curve.interest_sensitivity", "baseline": 0.6, "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world3": {"key": "is_curve.interest_sensitivity", "baseline": 0.6, "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world4": {"key": "is_curve.interest_sensitivity", "baseline": 0.6, "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
    },
    "wage_gap_slope": {
        "world4": {"key": "wage_phillips.unemployment_gap_slope", "baseline": 0.5, "settings": [0.1, 0.3, 0.5, 0.7, 1.0]},
    },
}

# Relabel subset
RELABEL_SUBSET = {("world1", "phillips_slope"), ("world2", "phillips_slope")}

# ═══════════════════════════════════════════════════════════════════════
# NARRATIVE GENERATION
# ═══════════════════════════════════════════════════════════════════════

TOLD_TEMPLATES = {
    "phillips_curve.output_slope": (
        "Vantria's Phillips curve has a slope of {value:.2f}, meaning each "
        "percentage point of output gap translates into {value:.2f} percentage "
        "points of additional inflation pressure."
    ),
    "price_phillips.output_slope": (
        "Vantria's price Phillips curve has an output slope of {value:.2f}, meaning "
        "each percentage point of output gap translates into {value:.2f} percentage "
        "points of additional inflation pressure through the wage-price channel."
    ),
    "taylor_rule.inflation_coefficient": (
        "The Central Bank of Vantria follows a Taylor rule with an inflation "
        "coefficient of {value:.1f}, meaning it raises the policy rate by "
        "{value:.1f} percentage points for each percentage point of inflation "
        "above target."
    ),
    "is_curve.interest_sensitivity": (
        "Vantria's IS curve has an interest rate sensitivity of {value:.1f}, "
        "meaning each percentage point increase in the real interest rate above "
        "its neutral level reduces the output gap by {value:.1f} percentage points."
    ),
    "wage_phillips.unemployment_gap_slope": (
        "Vantria's wage Phillips curve has an unemployment gap slope of {value:.1f}, "
        "meaning each percentage point of unemployment above the natural rate "
        "reduces wage inflation by {value:.1f} percentage points."
    ),
}


def build_prompt(narrative: str, condition: str, config_key: str,
                 param_value: float, variables: list[str]) -> str:
    """Build the full forecast prompt."""
    parts = [narrative, ""]

    if condition == "told":
        template = TOLD_TEMPLATES.get(config_key)
        if template:
            parts.append(template.format(value=param_value))
            parts.append("")

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
# PARSING
# ═══════════════════════════════════════════════════════════════════════

def parse_forecast(text: str, expected_keys: list[str]) -> dict:
    """Parse and validate a forecast response. Returns a result dict."""
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
    depth = 0; s = e = -1
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0: s = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: e = i + 1
    if s >= 0 and e > s:
        parsed, method = _try_json(text[s:e], "brace_extract")
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
    return {"success": success, "method": method, "forecast": parsed if success else None,
            "missing": missing, "malformed": malformed}


# ═══════════════════════════════════════════════════════════════════════
# CALL + SAVE
# ═══════════════════════════════════════════════════════════════════════

def make_call(model_cfg: dict, prompt: str, call_id: str,
              spend: dict) -> dict:
    """Make one API call, save raw response, return result."""
    platform = "google" if model_cfg["caller"] == "gemini" else "openrouter"

    # Spend check
    if spend[platform] >= SPEND_CAPS[platform]:
        return {"error": f"SPEND CAP HIT on {platform} (${spend[platform]:.2f})",
                "call_id": call_id}

    # Call
    if model_cfg["caller"] == "openrouter":
        kwargs = {"model_slug": model_cfg["slug"], "prompt": prompt,
                  "max_tokens": model_cfg["max_tokens"], "temperature": 0}
        if model_cfg["json_mode"]:
            # Use httpx directly for response_format
            import httpx
            headers = {
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_cfg["slug"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": model_cfg["max_tokens"],
                "response_format": {"type": "json_object"},
            }
            start = time.time()
            try:
                with httpx.Client(timeout=300.0) as client:
                    resp = client.post("https://openrouter.ai/api/v1/chat/completions",
                                       headers=headers, json=payload)
                latency = time.time() - start
                if resp.status_code != 200:
                    return {"error": f"HTTP {resp.status_code}", "call_id": call_id}
                data = resp.json()
                usage = data.get("usage", {})
                result = {
                    "text": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    "tokens_in": usage.get("prompt_tokens", 0),
                    "tokens_out": usage.get("completion_tokens", 0),
                    "thinking_tokens": 0,
                    "cost_usd": float(usage.get("total_cost", 0.0)),
                    "model_served": data.get("model", model_cfg["slug"]),
                    "latency_s": latency, "error": None,
                }
            except Exception as ex:
                return {"error": str(ex), "call_id": call_id}
        else:
            result = call_openrouter(**kwargs)
    else:
        result = call_gemini(model_cfg["slug"], prompt,
                             model_cfg["max_tokens"], temperature=0)

    if result.get("error"):
        return {"error": result["error"], "call_id": call_id}

    # Track spend
    cost = result.get("cost_usd", 0)
    if cost == 0 and platform == "openrouter":
        # Estimate from tokens if OpenRouter reports $0
        tokens_in = result.get("tokens_in", 0)
        tokens_out = result.get("tokens_out", 0)
        rates = {"openai/gpt-5.5": (5.0, 30.0), "anthropic/claude-sonnet-4.6": (3.0, 15.0),
                 "deepseek/deepseek-v4-pro": (0.43, 0.87)}
        r = rates.get(model_cfg["slug"], (5.0, 30.0))
        cost = tokens_in * r[0] / 1e6 + tokens_out * r[1] / 1e6
        result["cost_usd"] = cost

    spend[platform] += cost

    # Save raw response
    raw_path = RAW_DIR / f"{call_id}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump({"call_id": call_id, "model": model_cfg["name"],
                    "text": result.get("text", ""),
                    "tokens_in": result.get("tokens_in", 0),
                    "tokens_out": result.get("tokens_out", 0),
                    "thinking_tokens": result.get("thinking_tokens", 0),
                    "cost_usd": cost,
                    "model_served": result.get("model_served"),
                    "latency_s": result.get("latency_s", 0),
                    "timestamp": datetime.now().isoformat()}, f)

    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (V2_ROOT / "outputs").mkdir(exist_ok=True)

    # Load completed call IDs for resumability
    completed = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                r = json.loads(line)
                completed.add(r["call_id"])
        logger.info(f"Resuming: {len(completed)} calls already completed")

    spend = {"openrouter": 0.0, "google": 0.0}
    stats = defaultdict(lambda: {"calls": 0, "parse_ok": 0, "parse_fail": 0,
                                  "errors": 0, "tokens_in": 0, "tokens_out": 0,
                                  "thinking": 0, "cost": 0.0})

    # ── Phase 1: Generate narratives and histories ──────────────────
    logger.info("Phase 1: Generating histories and narratives...")

    # Cache: (world, config_key, setting, seed) -> (history_df, narrative)
    history_cache = {}
    sim_cache = {}

    for world_name, wc in WORLDS.items():
        sim_mod = importlib.import_module(wc["sim_module"])
        sim_class = getattr(sim_mod, wc["sim_class"])
        sim_cache[world_name] = sim_class

    from llmmatrix.narrative import generate_narrative

    def get_history_and_narrative(world_name, config_key, setting, seed):
        cache_key = (world_name, config_key, setting, seed)
        if cache_key in history_cache:
            return history_cache[cache_key]

        wc = WORLDS[world_name]
        sim_class = sim_cache[world_name]

        # Modify config
        with open(wc["config"]) as f:
            config = yaml.safe_load(f)
        if config_key:  # None for baseline
            keys = config_key.split(".")
            target = config
            for k in keys[:-1]:
                target = target[k]
            target[keys[-1]] = setting

        import tempfile
        tmp_dir = Path(tempfile.gettempdir()) / "lmm2_run"
        tmp_dir.mkdir(exist_ok=True)
        safe_key = (config_key or "baseline").replace(".", "_")
        tmp_path = tmp_dir / f"{config['name']}_{safe_key}_{setting}_s{seed}.yaml"
        with open(tmp_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        sim = sim_class(str(tmp_path), seed=seed)
        initial = sim.get_initial_state()
        traj = sim.run(60, initial)
        history_df = sim.to_dataframe(traj)
        narrative = generate_narrative(history_df)

        history_cache[cache_key] = (history_df, narrative)
        return history_df, narrative

    # ── Phase 2: Compute baselines ──────────────────────────────────
    logger.info("Phase 2: Computing baselines...")
    baseline_results = {}

    for param_name, param_worlds in PARAMS.items():
        for world_name, wcfg in param_worlds.items():
            for setting in wcfg["settings"]:
                bl_key = f"{world_name}__{param_name}__{setting}"
                if bl_key in baseline_results:
                    continue

                # Use seed=42 for baseline computation
                history_df, _ = get_history_and_narrative(
                    world_name, wcfg["key"], setting, 42)
                variables = WORLDS[world_name]["variables"]

                bl = {}
                bl["naive"] = naive_baseline(history_df, variables, n_horizons=max(HORIZONS))
                bl["ar1"] = ar1_baseline(history_df, variables, n_horizons=max(HORIZONS))
                try:
                    bl["var"] = var_baseline(history_df, variables, n_horizons=max(HORIZONS))
                    bl["var_status"] = "OK"
                except ValueError as e:
                    bl["var"] = None
                    bl["var_status"] = f"FAILED: {e}"
                    logger.warning(f"VAR failed for {bl_key}: {e}")

                baseline_results[bl_key] = bl

    # Save baselines
    # Convert forecasts to serializable format
    bl_serializable = {}
    for k, v in baseline_results.items():
        bl_serializable[k] = {
            "naive": v["naive"],
            "ar1": v["ar1"],
            "var": v["var"],
            "var_status": v.get("var_status", "OK"),
        }
    with open(BASELINE_RESULTS_FILE, "w") as f:
        json.dump(bl_serializable, f, indent=2, default=str)
    logger.info(f"Baselines saved: {len(baseline_results)} settings")

    # ── Phase 3: Run LLM calls ──────────────────────────────────────
    logger.info("Phase 3: Running LLM calls...")

    # Build the full call schedule
    schedule = []

    for param_name, param_worlds in PARAMS.items():
        for world_name, wcfg in param_worlds.items():
            variables = WORLDS[world_name]["variables"]
            expected_keys = [f"{v}_{h}" for v in variables for h in HORIZONS]

            for setting in wcfg["settings"]:
                is_baseline_setting = (setting == wcfg["baseline"])

                # Determine conditions
                conditions = ["told", "infer"]
                if (world_name, param_name) in RELABEL_SUBSET:
                    conditions.append("relabel")

                for condition in conditions:
                    # Skip baseline settings for told/relabel (baseline has
                    # no parameter to state or infer)
                    # Actually we DO need baselines — they're the reference.
                    # Every setting gets run including baseline.
                    for seed in SEEDS:
                        for model_cfg in MODELS:
                            call_id = (
                                f"{world_name}__{param_name}__{setting}__"
                                f"{condition}__s{seed}__{model_cfg['name']}"
                            ).replace(" ", "_")

                            schedule.append({
                                "call_id": call_id,
                                "world": world_name,
                                "param": param_name,
                                "config_key": wcfg["key"],
                                "setting": setting,
                                "baseline_val": wcfg["baseline"],
                                "condition": condition,
                                "seed": seed,
                                "model": model_cfg,
                                "variables": variables,
                                "expected_keys": expected_keys,
                            })

    logger.info(f"Total scheduled calls: {len(schedule)}")
    logger.info(f"Already completed: {len(completed)}")
    remaining = [s for s in schedule if s["call_id"] not in completed]
    logger.info(f"Remaining: {len(remaining)}")

    # Run
    checkpoint_f = open(CHECKPOINT_FILE, "a")
    batch_size = 10
    stop_flag = False

    for i, item in enumerate(remaining):
        if stop_flag:
            break

        call_id = item["call_id"]

        # Generate history + narrative
        history_df, narrative = get_history_and_narrative(
            item["world"], item["config_key"], item["setting"], item["seed"])

        # Build prompt
        prompt = build_prompt(
            narrative, item["condition"], item["config_key"],
            item["setting"], item["variables"])

        # Make call
        result = make_call(item["model"], prompt, call_id, spend)

        if result.get("error"):
            if "SPEND CAP" in str(result.get("error", "")):
                logger.error(f"STOPPING: {result['error']}")
                stop_flag = True
            else:
                logger.error(f"API error on {call_id}: {result['error']}")
            stats[item["model"]["name"]]["errors"] += 1

            # Checkpoint the error
            record = {
                "call_id": call_id,
                "world": item["world"], "param": item["param"],
                "setting": item["setting"], "condition": item["condition"],
                "seed": item["seed"], "model": item["model"]["name"],
                "error": result.get("error"),
                "parse_success": False, "forecast": None,
                "timestamp": datetime.now().isoformat(),
            }
            checkpoint_f.write(json.dumps(record, default=str) + "\n")
            completed.add(call_id)
            continue

        # Parse
        parse_result = parse_forecast(result.get("text", ""), item["expected_keys"])

        # Update stats
        model_name = item["model"]["name"]
        stats[model_name]["calls"] += 1
        stats[model_name]["tokens_in"] += result.get("tokens_in", 0)
        stats[model_name]["tokens_out"] += result.get("tokens_out", 0)
        stats[model_name]["thinking"] += result.get("thinking_tokens", 0)
        stats[model_name]["cost"] += result.get("cost_usd", 0)

        if parse_result["success"]:
            stats[model_name]["parse_ok"] += 1
        else:
            stats[model_name]["parse_fail"] += 1
            logger.warning(
                f"PARSE FAIL: {call_id} method={parse_result['method']} "
                f"missing={parse_result['missing'][:3]}... "
                f"malformed={parse_result['malformed'][:3]}..."
            )

        # Checkpoint
        record = {
            "call_id": call_id,
            "world": item["world"], "param": item["param"],
            "setting": item["setting"], "condition": item["condition"],
            "seed": item["seed"], "model": model_name,
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
            "timestamp": datetime.now().isoformat(),
        }
        checkpoint_f.write(json.dumps(record, default=str) + "\n")
        checkpoint_f.flush()
        completed.add(call_id)

        # Progress
        if (i + 1) % batch_size == 0:
            pct = (len(completed)) / len(schedule) * 100
            logger.info(
                f"Progress: {len(completed)}/{len(schedule)} ({pct:.1f}%) | "
                f"Spend: OR=${spend['openrouter']:.2f} G=${spend['google']:.2f} | "
                f"Parse: {sum(s['parse_ok'] for s in stats.values())}ok "
                f"{sum(s['parse_fail'] for s in stats.values())}fail "
                f"{sum(s['errors'] for s in stats.values())}err"
            )

    checkpoint_f.close()

    # ── Phase 4: Final report ───────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("RUN COMPLETE")
    logger.info("=" * 70)

    total_calls = sum(s["calls"] for s in stats.values())
    total_parse_ok = sum(s["parse_ok"] for s in stats.values())
    total_parse_fail = sum(s["parse_fail"] for s in stats.values())
    total_errors = sum(s["errors"] for s in stats.values())

    logger.info(f"Total calls: {total_calls}")
    logger.info(f"Parse OK: {total_parse_ok}")
    logger.info(f"Parse fail: {total_parse_fail}")
    logger.info(f"API errors: {total_errors}")
    logger.info(f"Spend: OpenRouter=${spend['openrouter']:.2f}, Google=${spend['google']:.2f}")

    # Save final stats
    final_stats = {
        "run_id": RUN_ID,
        "completed": datetime.now().isoformat(),
        "total_scheduled": len(schedule),
        "total_completed": len(completed),
        "spend": spend,
        "per_model": {k: dict(v) for k, v in stats.items()},
        "stop_reason": "spend_cap" if stop_flag else "complete",
    }
    with open(OUTPUT_DIR / "run_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)

    # Print per-model summary
    print(f"\n{'Model':<25} {'Calls':>6} {'Parse OK':>9} {'Fail':>5} {'Err':>4} "
          f"{'Avg Out':>8} {'Think':>8} {'Cost':>8}")
    print("-" * 80)
    for model_name, s in sorted(stats.items()):
        avg_out = s["tokens_out"] / s["calls"] if s["calls"] > 0 else 0
        avg_think = s["thinking"] / s["calls"] if s["calls"] > 0 else 0
        rate = s["parse_ok"] / (s["parse_ok"] + s["parse_fail"]) * 100 if (s["parse_ok"] + s["parse_fail"]) > 0 else 0
        print(f"{model_name:<25} {s['calls']:>6} {s['parse_ok']:>6} ({rate:.0f}%) "
              f"{s['parse_fail']:>5} {s['errors']:>4} {avg_out:>8.0f} {avg_think:>8.0f} "
              f"${s['cost']:>7.2f}")


if __name__ == "__main__":
    main()
