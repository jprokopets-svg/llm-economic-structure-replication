"""Phase C v2 relabel run: hand-written static templates.

Uses ``lmm2.relabel_template.build_relabel_prompt`` (NOT the old
substitution-based ``lmm2.narrative.build_forecast_prompt``).

Guarantees:
  - Each prompt is verified against the pre-flight manifest by SHA256
    immediately before it is sent to the API. If any hash disagrees,
    the run aborts with the offending call_id.
  - The whitelist verifier is invoked on every prompt inline (belt and
    suspenders — the manifest was already verified in pre-flight).
  - Credit guard runs before each model block; parse-rate pause fires
    at 85% in the first 50 attempts.
  - New run_id (default ``relabel_run_v4``) so historical v3 checkpoint
    is preserved untouched.

Excluded models: DeepSeek V4 Pro (documented — reasoning tokens
consumed all output budget in v3 even at 8192 max_tokens); Claude Opus
4.8 (deferred by Jake, not cancelled — the projection exceeded the 85%
credit-guard threshold in the pre-flight).
"""

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv


V2_ROOT = Path(__file__).resolve().parent.parent
V1_ROOT = V2_ROOT

sys.path.insert(0, str(V2_ROOT / "src"))

_env_data = os.environ.get("LMM2_DATA_ROOT")
if _env_data:
    V2_ROOT = Path(_env_data).resolve()
_env_v1 = os.environ.get("LMM2_V1_ROOT")
if _env_v1:
    V1_ROOT = Path(_env_v1).resolve()
    

load_dotenv(V2_ROOT / ".env")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


from lmm2.relabel_template import (
    build_relabel_prompt, verify_whitelist, RELABEL_VARIABLE_MAP,
)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

HORIZONS = [1, 4, 8]
SEEDS = list(range(10))

# Ordered: cheapest first (defensive if credits run low unexpectedly),
# with a per-model max_tokens override for GPT-5.5 (heavier reasoning
# preamble at v2 output volume).
MODELS = {
    "Gemini 3.5 Flash": {
        "slug": "google/gemini-3.5-flash",
        "route": "gemini_direct",       # falls back to openrouter on 429
        "workers": 8,
        "max_tokens": 4096,
        "response_format": None,
    },
    "GPT-5.5": {
        "slug": "openai/gpt-5.5",
        "route": "openrouter",
        "workers": 8,
        # v4 relabel run paused GPT-5.5 at 84.9% parse; 17/18 failures
        # were empty-text responses at tokens_out=8192 — the model
        # burned every token on internal reasoning before emitting any
        # JSON. Setting reasoning.effort="minimal" reduces reasoning
        # so the visible-content budget is preserved. Probe confirmed
        # tokens_out drops ~8192 -> ~1450 and JSON parses cleanly at
        # 4096 max_tokens.
        "max_tokens": 4096,
        "response_format": None,
        "reasoning": {"effort": "minimal"},
    },
    "Claude Sonnet 4.6": {
        "slug": "anthropic/claude-sonnet-4.6",
        "route": "openrouter",
        "workers": 8,
        "max_tokens": 4096,
        "response_format": None,
    },
}

WORLDS = {
    "world1": {"config_rel": "config/ball_baseline.yaml",
               "sim_module": "llmmatrix.simulator", "sim_class": "Sim",
               "variables": ["y", "pi", "r", "e", "u"]},
    "world2": {"config_rel": "config/world2_closed_economy.yaml",
               "sim_module": "llmmatrix.world2_simulator", "sim_class": "ClosedSim",
               "variables": ["y", "pi", "r", "u"]},
    "world3": {"config_rel": "config/world3_emerging_market.yaml",
               "sim_module": "llmmatrix.world3_simulator", "sim_class": "EmergingSim",
               "variables": ["y", "pi", "r", "e", "u"]},
    "world4": {"config_rel": "config/world4_labor_hysteresis.yaml",
               "sim_module": "llmmatrix.world4_simulator", "sim_class": "HysteresisSim",
               "variables": ["y", "pi", "r", "u", "w", "u_natural"]},
}

PARAMS = {
    "phillips_slope": {
        "world1": {"key": "phillips_curve.output_slope", "settings": [0.1, 0.2, 0.4, 0.6, 0.8]},
        "world2": {"key": "phillips_curve.output_slope", "settings": [0.1, 0.2, 0.4, 0.6, 0.8]},
        "world3": {"key": "phillips_curve.output_slope", "settings": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "world4": {"key": "price_phillips.output_slope", "settings": [0.05, 0.1, 0.2, 0.3, 0.5]},
    },
    "taylor_phi_pi": {
        "world1": {"key": "taylor_rule.inflation_coefficient", "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
        "world2": {"key": "taylor_rule.inflation_coefficient", "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
        "world3": {"key": "taylor_rule.inflation_coefficient", "settings": [1.1, 1.5, 2.0, 2.5, 3.0]},
        "world4": {"key": "taylor_rule.inflation_coefficient", "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
    },
    "is_sensitivity": {
        "world1": {"key": "is_curve.interest_sensitivity", "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world2": {"key": "is_curve.interest_sensitivity", "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world3": {"key": "is_curve.interest_sensitivity", "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world4": {"key": "is_curve.interest_sensitivity", "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
    },
    "wage_gap_slope": {
        "world4": {"key": "wage_phillips.unemployment_gap_slope", "settings": [0.1, 0.3, 0.5, 0.7, 1.0]},
    },
}


OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CREDITS = "https://openrouter.ai/api/v1/credits"


# ═══════════════════════════════════════════════════════════════════════
# Prompt cache + manifest verification
# ═══════════════════════════════════════════════════════════════════════

_PROMPT_CACHE: dict[tuple, tuple[str, list[str]]] = {}
_PROMPT_CACHE_LOCK = threading.Lock()

_MANIFEST: dict[str, str] = {}  # call_id -> sha256 (populated at start)


def _load_manifest(manifest_path: Path) -> None:
    global _MANIFEST
    with open(manifest_path) as f:
        data = json.load(f)
    _MANIFEST = {r["call_id"]: r["prompt_sha256"] for r in data["records"]}
    logger.info(f"loaded manifest with {len(_MANIFEST)} SHA256 entries")


def _get_history(world: str, config_key: str, setting, seed: int):
    wc = WORLDS[world]
    with open(V1_ROOT / wc["config_rel"]) as f:
        cfg = yaml.safe_load(f)
    keys = config_key.split(".")
    target = cfg
    for k in keys[:-1]:
        target = target[k]
    target[keys[-1]] = setting
    tmp = Path(tempfile.gettempdir()) / "lmm2_relabel_v4"
    tmp.mkdir(exist_ok=True)
    tp = tmp / f"cfg_{world}_{config_key.replace('.', '_')}_{setting}_s{seed}.yaml"
    with open(tp, "w") as f:
        yaml.dump(cfg, f)
    sim_mod = importlib.import_module(wc["sim_module"])
    sim_class = getattr(sim_mod, wc["sim_class"])
    sim = sim_class(str(tp), seed=seed)
    return sim.to_dataframe(sim.run(60, sim.get_initial_state()))


def _get_prompt(world: str, config_key: str, setting, seed: int,
                 manifest_call_id: str) -> tuple[str, list[str]]:
    """Build (or fetch cached) prompt + expected keys. Assert its SHA256
    matches the pre-flight manifest; and run the whitelist verifier
    inline as a belt-and-suspenders check."""
    ck = (world, config_key, setting, seed)
    with _PROMPT_CACHE_LOCK:
        cached = _PROMPT_CACHE.get(ck)
    if cached is not None:
        return cached

    hist = _get_history(world, config_key, setting, seed)
    prompt = build_relabel_prompt(
        hist, world, WORLDS[world]["variables"], HORIZONS,
    )
    expected_keys = [
        f"{RELABEL_VARIABLE_MAP[v]}_{h}"
        for v in WORLDS[world]["variables"]
        for h in HORIZONS
    ]

    # Manifest hash check.
    actual = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    expected = _MANIFEST.get(manifest_call_id)
    if expected is None:
        raise RuntimeError(
            f"call_id {manifest_call_id!r} missing from manifest"
        )
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {manifest_call_id!r}: "
            f"expected {expected}, got {actual}"
        )
    verify_whitelist(prompt, prompt_id=manifest_call_id)

    with _PROMPT_CACHE_LOCK:
        _PROMPT_CACHE[ck] = (prompt, expected_keys)
    return prompt, expected_keys


# ═══════════════════════════════════════════════════════════════════════
# API callers
# ═══════════════════════════════════════════════════════════════════════

def _call_openrouter(slug: str, prompt: str, max_tokens: int,
                     response_format: dict | None,
                     client: httpx.Client,
                     reasoning: dict | None = None) -> dict:
    payload = {
        "model": slug,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    if reasoning:
        payload["reasoning"] = reasoning
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/llm-matrix2",
        "X-Title": "LLM-Matrix2 relabel v2",
    }
    t0 = time.time()
    try:
        r = client.post(OPENROUTER, headers=headers, json=payload, timeout=180)
        latency = time.time() - t0
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:400]}",
                    "latency_s": latency}
        d = r.json()
        text = d.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = d.get("usage", {})
        return {
            "text": text,
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
            "cost_usd": float(usage.get("total_cost", 0.0)),
            "model_served": d.get("model", slug),
            "latency_s": latency,
            "error": None,
        }
    except Exception as exc:
        return {"error": str(exc), "latency_s": time.time() - t0}


def _call_gemini_direct(prompt: str, max_tokens: int) -> dict:
    from lmm2.openrouter_caller import call_gemini
    return call_gemini("gemini-3.5-flash", prompt,
                       max_tokens=max_tokens, temperature=0)


def _openrouter_balance() -> float | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        r = httpx.get(OPENROUTER_CREDITS,
                      headers={"Authorization": f"Bearer {key}"},
                      timeout=15)
        if r.status_code != 200:
            return None
        d = r.json().get("data", {})
        return float(d.get("total_credits", 0)) - float(d.get("total_usage", 0))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Parsing (tolerant, unchanged from v1 style)
# ═══════════════════════════════════════════════════════════════════════

def _parse_relabel(text: str, expected_keys: list[str]) -> dict:
    if not text:
        return {"success": False, "method": "empty",
                "forecast": None, "missing": expected_keys, "malformed": []}
    cleaned = text.strip()
    for pattern in [r"```(?:json)?\s*(.*?)```", r"<JSON>(.*?)</JSON>"]:
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()
            break
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        depth = 0
        s = e = -1
        for i, c in enumerate(cleaned):
            if c == "{":
                if depth == 0:
                    s = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    e = i + 1
                    break
        if s >= 0 and e > s:
            try:
                parsed = json.loads(cleaned[s:e])
            except Exception:
                parsed = None
    if not isinstance(parsed, dict):
        return {"success": False, "method": "brace_extract",
                "forecast": None, "missing": expected_keys, "malformed": []}
    parsed = parsed.get("predictions", parsed)
    missing = [k for k in expected_keys if k not in parsed]
    malformed = []
    for k in expected_keys:
        if k in parsed:
            v = parsed[k]
            if not isinstance(v, dict) or not all(
                isinstance(v.get(f), (int, float))
                for f in ("point", "ci_low", "ci_high")
            ):
                malformed.append(k)
    ok = not missing and not malformed
    return {"success": ok, "method": "parsed", "forecast": parsed if ok else None,
            "missing": missing, "malformed": malformed}


# ═══════════════════════════════════════════════════════════════════════
# Run state
# ═══════════════════════════════════════════════════════════════════════

class RunState:
    def __init__(self, checkpoint_path: Path, raw_dir: Path,
                  spend_cap: float):
        self.checkpoint_path = checkpoint_path
        self.raw_dir = raw_dir
        self.spend_cap = spend_cap
        self.spend = 0.0
        self.stats = {"ok": 0, "fail": 0, "error": 0,
                      "tokens_in": 0, "tokens_out": 0, "cost": 0.0}
        self.completed: set[str] = set()
        self.lock = threading.Lock()
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.completed.add(r["call_id"])
                    except Exception:
                        pass

    def record(self, record: dict):
        with self.lock:
            if record.get("parse_success"):
                self.stats["ok"] += 1
            elif record.get("error"):
                self.stats["error"] += 1
            else:
                self.stats["fail"] += 1
            self.stats["tokens_in"] += record.get("tokens_in", 0)
            self.stats["tokens_out"] += record.get("tokens_out", 0)
            self.stats["cost"] += record.get("cost_usd", 0.0)
            self.spend += record.get("cost_usd", 0.0)
            self.completed.add(record["call_id"])
            with open(self.checkpoint_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def over_cap(self) -> bool:
        return self.spend >= self.spend_cap


def _build_schedule(model_name: str) -> list[dict]:
    schedule = []
    for param_name, worlds in PARAMS.items():
        for world_name, cfg in worlds.items():
            for setting in cfg["settings"]:
                for seed in SEEDS:
                    call_id = (
                        f"{world_name}__{param_name}__{setting}__relabel__s{seed}__"
                        + model_name.replace(" ", "_")
                    )
                    manifest_id = (
                        f"{world_name}__{param_name}__{setting}__relabel__s{seed}"
                    )
                    schedule.append({
                        "call_id": call_id,
                        "manifest_id": manifest_id,
                        "world": world_name, "param": param_name,
                        "config_key": cfg["key"], "setting": setting,
                        "seed": seed,
                    })
    return schedule


def _process_item(client: httpx.Client, item: dict, model_name: str,
                   model_cfg: dict, state: RunState) -> None:
    prompt, expected_keys = _get_prompt(
        item["world"], item["config_key"], item["setting"], item["seed"],
        item["manifest_id"],
    )
    if state.over_cap():
        return
    if model_cfg.get("route") == "gemini_direct":
        resp = _call_gemini_direct(prompt, model_cfg["max_tokens"])
    else:
        resp = _call_openrouter(
            model_cfg["slug"], prompt, model_cfg["max_tokens"],
            model_cfg.get("response_format"), client,
            reasoning=model_cfg.get("reasoning"),
        )
    cost = resp.get("cost_usd", 0.0) or 0.0
    if cost == 0 and not resp.get("error"):
        # Fallback estimate: use safer per-token pricing.
        cost = (resp.get("tokens_in", 0) * 3.0 / 1e6
                + resp.get("tokens_out", 0) * 15.0 / 1e6)

    raw_path = state.raw_dir / f"{item['call_id']}.json"
    with open(raw_path, "w") as f:
        json.dump({
            "call_id": item["call_id"], "model": model_name,
            "text": resp.get("text"), "tokens_in": resp.get("tokens_in", 0),
            "tokens_out": resp.get("tokens_out", 0), "cost_usd": cost,
            "model_served": resp.get("model_served"),
            "latency_s": resp.get("latency_s", 0),
            "timestamp": datetime.now().isoformat(),
            "error": resp.get("error"),
        }, f)

    if resp.get("error"):
        state.record({
            "call_id": item["call_id"], "world": item["world"],
            "param": item["param"], "setting": item["setting"],
            "condition": "relabel", "seed": item["seed"], "model": model_name,
            "parse_success": False, "error": resp["error"],
            "timestamp": datetime.now().isoformat(),
        })
        return
    parsed = _parse_relabel(resp.get("text") or "", expected_keys)
    state.record({
        "call_id": item["call_id"], "world": item["world"],
        "param": item["param"], "setting": item["setting"],
        "condition": "relabel", "seed": item["seed"], "model": model_name,
        "parse_success": parsed["success"], "parse_method": parsed["method"],
        "forecast": parsed["forecast"],
        "missing_keys": parsed["missing"][:5],
        "malformed_keys": parsed["malformed"][:5],
        "tokens_in": resp["tokens_in"], "tokens_out": resp["tokens_out"],
        "cost_usd": cost, "model_served": resp.get("model_served"),
        "latency_s": resp.get("latency_s", 0),
        "timestamp": datetime.now().isoformat(),
    })


def _run_one_model(model_name: str, run_root: Path, spend_cap: float,
                    stop_event: threading.Event,
                    running_total_ref: list[float],
                    parse_pause_threshold: float = 0.85,
                    parse_min_calls: int = 50) -> dict:
    model_cfg = MODELS[model_name]
    checkpoint_path = run_root / "checkpoint.jsonl"
    raw_dir = run_root / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(checkpoint_path, raw_dir, spend_cap=spend_cap)
    schedule = _build_schedule(model_name)
    remaining = [it for it in schedule if it["call_id"] not in state.completed]
    logger.info(
        f"{model_name}: scheduled {len(schedule)}, remaining {len(remaining)} "
        f"(already done: {len(schedule) - len(remaining)})"
    )
    if not remaining:
        return {"model": model_name, "status": "already_done",
                "ok": state.stats["ok"], "fail": state.stats["fail"],
                "error": state.stats["error"], "cost": 0.0}

    workers = model_cfg["workers"]
    total = len(schedule)
    paused_reason = None
    start_cost = state.stats["cost"]

    def _worker(item):
        if stop_event.is_set() or state.over_cap():
            return
        try:
            with httpx.Client() as client:
                _process_item(client, item, model_name, model_cfg, state)
        except Exception as exc:
            logger.error(f"worker exception on {item['call_id']}: {exc}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, it) for it in remaining]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            _ = fut.result()
            if i % 25 == 0 or i == len(futures):
                model_cost = state.stats["cost"] - start_cost
                pct = len(state.completed) / total * 100
                logger.info(
                    f"{model_name}: {len(state.completed)}/{total} "
                    f"({pct:.1f}%) | ok={state.stats['ok']} "
                    f"fail={state.stats['fail']} err={state.stats['error']} "
                    f"| model cost ${model_cost:.2f} "
                    f"| running total ${running_total_ref[0] + model_cost:.2f}"
                )
            attempted = state.stats["ok"] + state.stats["fail"]
            if attempted >= parse_min_calls:
                pr = state.stats["ok"] / attempted
                if pr < parse_pause_threshold:
                    logger.warning(
                        f"{model_name}: parse rate {pr*100:.1f}% below "
                        f"{parse_pause_threshold*100:.0f}% after {attempted} "
                        "attempts — PAUSING model"
                    )
                    paused_reason = (
                        f"parse rate {pr*100:.1f}% < "
                        f"{parse_pause_threshold*100:.0f}%"
                    )
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
            if state.over_cap():
                stop_event.set()
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

    model_cost = state.stats["cost"] - start_cost
    running_total_ref[0] += model_cost
    attempted = state.stats["ok"] + state.stats["fail"]
    pr = state.stats["ok"] / attempted if attempted else None
    logger.info(
        f"{model_name} DONE: ok={state.stats['ok']} fail={state.stats['fail']} "
        f"err={state.stats['error']} model_cost=${model_cost:.2f} "
        f"running=${running_total_ref[0]:.2f}"
        + (f" | PAUSED: {paused_reason}" if paused_reason else "")
    )
    return {
        "model": model_name,
        "status": "paused" if paused_reason else "done",
        "paused_reason": paused_reason,
        "ok": state.stats["ok"], "fail": state.stats["fail"],
        "error": state.stats["error"], "cost": model_cost,
        "parse_rate": pr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="relabel_run_v4")
    parser.add_argument("--manifest",
                       default=str(V2_ROOT
                                   / "outputs/analysis/relabel_v2_preflight/manifest.json"))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--spend-cap", type=float,
                       default=float(os.environ.get("LMM2_SPEND_CAP", "60")))
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set — check .env")

    _load_manifest(Path(args.manifest))
    run_root = V2_ROOT / "outputs" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"run root: {run_root}")

    stop_event = threading.Event()
    running_total = [0.0]
    summaries: list[dict] = []
    models_to_run = args.models or list(MODELS)

    # Optional Gemini direct probe.
    if "Gemini 3.5 Flash" in models_to_run:
        try:
            probe = _call_gemini_direct(
                "Reply with the exact JSON: {\"ok\": true}", 100,
            )
            if probe.get("error"):
                logger.warning(
                    f"Gemini direct probe failed ({probe['error'][:100]}); "
                    "falling back to OpenRouter for Gemini."
                )
                MODELS["Gemini 3.5 Flash"]["route"] = "openrouter"
        except Exception as exc:
            logger.warning(f"Gemini direct probe exception ({exc}); "
                           "using OpenRouter.")
            MODELS["Gemini 3.5 Flash"]["route"] = "openrouter"

    for model_name in models_to_run:
        if model_name not in MODELS:
            logger.warning(f"unknown model: {model_name}")
            continue
        if stop_event.is_set():
            break
        # Credit guard (fetches balance live).
        or_balance = _openrouter_balance()
        or_balance_s = f"${or_balance:.2f}" if or_balance is not None else "unknown"
        logger.info(f"[credit guard] {model_name}: OpenRouter balance {or_balance_s}")
        summary = _run_one_model(
            model_name, run_root, args.spend_cap, stop_event,
            running_total_ref=running_total,
        )
        summaries.append(summary)

    logger.info(f"session total spend: ${running_total[0]:.2f}")
    (run_root / "session_summary.json").write_text(json.dumps({
        "session_total_cost": running_total[0],
        "summaries": summaries,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))


if __name__ == "__main__":
    main()
