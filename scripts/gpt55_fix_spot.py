"""Phase 3 spot check: retry 10 previously-failed GPT-5.5 cells with
the reasoning=minimal fix. Gate: >=9/10 parse success. Does NOT
touch the checkpoint — this is a stand-alone probe."""

import json
import os
import sys
import tempfile
import time
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

from lmm2.relabel_template import (
    build_relabel_prompt, RELABEL_VARIABLE_MAP,
)


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
CONFIG_KEY_BY_PARAM = {
    "phillips_slope": {
        "world1": "phillips_curve.output_slope",
        "world2": "phillips_curve.output_slope",
        "world3": "phillips_curve.output_slope",
        "world4": "price_phillips.output_slope",
    },
    "taylor_phi_pi": {w: "taylor_rule.inflation_coefficient" for w in WORLDS},
    "is_sensitivity": {w: "is_curve.interest_sensitivity" for w in WORLDS},
    "wage_gap_slope": {"world4": "wage_phillips.unemployment_gap_slope"},
}


def build_history(world, config_key, setting, seed):
    import importlib
    wc = WORLDS[world]
    with open(V1_ROOT / wc["config_rel"]) as f:
        cfg = yaml.safe_load(f)
    keys = config_key.split(".")
    target = cfg
    for k in keys[:-1]:
        target = target[k]
    target[keys[-1]] = setting
    tmp = Path(tempfile.gettempdir()) / "lmm2_gpt55_spot"
    tmp.mkdir(exist_ok=True)
    tp = tmp / f"cfg_{world}_{config_key.replace('.','_')}_{setting}_s{seed}.yaml"
    with open(tp, "w") as f:
        yaml.dump(cfg, f)
    sim_mod = importlib.import_module(wc["sim_module"])
    sim_class = getattr(sim_mod, wc["sim_class"])
    sim = sim_class(str(tp), seed=seed)
    return sim.to_dataframe(sim.run(60, sim.get_initial_state()))


def parse_call_id(cid: str):
    parts = cid.split("__")
    world = parts[0]
    param = parts[1]
    setting = float(parts[2])
    seed = int(parts[4][1:])
    return world, param, setting, seed


def call_gpt55(prompt: str) -> dict:
    key = os.environ["OPENROUTER_API_KEY"]
    payload = {
        "model": "openai/gpt-5.5",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 4096,
        "reasoning": {"effort": "minimal"},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/llm-matrix2",
        "X-Title": "LLM-Matrix2 GPT-5.5 spot check",
    }
    t0 = time.time()
    r = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                   headers=headers, json=payload, timeout=180)
    latency = time.time() - t0
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}", "latency_s": latency}
    d = r.json()
    text = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = d.get("usage", {})
    return {
        "text": text,
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "cost_usd": float(usage.get("total_cost", 0.0)),
        "latency_s": latency,
    }


def parse_json(text: str, expected_keys: list[str]) -> dict:
    import re
    if not text:
        return {"success": False, "reason": "empty"}
    cleaned = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return {"success": False, "reason": "no_json"}
    if not isinstance(parsed, dict):
        return {"success": False, "reason": "not_object"}
    missing = [k for k in expected_keys if k not in parsed]
    if missing:
        return {"success": False, "reason": "missing", "n_missing": len(missing),
                "n_present": len(parsed)}
    return {"success": True, "n_keys": len(parsed)}


def main():
    failed = json.load(open("/tmp/gpt55_failed.json"))
    # Pick 10 across worlds/params (all failures are world2/3 phillips_slope
    # in this case, but include variety of settings and seeds).
    picks = [failed[i] for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]]

    results = []
    total_cost = 0.0
    for i, cid in enumerate(picks, 1):
        world, param, setting, seed = parse_call_id(cid)
        cfg_key = CONFIG_KEY_BY_PARAM[param][world]
        hist = build_history(world, cfg_key, setting, seed)
        prompt = build_relabel_prompt(hist, world, WORLDS[world]["variables"], [1, 4, 8])
        expected = [f"{RELABEL_VARIABLE_MAP[v]}_{h}"
                    for v in WORLDS[world]["variables"] for h in [1, 4, 8]]
        print(f"[{i}/10] {cid}")
        resp = call_gpt55(prompt)
        cost = resp.get("cost_usd", 0.0) or 0.0
        total_cost += cost
        if resp.get("error"):
            print(f"     ERROR: {resp['error']}")
            results.append({"call_id": cid, "success": False,
                           "reason": "api_error"})
            continue
        parsed = parse_json(resp.get("text") or "", expected)
        result = {
            "call_id": cid, "world": world, "success": parsed["success"],
            "reason": parsed.get("reason") if not parsed["success"] else None,
            "n_keys": parsed.get("n_keys") or parsed.get("n_present") or 0,
            "tokens_in": resp.get("tokens_in"),
            "tokens_out": resp.get("tokens_out"),
            "cost": cost,
            "text_preview": (resp.get("text") or "")[:180],
        }
        results.append(result)
        print(f"     {'OK' if parsed['success'] else 'FAIL'} "
              f"tokens_out={resp.get('tokens_out')} cost=${cost:.4f} "
              f"reason={parsed.get('reason', '-')}")

    ok = sum(1 for r in results if r["success"])
    print()
    print("=" * 60)
    print(f"Spot check result: {ok}/10 parse success. Cost: ${total_cost:.4f}")
    if ok >= 9:
        print("GATE PASS — safe to resume the full run.")
    else:
        print("GATE FAIL — do NOT resume without further investigation.")
    print("=" * 60)

    out_path = V2_ROOT / "outputs/analysis/gpt55_fix_spot.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"cost": total_cost, "ok": ok, "results": results}, f, indent=2)
    print(f"Details: {out_path}")


if __name__ == "__main__":
    main()
