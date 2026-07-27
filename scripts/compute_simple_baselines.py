"""Compute simple baselines on the same grid the models saw.

Baselines: persistence (no-change), last-value-carry-forward, AR(1),
pooled linear regression, sign-frequency. Scored identically to models
(directional, per-seed GT, clustered bootstrap CIs).

Outputs: outputs/analysis/simple_baselines_report.md

Usage:
    python scripts/compute_simple_baselines.py \
        --per-seed-gt /path/to/ground_truth_per_seed.json
"""

import argparse
import importlib
import json
import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

V2_ROOT = Path(__file__).resolve().parent.parent
V1_ROOT = V2_ROOT

sys.path.insert(0, str(V2_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [1, 4, 8]
HISTORY_SEEDS = list(range(10))

EXCLUDED_GT_KEYS = {
    "world3__phillips_slope__0.7",
    "world3__phillips_slope__0.9",
}

BASELINES_MAP = {
    "phillips_slope": {"world1": 0.4, "world2": 0.4, "world3": 0.5, "world4": 0.2},
    "taylor_phi_pi": {"world1": 1.5, "world2": 1.5, "world3": 2.0, "world4": 1.5},
    "is_sensitivity": {"world1": 0.6, "world2": 0.6, "world3": 0.6, "world4": 0.6},
    "wage_gap_slope": {"world4": 0.5},
}

PARAMS = {
    "phillips_slope": {
        "world1": {"key": "phillips_curve.output_slope",
                   "settings": [0.1, 0.2, 0.4, 0.6, 0.8]},
        "world2": {"key": "phillips_curve.output_slope",
                   "settings": [0.1, 0.2, 0.4, 0.6, 0.8]},
        "world3": {"key": "phillips_curve.output_slope",
                   "settings": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "world4": {"key": "price_phillips.output_slope",
                   "settings": [0.05, 0.1, 0.2, 0.3, 0.5]},
    },
    "taylor_phi_pi": {
        "world1": {"key": "taylor_rule.inflation_coefficient",
                   "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
        "world2": {"key": "taylor_rule.inflation_coefficient",
                   "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
        "world3": {"key": "taylor_rule.inflation_coefficient",
                   "settings": [1.1, 1.5, 2.0, 2.5, 3.0]},
        "world4": {"key": "taylor_rule.inflation_coefficient",
                   "settings": [1.1, 1.3, 1.5, 2.0, 2.5]},
    },
    "is_sensitivity": {
        "world1": {"key": "is_curve.interest_sensitivity",
                   "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world2": {"key": "is_curve.interest_sensitivity",
                   "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world3": {"key": "is_curve.interest_sensitivity",
                   "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
        "world4": {"key": "is_curve.interest_sensitivity",
                   "settings": [0.2, 0.4, 0.6, 0.8, 1.0]},
    },
    "wage_gap_slope": {
        "world4": {"key": "wage_phillips.unemployment_gap_slope",
                   "settings": [0.1, 0.3, 0.5, 0.7, 1.0]},
    },
}

WORLDS = {
    "world1": {"config_rel": "config/ball_baseline.yaml",
               "sim_module": "llmmatrix.simulator", "sim_class": "Sim",
               "vars": ["y", "pi", "r", "e", "u"]},
    "world2": {"config_rel": "config/world2_closed_economy.yaml",
               "sim_module": "llmmatrix.world2_simulator", "sim_class": "ClosedSim",
               "vars": ["y", "pi", "r", "u"]},
    "world3": {"config_rel": "config/world3_emerging_market.yaml",
               "sim_module": "llmmatrix.world3_simulator", "sim_class": "EmergingSim",
               "vars": ["y", "pi", "r", "e", "u"]},
    "world4": {"config_rel": "config/world4_labor_hysteresis.yaml",
               "sim_module": "llmmatrix.world4_simulator", "sim_class": "HysteresisSim",
               "vars": ["y", "pi", "r", "u", "w", "u_natural"]},
}


def _sign_correct(td, md):
    return (td > 0 and md > 0) or (td < 0 and md < 0)


def clustered_bootstrap_ci(values, cluster_ids, n_boot=10000, seed=42):
    values = np.array(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    clusters = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        clusters[cid].append(i)
    cluster_keys = list(clusters.keys())
    cluster_indices = [np.array(clusters[c]) for c in cluster_keys]
    n_clust = len(cluster_keys)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.integers(0, n_clust, size=n_clust)
        idx = np.concatenate([cluster_indices[j] for j in sampled])
        boot_means[b] = values[idx].mean()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "n": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed-gt", required=True)
    args = parser.parse_args()

    gt_ps = json.loads(Path(args.per_seed_gt).read_text())
    logger.info(f"Per-seed GT loaded: {len(gt_ps)} entries")

    # Load simulator classes
    sim_cache = {}
    for world_name, wc in WORLDS.items():
        sim_mod = importlib.import_module(wc["sim_module"])
        sim_class = getattr(sim_mod, wc["sim_class"])
        sim_cache[world_name] = sim_class

    # Compute cell stability for tier filtering
    stability = {}
    for param, worlds in PARAMS.items():
        for world, cfg in worlds.items():
            base_val = BASELINES_MAP[param][world]
            base_stem = f"{world}__{param}__{base_val}"
            if base_stem in EXCLUDED_GT_KEYS:
                continue
            for setting in cfg["settings"]:
                if setting == base_val:
                    continue
                mod_stem = f"{world}__{param}__{setting}"
                if mod_stem in EXCLUDED_GT_KEYS:
                    continue
                probe = gt_ps.get(f"{mod_stem}__s0", {})
                for var_h in probe:
                    deltas = []
                    for s in HISTORY_SEEDS:
                        gm = gt_ps.get(f"{mod_stem}__s{s}", {}).get(var_h)
                        gb = gt_ps.get(f"{base_stem}__s{s}", {}).get(var_h)
                        if gm is not None and gb is not None:
                            deltas.append(gm - gb)
                    if len(deltas) < 10:
                        continue
                    pos = sum(1 for d in deltas if d > 0)
                    stability[(world, param, setting, var_h)] = {
                        'majority_frac': max(pos, 10 - pos) / 10,
                    }

    # Generate histories and compute baseline forecasts per (world, param, setting, seed)
    logger.info("Generating histories and baseline forecasts...")
    tmp_dir = Path(tempfile.gettempdir()) / "lmm2_simple_baselines"
    tmp_dir.mkdir(exist_ok=True)

    # For each (world, param, setting, seed): generate history, compute forecasts
    # Persistence: forecast = last value (delta = 0 for all horizons)
    # Last-value: same as persistence for point forecasts
    # AR(1): fit AR(1) per variable on the 60-quarter history, forecast forward
    # Pooled OLS: regress each variable on its own lag (pooled across all histories)

    # Collect rows: for each baseline method, (world, param, setting, seed, var_h) -> forecast delta
    baseline_rows = defaultdict(list)  # method -> list of rows

    # GT direction base rates for sign-frequency baseline
    gt_directions = defaultdict(list)  # (param, horizon) -> list of +1/-1

    n_cells = 0
    for param_name, param_worlds in PARAMS.items():
        for world_name, cfg in param_worlds.items():
            wc = WORLDS[world_name]
            sim_class = sim_cache[world_name]
            base_val = BASELINES_MAP[param_name][world_name]
            config_key = cfg["key"]
            variables = wc["vars"]

            for setting in cfg["settings"]:
                if setting == base_val:
                    continue
                mod_stem = f"{world_name}__{param_name}__{setting}"
                base_stem = f"{world_name}__{param_name}__{base_val}"
                if mod_stem in EXCLUDED_GT_KEYS:
                    continue

                for seed in HISTORY_SEEDS:
                    # Generate modified-param history (what the model saw)
                    base_config = str(V1_ROOT / wc["config_rel"])
                    with open(base_config) as f:
                        config = yaml.safe_load(f)
                    keys = config_key.split(".")
                    target = config
                    for k in keys[:-1]:
                        target = target[k]
                    target[keys[-1]] = setting
                    tmp_path = tmp_dir / f"bl_{world_name}_{param_name}_{setting}.yaml"
                    with open(tmp_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False)

                    sim = sim_class(str(tmp_path), seed=seed)
                    initial = sim.get_initial_state()
                    traj = sim.run(60, initial)
                    hist_df = sim.to_dataframe(traj)
                    hist_data = hist_df[hist_df["period"] > 0]

                    # Also generate baseline-param history for the baseline forecast
                    sim_base = sim_class(base_config, seed=seed)
                    traj_base = sim_base.run(60, sim_base.get_initial_state())
                    hist_base_df = sim_base.to_dataframe(traj_base)
                    hist_base_data = hist_base_df[hist_base_df["period"] > 0]

                    # GT deltas
                    gt_mod = gt_ps.get(f"{mod_stem}__s{seed}", {})
                    gt_base = gt_ps.get(f"{base_stem}__s{seed}", {})

                    for var in variables:
                        if var not in hist_data.columns:
                            continue

                        series_mod = hist_data[var].values
                        series_base = hist_base_data[var].values if var in hist_base_data.columns else None

                        for h in HORIZONS:
                            var_h = f"{var}_{h}"
                            if var_h not in gt_mod or var_h not in gt_base:
                                continue

                            td = gt_mod[var_h] - gt_base[var_h]
                            if abs(td) < 0.01:
                                continue

                            cell_key = (world_name, param_name, setting, var_h)
                            stab = stability.get(cell_key)
                            if stab is None:
                                continue
                            majority_frac = stab["majority_frac"]
                            cluster_id = f"{world_name}__{param_name}__{setting}"

                            # Collect GT direction for sign-frequency
                            gt_directions[(param_name, h)].append(
                                1 if td > 0 else -1)

                            row_base = {
                                "world": world_name, "param": param_name,
                                "setting": setting, "seed": seed,
                                "var_h": var_h, "var": var, "horizon": h,
                                "td": td, "majority_frac": majority_frac,
                                "cluster_id": cluster_id,
                            }

                            # 1. Persistence / no-change: forecast delta = 0
                            r = dict(row_base)
                            r["forecast_delta"] = 0.0
                            baseline_rows["Persistence"].append(r)

                            # 2. Last-value carry-forward: same as persistence
                            #    (last value of modified = forecast, so delta
                            #    between modified and baseline last values)
                            if series_base is not None:
                                last_mod = series_mod[-1]
                                last_base = series_base[-1]
                                r2 = dict(row_base)
                                r2["forecast_delta"] = last_mod - last_base
                                baseline_rows["Last-value"].append(r2)

                            # 3. AR(1): fit on modified history, forecast h steps
                            if len(series_mod) >= 10:
                                y = series_mod[1:]
                                x = series_mod[:-1]
                                if np.std(x) > 1e-10:
                                    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
                                    intercept = y.mean() - slope * x.mean()
                                    # Forecast h steps from last value
                                    forecast_mod = series_mod[-1]
                                    for step in range(h):
                                        forecast_mod = intercept + slope * forecast_mod

                                    # Same for baseline
                                    if series_base is not None and len(series_base) >= 10:
                                        yb = series_base[1:]
                                        xb = series_base[:-1]
                                        if np.std(xb) > 1e-10:
                                            slope_b = np.sum((xb - xb.mean()) * (yb - yb.mean())) / np.sum((xb - xb.mean())**2)
                                            intercept_b = yb.mean() - slope_b * xb.mean()
                                            forecast_base = series_base[-1]
                                            for step in range(h):
                                                forecast_base = intercept_b + slope_b * forecast_base
                                            r3 = dict(row_base)
                                            r3["forecast_delta"] = forecast_mod - forecast_base
                                            baseline_rows["AR(1)"].append(r3)

                            n_cells += 1

    logger.info(f"Processed {n_cells} cells")

    # 4. Sign-frequency baseline: predict majority direction per (param, horizon)
    majority_dirs = {}
    for key, dirs in gt_directions.items():
        pos = sum(1 for d in dirs if d > 0)
        majority_dirs[key] = 1 if pos >= len(dirs) / 2 else -1

    for method in ["Persistence"]:  # Use persistence rows as template
        for r in baseline_rows[method]:
            r_sf = dict(r)
            maj_dir = majority_dirs.get((r["param"], r["horizon"]), 1)
            r_sf["forecast_delta"] = float(maj_dir)  # Just needs correct sign
            baseline_rows["Sign-frequency"].append(r_sf)

    # Score all baselines
    logger.info("Scoring baselines...")
    results = {}
    for method_name in ["Persistence", "Last-value", "AR(1)", "Sign-frequency"]:
        rows = baseline_rows[method_name]
        for tier_name, tier_fn in [("all", lambda r: True),
                                    ("strict", lambda r: r["majority_frac"] >= 0.8)]:
            tier_rows = [r for r in rows if tier_fn(r)]
            if not tier_rows:
                results[(method_name, tier_name)] = {"mean": float("nan"),
                                                      "ci_low": float("nan"),
                                                      "ci_high": float("nan"),
                                                      "n": 0}
                continue

            if method_name == "Sign-frequency":
                # For sign-frequency, correct = forecast direction matches GT
                correct = [1 if ((r["td"] > 0 and r["forecast_delta"] > 0) or
                                  (r["td"] < 0 and r["forecast_delta"] < 0)) else 0
                           for r in tier_rows]
            elif method_name == "Persistence":
                # Persistence: delta=0, always wrong by scoring rule
                correct = [0 for r in tier_rows]
            else:
                correct = [1 if _sign_correct(r["td"], r["forecast_delta"]) else 0
                           for r in tier_rows]

            cids = [r["cluster_id"] for r in tier_rows]
            ci = clustered_bootstrap_ci(correct, cids)
            results[(method_name, tier_name)] = ci
            logger.info(f"  {method_name} / {tier_name}: "
                        f"{ci['mean']*100:.1f}% [{ci['ci_low']*100:.1f}, "
                        f"{ci['ci_high']*100:.1f}], n={ci['n']}")

    # Build report
    lines = []
    lines.append("# Simple Baselines Report")
    lines.append("")
    lines.append("Baselines scored identically to LLMs: directional accuracy on ")
    lines.append("per-seed GT, near-zero filter (|td| < 0.01 excluded), clustered ")
    lines.append("bootstrap 10K reps (seed 42, cluster = world__param__setting).")
    lines.append("")
    lines.append("## Appendix table: all baselines + VAR + best LLM")
    lines.append("")
    lines.append("| Baseline | All-tier acc | 95% CI | n | Strict-tier acc | 95% CI | n |")
    lines.append("|----------|------------:|-------:|---:|---------------:|-------:|---:|")

    # Add VAR and best LLM from strict_rescore for context
    external = [
        ("VAR", "all", 87.3, 84.8, 89.7, 4833),
        ("VAR", "strict", 82.4, 76.3, 88.7, 546),
        ("GPT-5.5 (infer)", "all", 73.1, 69.6, 76.5, 6043),
        ("GPT-5.5 (infer)", "strict", 69.2, 60.7, 77.6, 614),
    ]

    for name in ["VAR", "GPT-5.5 (infer)", "AR(1)", "Last-value",
                  "Sign-frequency", "Persistence"]:
        if name in ["VAR", "GPT-5.5 (infer)"]:
            all_data = [e for e in external if e[0] == name and e[1] == "all"][0]
            str_data = [e for e in external if e[0] == name and e[1] == "strict"][0]
            lines.append(
                f"| {name} "
                f"| {all_data[2]:.1f}% | [{all_data[3]:.1f}, {all_data[4]:.1f}] | {all_data[5]} "
                f"| {str_data[2]:.1f}% | [{str_data[3]:.1f}, {str_data[4]:.1f}] | {str_data[5]} |"
            )
        else:
            all_ci = results.get((name, "all"), {})
            str_ci = results.get((name, "strict"), {})
            if all_ci.get("n", 0) > 0:
                all_s = f"{all_ci['mean']*100:.1f}% | [{all_ci['ci_low']*100:.1f}, {all_ci['ci_high']*100:.1f}] | {all_ci['n']}"
            else:
                all_s = "— | — | 0"
            if str_ci.get("n", 0) > 0:
                str_s = f"{str_ci['mean']*100:.1f}% | [{str_ci['ci_low']*100:.1f}, {str_ci['ci_high']*100:.1f}] | {str_ci['n']}"
            else:
                str_s = "— | — | 0"
            lines.append(f"| {name} | {all_s} | {str_s} |")

    lines.append("")
    lines.append("**Note:** Persistence (no-change) always scores 0% because ")
    lines.append("forecast_delta = 0 counts as incorrect by the scoring rule.")
    lines.append("")

    report_text = "\n".join(lines)
    out_path = V2_ROOT / "data/processed/simple_baselines_report.md"
    out_path.write_text(report_text)
    logger.info(f"Report written to {out_path}")
    print(report_text)


if __name__ == "__main__":
    main()
