"""
Add VAR baseline and clustered bootstrap CIs to existing v1 results.

This script:
  1. Loads v1's existing prediction checkpoints and synthetic histories.
  2. Computes Monte Carlo ground truth (same as score_all_worlds.py: 2000 paths,
     seed=10000).
  3. Fits Naive, AR(1), and VAR baselines on each world's 60-quarter history.
  4. Scores all baselines against the same ground truth using the same metrics
     (MAE, direction accuracy, CI coverage).
  5. Recomputes LLM bootstrap CIs with both i.i.d. and clustered methods
     side by side.
  6. Outputs updated results to LLM-Matrix2/outputs/v2_results/.

No LLM API calls are made. All model predictions come from v1 checkpoints.
"""

import importlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
V2_ROOT = Path(__file__).resolve().parent.parent
V1_ROOT = V2_ROOT

# Add both v1 and v2 source to path

sys.path.insert(0, str(V2_ROOT / "src"))

from llmmatrix.monte_carlo import run_counterfactual
from llmmatrix.shocks import (
    monetary_tightening_shock, demand_shock, cost_push_shock,
    exchange_rate_shock, wage_shock, productivity_shock,
    labor_supply_shock, prolonged_demand_shock,
)

from lmm2.baselines import naive_baseline, ar1_baseline, var_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# World configurations (mirrors score_all_worlds.py)
# ---------------------------------------------------------------------------

WORLDS = {
    "world1": {
        "config": "config/ball_baseline.yaml",
        "history": "data/simulator/history.csv",
        "checkpoint": "data/benchmark_runs/world1_checkpoint.jsonl",
        "module": "llmmatrix.questions_world1",
        "attr": "WORLD1_QUESTIONS",
        "sim_module": "llmmatrix.simulator",
        "sim_class": "Sim",
        "var_index": {"y": 0, "pi": 1, "r": 2, "e": 3, "u": 4},
        "mc_vars": ["y", "pi", "r", "e", "u"],
        "forecast_vars": ["y", "pi", "r", "e", "u"],
    },
    "world2": {
        "config": "config/world2_closed_economy.yaml",
        "history": "data/world2/history.csv",
        "checkpoint": "data/benchmark_runs/world2_checkpoint.jsonl",
        "module": "llmmatrix.questions_world2",
        "attr": "WORLD2_QUESTIONS",
        "sim_module": "llmmatrix.world2_simulator",
        "sim_class": "ClosedSim",
        "var_index": {"y": 0, "pi": 1, "r": 2, "e": 3, "u": 4},
        "mc_vars": ["y", "pi", "r", "e", "u"],
        "forecast_vars": ["y", "pi", "r", "u"],
    },
    "world3": {
        "config": "config/world3_emerging_market.yaml",
        "history": "data/world3/history.csv",
        "checkpoint": "data/benchmark_runs/world3_checkpoint.jsonl",
        "module": "llmmatrix.questions_world3",
        "attr": "WORLD3_QUESTIONS",
        "sim_module": "llmmatrix.world3_simulator",
        "sim_class": "EmergingSim",
        "var_index": {"y": 0, "pi": 1, "r": 2, "e": 3, "u": 4},
        "mc_vars": ["y", "pi", "r", "e", "u"],
        "forecast_vars": ["y", "pi", "r", "e", "u"],
    },
    "world4": {
        "config": "config/world4_labor_hysteresis.yaml",
        "history": "data/world4/history.csv",
        "checkpoint": "data/benchmark_runs/world4_checkpoint.jsonl",
        "module": "llmmatrix.questions_world4",
        "attr": "WORLD4_QUESTIONS",
        "sim_module": "llmmatrix.world4_simulator",
        "sim_class": "HysteresisSim",
        "var_index": {"y": 0, "pi": 1, "r": 2, "e": 3, "u": 4, "u_natural": 5, "w": 6},
        "mc_vars": ["y", "pi", "r", "e", "u", "u_natural", "w"],
        "forecast_vars": ["y", "pi", "r", "u", "u_natural", "w"],
    },
}

SHOCK_FNS = {
    "monetary": monetary_tightening_shock,
    "demand": demand_shock,
    "cost_push": cost_push_shock,
    "exchange_rate": exchange_rate_shock,
    "wage": wage_shock,
    "productivity": productivity_shock,
    "labor_supply": labor_supply_shock,
    "prolonged_demand": prolonged_demand_shock,
}


# ---------------------------------------------------------------------------
# Helpers (same logic as score_all_worlds.py)
# ---------------------------------------------------------------------------

def classify_direction(value: float, baseline: float, threshold: float = 0.1) -> str:
    diff = value - baseline
    if abs(diff) < threshold:
        return "neutral"
    return "up" if diff > 0 else "down"


def bootstrap_ci_iid(
    values: np.ndarray, n_boot: int = 1000, seed: int = 42,
) -> tuple[float, float]:
    """i.i.d. bootstrap 95% CI on the mean (matches v1's method)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(values[idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_ci_clustered(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Clustered bootstrap 95% CI on the mean.

    Resamples at the cluster level: when a cluster is drawn, all observations
    in that cluster are included. This preserves within-cluster dependence.

    Cluster unit: question_id within a world. All predictions for the same
    question (across variables and horizons) share a cluster because they
    come from the same model call under the same shock scenario.
    """
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    # Build index arrays per cluster
    cluster_to_indices = {}
    for cid in unique_clusters:
        cluster_to_indices[cid] = np.where(cluster_ids == cid)[0]

    boots = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = np.concatenate(
            [cluster_to_indices[c] for c in sampled_clusters]
        )
        boots.append(np.mean(values[sampled_indices]))

    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = V2_ROOT / "outputs" / "v2_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Phase 1: Load data and compute ground truth =====
    all_world_data = {}
    for world_name, wc in WORLDS.items():
        config_path = str(V1_ROOT / wc["config"])
        history_path = str(V1_ROOT / wc["history"])
        checkpoint_path = str(V1_ROOT / wc["checkpoint"])

        if not os.path.exists(checkpoint_path):
            logger.warning(f"{world_name}: no checkpoint, skipping")
            continue

        logger.info(f"Loading {world_name}...")
        history_df = pd.read_csv(history_path)
        last_row = history_df.iloc[-1]
        shock_period = int(last_row["period"]) + 1

        # Load simulator class
        sim_mod = importlib.import_module(wc["sim_module"])
        sim_class = getattr(sim_mod, wc["sim_class"])

        # Load questions
        q_mod = importlib.import_module(wc["module"])
        questions = getattr(q_mod, wc["attr"])
        q_by_id = {q.id: q for q in questions}

        # Compute ground truth via MC (same params as score_all_worlds.py)
        truth = {}
        for q in questions:
            if q.shock_type not in SHOCK_FNS:
                continue
            fn = SHOCK_FNS[q.shock_type]
            max_h = max(q.target_horizons)
            paths = run_counterfactual(
                config_path, history_df,
                lambda fn=fn, m=q.shock_magnitude, p=shock_period: fn(magnitude=m, period=p),
                2000, max_h, 10000, sim_class, wc["mc_vars"],
            )
            qt = {}
            for var in q.target_variables:
                for h in q.target_horizons:
                    key = f"{var}_{h}"
                    vi = wc["var_index"][var]
                    qt[key] = float(np.mean(paths[:, h - 1, vi]))
            truth[q.id] = qt

        # Baseline vals (last observed, used for direction classification)
        baseline_vals = {}
        for var in wc["var_index"]:
            if var in history_df.columns:
                baseline_vals[var] = float(last_row[var])

        # Load LLM predictions
        records = []
        with open(checkpoint_path) as f:
            for line in f:
                r = json.loads(line)
                if (r.get("question_id") != "contamination_probe"
                        and r.get("parse_success")):
                    records.append(r)

        # Fit baselines on history
        forecast_vars = wc["forecast_vars"]
        max_horizon = max(
            h for q in questions for h in q.target_horizons
            if q.shock_type in SHOCK_FNS
        )

        baseline_forecasts = {}
        baseline_forecasts["Naive"] = naive_baseline(
            history_df, forecast_vars, n_horizons=max_horizon,
        )
        baseline_forecasts["AR(1)"] = ar1_baseline(
            history_df, forecast_vars, n_horizons=max_horizon,
        )
        try:
            baseline_forecasts["VAR"] = var_baseline(
                history_df, forecast_vars, n_horizons=max_horizon,
            )
            logger.info(f"  {world_name}: VAR fit succeeded")
        except ValueError as e:
            logger.error(f"  {world_name}: VAR FAILED — {e}")
            baseline_forecasts["VAR"] = None

        all_world_data[world_name] = {
            "questions": questions,
            "q_by_id": q_by_id,
            "truth": truth,
            "baseline_vals": baseline_vals,
            "baseline_forecasts": baseline_forecasts,
            "records": records,
            "history_df": history_df,
            "config": wc,
        }

        logger.info(
            f"  {world_name}: {len(records)} predictions, "
            f"{len(truth)} scored questions, "
            f"baselines: {[k for k, v in baseline_forecasts.items() if v is not None]}"
        )

    # ===== Phase 2: Score everything =====

    # --- Score LLMs (with cluster tracking for bootstrap) ---
    llm_results_per_world = {}
    llm_results_aggregated = defaultdict(lambda: {
        "errors": [], "cluster_ids": [],
        "dir_correct": 0, "dir_total": 0,
        "ci_contains": 0, "ci_total": 0,
    })

    for world_name, data in all_world_data.items():
        world_results = defaultdict(lambda: {
            "errors": [], "cluster_ids": [],
            "dir_correct": 0, "dir_total": 0,
            "ci_contains": 0, "ci_total": 0,
        })

        for r in data["records"]:
            qid = r["question_id"]
            model = r["display_name"]
            fmt = r["prompt_format"]
            key = f"{model}_{fmt}"
            cluster_id = f"{world_name}__{qid}"

            if qid not in data["truth"]:
                continue
            preds = r.get("parsed_predictions", {})
            q = data["q_by_id"].get(qid)
            if not q:
                continue

            for pred_key, truth_val in data["truth"][qid].items():
                var = pred_key.rsplit("_", 1)[0]

                if pred_key not in preds or not isinstance(preds[pred_key], dict):
                    continue
                p = preds[pred_key]
                pt = p.get("point")
                ci_lo = p.get("ci_low")
                ci_hi = p.get("ci_high")
                if not isinstance(pt, (int, float)):
                    continue

                error = abs(pt - truth_val)
                world_results[key]["errors"].append(error)
                world_results[key]["cluster_ids"].append(cluster_id)

                # Also accumulate into aggregated
                llm_results_aggregated[key]["errors"].append(error)
                llm_results_aggregated[key]["cluster_ids"].append(cluster_id)

                if var in data["baseline_vals"]:
                    truth_dir = classify_direction(truth_val, data["baseline_vals"][var])
                    pred_dir = classify_direction(pt, data["baseline_vals"][var])
                    world_results[key]["dir_total"] += 1
                    llm_results_aggregated[key]["dir_total"] += 1
                    if pred_dir == truth_dir:
                        world_results[key]["dir_correct"] += 1
                        llm_results_aggregated[key]["dir_correct"] += 1

                if isinstance(ci_lo, (int, float)) and isinstance(ci_hi, (int, float)):
                    world_results[key]["ci_total"] += 1
                    llm_results_aggregated[key]["ci_total"] += 1
                    if ci_lo <= truth_val <= ci_hi:
                        world_results[key]["ci_contains"] += 1
                        llm_results_aggregated[key]["ci_contains"] += 1

        llm_results_per_world[world_name] = dict(world_results)

    # --- Score baselines ---
    baseline_results_per_world = {}
    baseline_results_aggregated = defaultdict(lambda: {
        "errors": [], "cluster_ids": [],
        "dir_correct": 0, "dir_total": 0,
        "ci_contains": 0, "ci_total": 0,
    })

    for world_name, data in all_world_data.items():
        world_baseline_results = {}

        for bl_name, bl_forecast in data["baseline_forecasts"].items():
            if bl_forecast is None:
                world_baseline_results[bl_name] = {
                    "status": "FAILED",
                    "errors": [], "cluster_ids": [],
                    "dir_correct": 0, "dir_total": 0,
                    "ci_contains": 0, "ci_total": 0,
                }
                baseline_results_aggregated[bl_name]["status"] = "PARTIAL_FAIL"
                continue

            bl_result = {
                "status": "OK",
                "errors": [], "cluster_ids": [],
                "dir_correct": 0, "dir_total": 0,
                "ci_contains": 0, "ci_total": 0,
            }

            for q in data["questions"]:
                if q.id not in data["truth"]:
                    continue
                cluster_id = f"{world_name}__{q.id}"

                for var in q.target_variables:
                    for h in q.target_horizons:
                        pred_key = f"{var}_{h}"
                        if pred_key not in data["truth"][q.id]:
                            continue
                        if pred_key not in bl_forecast:
                            continue

                        truth_val = data["truth"][q.id][pred_key]
                        bl_pred = bl_forecast[pred_key]
                        pt = bl_pred["point"]
                        ci_lo = bl_pred["ci_low"]
                        ci_hi = bl_pred["ci_high"]

                        error = abs(pt - truth_val)
                        bl_result["errors"].append(error)
                        bl_result["cluster_ids"].append(cluster_id)
                        baseline_results_aggregated[bl_name]["errors"].append(error)
                        baseline_results_aggregated[bl_name]["cluster_ids"].append(cluster_id)

                        if var in data["baseline_vals"]:
                            truth_dir = classify_direction(
                                truth_val, data["baseline_vals"][var],
                            )
                            pred_dir = classify_direction(
                                pt, data["baseline_vals"][var],
                            )
                            bl_result["dir_total"] += 1
                            baseline_results_aggregated[bl_name]["dir_total"] += 1
                            if pred_dir == truth_dir:
                                bl_result["dir_correct"] += 1
                                baseline_results_aggregated[bl_name]["dir_correct"] += 1

                        bl_result["ci_total"] += 1
                        baseline_results_aggregated[bl_name]["ci_total"] += 1
                        if ci_lo <= truth_val <= ci_hi:
                            bl_result["ci_contains"] += 1
                            baseline_results_aggregated[bl_name]["ci_contains"] += 1

            world_baseline_results[bl_name] = bl_result

        baseline_results_per_world[world_name] = world_baseline_results

    # ===== Phase 3: Build output tables =====

    def format_entry(errors, cluster_ids, dir_correct, dir_total,
                     ci_contains, ci_total, status="OK"):
        """Build a results dict with both i.i.d. and clustered bootstrap CIs."""
        if not errors or status == "FAILED":
            return {
                "status": status,
                "mae": None, "n": 0,
                "ci_iid_low": None, "ci_iid_high": None,
                "ci_clust_low": None, "ci_clust_high": None,
                "dir_acc": None, "ci_cov": None,
            }

        errors_arr = np.array(errors)
        cluster_arr = np.array(cluster_ids)
        mae = float(np.mean(errors_arr))

        ci_iid_low, ci_iid_high = bootstrap_ci_iid(errors_arr)
        ci_clust_low, ci_clust_high = bootstrap_ci_clustered(
            errors_arr, cluster_arr,
        )

        dir_acc = (dir_correct / dir_total * 100) if dir_total > 0 else None
        ci_cov = (ci_contains / ci_total * 100) if ci_total > 0 else None

        return {
            "status": "OK",
            "mae": mae,
            "n": len(errors),
            "ci_iid_low": ci_iid_low,
            "ci_iid_high": ci_iid_high,
            "ci_clust_low": ci_clust_low,
            "ci_clust_high": ci_clust_high,
            "dir_acc": dir_acc,
            "ci_cov": ci_cov,
        }

    # --- Aggregated table ---
    aggregated_table = {}

    # LLMs
    for key, r in llm_results_aggregated.items():
        aggregated_table[key] = format_entry(
            r["errors"], r["cluster_ids"],
            r["dir_correct"], r["dir_total"],
            r["ci_contains"], r["ci_total"],
        )

    # Baselines
    for bl_name, r in baseline_results_aggregated.items():
        status = r.get("status", "OK")
        aggregated_table[f"BASELINE_{bl_name}"] = format_entry(
            r["errors"], r["cluster_ids"],
            r["dir_correct"], r["dir_total"],
            r["ci_contains"], r["ci_total"],
            status=status,
        )

    # --- Per-world tables ---
    per_world_table = {}
    for world_name in all_world_data:
        world_table = {}

        # LLMs
        for key, r in llm_results_per_world.get(world_name, {}).items():
            world_table[key] = format_entry(
                r["errors"], r["cluster_ids"],
                r["dir_correct"], r["dir_total"],
                r["ci_contains"], r["ci_total"],
            )

        # Baselines
        for bl_name, r in baseline_results_per_world.get(world_name, {}).items():
            status = r.get("status", "OK")
            world_table[f"BASELINE_{bl_name}"] = format_entry(
                r["errors"], r["cluster_ids"],
                r["dir_correct"], r["dir_total"],
                r["ci_contains"], r["ci_total"],
                status=status,
            )

        per_world_table[world_name] = world_table

    # ===== Phase 4: Print and save =====

    # --- Print aggregated table ---
    print(f"\n{'='*120}")
    print("AGGREGATED RESULTS (all worlds) — v2 with VAR baseline + dual bootstrap CIs")
    print(f"{'='*120}")
    print(f"{'Name':<30} {'MAE':>7} {'95% CI (i.i.d.)':>20} {'95% CI (clustered)':>22} {'DirAcc':>7} {'CICov':>7} {'n':>6}")
    print("-" * 120)

    # Sort: baselines first, then LLMs by MAE
    baseline_entries = sorted(
        [(k, v) for k, v in aggregated_table.items() if k.startswith("BASELINE_")],
        key=lambda x: x[1]["mae"] if x[1]["mae"] is not None else 999,
    )
    llm_entries = sorted(
        [(k, v) for k, v in aggregated_table.items() if not k.startswith("BASELINE_")],
        key=lambda x: x[1]["mae"] if x[1]["mae"] is not None else 999,
    )

    for name, entry in baseline_entries + llm_entries:
        if entry["status"] == "FAILED" or entry["mae"] is None:
            print(f"{name:<30} {'FAILED — see per-world notes':>90}")
            continue
        ci_iid = f"[{entry['ci_iid_low']:.3f}, {entry['ci_iid_high']:.3f}]"
        ci_clust = f"[{entry['ci_clust_low']:.3f}, {entry['ci_clust_high']:.3f}]"
        dir_str = f"{entry['dir_acc']:.0f}%" if entry['dir_acc'] is not None else "—"
        cov_str = f"{entry['ci_cov']:.0f}%" if entry['ci_cov'] is not None else "—"
        print(
            f"{name:<30} {entry['mae']:>7.3f} {ci_iid:>20} {ci_clust:>22} "
            f"{dir_str:>7} {cov_str:>7} {entry['n']:>6}"
        )

    # --- Print per-world baseline table ---
    for world_name in sorted(per_world_table):
        wt = per_world_table[world_name]
        bl_keys = [k for k in wt if k.startswith("BASELINE_")]
        if not bl_keys:
            continue
        print(f"\n--- {world_name} baselines ---")
        print(f"{'Baseline':<20} {'MAE':>7} {'CI (i.i.d.)':>20} {'CI (clustered)':>22} {'DirAcc':>7} {'CICov':>7} {'n':>6}")
        for bk in sorted(bl_keys):
            e = wt[bk]
            if e["status"] == "FAILED":
                print(f"{bk:<20} FAILED — VAR could not fit this world")
                continue
            ci_iid = f"[{e['ci_iid_low']:.3f}, {e['ci_iid_high']:.3f}]"
            ci_clust = f"[{e['ci_clust_low']:.3f}, {e['ci_clust_high']:.3f}]"
            dir_str = f"{e['dir_acc']:.0f}%" if e['dir_acc'] is not None else "—"
            cov_str = f"{e['ci_cov']:.0f}%" if e['ci_cov'] is not None else "—"
            print(f"{bk:<20} {e['mae']:>7.3f} {ci_iid:>20} {ci_clust:>22} {dir_str:>7} {cov_str:>7} {e['n']:>6}")

    # --- CI width comparison ---
    print(f"\n{'='*120}")
    print("BOOTSTRAP CI WIDTH COMPARISON: i.i.d. vs clustered (MAE metric)")
    print("Cluster unit: question_id within world (all predictions for one question share a cluster)")
    print(f"{'='*120}")
    print(f"{'Name':<30} {'i.i.d. width':>14} {'clustered width':>16} {'ratio':>8}")
    print("-" * 70)

    for name, entry in sorted(aggregated_table.items(),
                               key=lambda x: x[0]):
        if entry["mae"] is None:
            continue
        iid_w = entry["ci_iid_high"] - entry["ci_iid_low"]
        clust_w = entry["ci_clust_high"] - entry["ci_clust_low"]
        ratio = clust_w / iid_w if iid_w > 0 else float("nan")
        print(f"{name:<30} {iid_w:>14.4f} {clust_w:>16.4f} {ratio:>8.2f}x")

    # --- Save to JSON ---
    output = {
        "description": (
            "v2 results: v1 LLM forecasts + VAR baseline + dual bootstrap CIs. "
            "No new model runs. VAR, Naive, AR(1) scored on same ground truth. "
            "Cluster unit for clustered bootstrap: question_id within world."
        ),
        "aggregated": aggregated_table,
        "per_world": per_world_table,
    }

    results_path = output_dir / "full_results_v2.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # --- Save paper table (best-of-format per model, ranked) ---
    # Group by model (strip format suffix), pick best MAE
    model_best = {}
    for key, entry in aggregated_table.items():
        if key.startswith("BASELINE_") or entry["mae"] is None:
            continue
        # key is "ModelName_format"
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        model_name = parts[0]
        if model_name not in model_best or entry["mae"] < model_best[model_name]["mae"]:
            model_best[model_name] = {**entry, "source_key": key}

    paper_table = sorted(model_best.items(), key=lambda x: x[1]["mae"])

    # Add baselines
    for bl_key in ["BASELINE_Naive", "BASELINE_AR(1)", "BASELINE_VAR"]:
        if bl_key in aggregated_table and aggregated_table[bl_key]["mae"] is not None:
            model_best[bl_key] = aggregated_table[bl_key]

    paper_output = {
        "table1_v2": [
            {
                "model": name,
                "mae": entry["mae"],
                "ci_iid": [entry["ci_iid_low"], entry["ci_iid_high"]],
                "ci_clustered": [entry["ci_clust_low"], entry["ci_clust_high"]],
                "dir_acc": entry["dir_acc"],
                "ci_cov": entry["ci_cov"],
                "n": entry["n"],
            }
            for name, entry in paper_table
        ],
        "baselines_v2": {
            bl_name: {
                "mae": aggregated_table[f"BASELINE_{bl_name}"]["mae"],
                "ci_iid": [
                    aggregated_table[f"BASELINE_{bl_name}"]["ci_iid_low"],
                    aggregated_table[f"BASELINE_{bl_name}"]["ci_iid_high"],
                ],
                "ci_clustered": [
                    aggregated_table[f"BASELINE_{bl_name}"]["ci_clust_low"],
                    aggregated_table[f"BASELINE_{bl_name}"]["ci_clust_high"],
                ],
                "dir_acc": aggregated_table[f"BASELINE_{bl_name}"]["dir_acc"],
                "ci_cov": aggregated_table[f"BASELINE_{bl_name}"]["ci_cov"],
                "n": aggregated_table[f"BASELINE_{bl_name}"]["n"],
            }
            for bl_name in ["Naive", "AR(1)", "VAR"]
            if aggregated_table.get(f"BASELINE_{bl_name}", {}).get("mae") is not None
        },
    }

    paper_path = output_dir / "paper_tables_v2.json"
    with open(paper_path, "w") as f:
        json.dump(paper_output, f, indent=2, default=str)
    logger.info(f"Paper tables saved to {paper_path}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
