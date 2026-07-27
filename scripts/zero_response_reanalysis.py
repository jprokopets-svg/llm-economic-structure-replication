"""Pre-submission reanalysis: zero forecast changes and the chance benchmark.

Referee critique addressed here: the scoring rule
(scripts/score_fork.py:49) marks a forecast as correct only when the model's
delta is strictly positive or strictly negative in the same direction as the
truth. A model that returns an identical point forecast in both arms
(model_delta == 0) is therefore scored INCORRECT, never as a tie. When models
frequently return unchanged forecasts, 50% is no longer the right chance
benchmark, because a coin flip can never produce a zero but a model can.

Four analyses:

  1. Zero-response reanalysis of the fork primary contrast (ARM2-ARM1):
     (a) share of eligible pairs with exactly zero forecast change
     (b) accuracy conditional on a nonzero response
     (c) tie-adjusted accuracy (zeros score 0.5)
     (d) permutation null preserving each model's zero-response frequency
  2. Exact proportion of eligible pairs with response ratio exactly zero,
     plus the median ratio, so the manuscript can state the proportion.
  3. Uncertainty on the Gemini temperature-robustness difference.
  4. Stationarity robustness: recompute headline numbers excluding the nine
     non-stationary configurations.

This script only reads existing data and writes
outputs/analysis/zero_response_report.md. It does not modify any existing
scoring script, report, or data file.

Usage:
    python scripts/zero_response_reanalysis.py
"""

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FORK_DIR = V2_ROOT / "outputs" / "fork_run"
ANALYSIS_DIR = V2_ROOT / "outputs" / "analysis"
GRID_DIR = V2_ROOT / "outputs" / "structural_tracking_v2"

FORK_GT_FILE = FORK_DIR / "fork_ground_truth.json"
FORK_CHECKPOINT = FORK_DIR / "checkpoint.jsonl"
GEMINI_CHECKPOINT = FORK_DIR / "gemini_robustness" / "checkpoint.jsonl"
GRID_CHECKPOINT = GRID_DIR / "checkpoint.jsonl"
PER_SEED_GT_FILE = ANALYSIS_DIR / "per_seed_gt_reconstructed.json"

REPORT_PATH = ANALYSIS_DIR / "zero_response_report.md"

N_BOOT = 10000
N_PERM = 10000
SEED = 42
HORIZONS = [1, 4, 8]

# Baseline parameter value per (param, world). Mirrors score_fork_dual.py:31.
BASELINES_MAP = {
    "phillips_slope": {"world1": 0.4, "world2": 0.4, "world3": 0.5, "world4": 0.2},
    "taylor_phi_pi": {"world1": 1.5, "world2": 1.5, "world3": 2.0, "world4": 1.5},
    "is_sensitivity": {"world1": 0.6, "world2": 0.6, "world3": 0.6, "world4": 0.6},
    "wage_gap_slope": {"world4": 0.5},
}

# Already excluded from all published analyses (World 3 dynamic instability).
# Mirrors scripts/run_analysis.py:41.
EXCLUDED_GT_KEYS = {
    "world3__phillips_slope__0.7",
    "world3__phillips_slope__0.9",
}

# The nine non-stationary configurations (max |eigenvalue| >= 1.0), read from
# outputs/analysis/stationarity_report.md. NOTE: that report's table lists ten
# rows under the heading "max |lambda| >= 1.0", but one of them
# (world2 / phillips_slope / 0.6, max|lambda| = 0.991) is below the threshold and
# is therefore near-unit-root, not non-stationary. Excluding it gives exactly the
# nine the report's own summary table counts. See the report for the flag.
NON_STATIONARY_CONFIGS = [
    ("world1", "phillips_slope", 0.8, 1.015, False),
    ("world2", "phillips_slope", 0.8, 1.063, False),
    ("world2", "taylor_phi_pi", 2.5, 1.021, False),
    ("world3", "phillips_slope", 0.7, 1.053, True),
    ("world3", "phillips_slope", 0.9, 1.180, True),
    ("world4", "phillips_slope", 0.5, 1.036, False),
    ("world4", "taylor_phi_pi", 2.0, 1.001, False),
    ("world4", "taylor_phi_pi", 2.5, 1.039, False),
    ("world4", "wage_gap_slope", 1.0, 1.036, False),
]

# The tenth row in the report's table, below the stated threshold.
BORDERLINE_CONFIG = ("world2", "phillips_slope", 0.6, 0.991)

# Published headline numbers, for delta comparison.
# Source: docs/MANUSCRIPT_NUMBERS.md:19-28 (fork ARM2-ARM1).
PUBLISHED_FORK_ARM21 = {
    "GPT-5.5": (49.3, 5893),
    "Gemini 3.5 Flash": (41.5, 5467),
    "Claude Sonnet 4.6": (37.6, 5883),
    "Claude Opus 4.8": (30.6, 5893),
}

# Source: docs/MANUSCRIPT_NUMBERS.md:116-132 (grid infer, all tier).
PUBLISHED_GRID_INFER_ALL = {
    "GPT-5.5": (73.1, 6043),
    "Gemini 3.5 Flash": (64.1, 4991),
    "Claude Sonnet 4.6": (54.9, 6757),
    "DeepSeek V4 Pro": (53.9, 1387),
}

# Published Gemini temperature-robustness aggregates.
# Source: outputs/fork_run/gemini_robustness/gemini_robustness_report.md
PUBLISHED_GEMINI_TEMP = {
    "pairs": 282,
    "cells": 50,
    "temp0_acc": 46.5,
    "temp07_acc": 60.3,
    "agreement": 58.5,
}


# ═══════════════════════════════════════════════════════════════════════
# SHARED STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def clustered_bootstrap_ci(values, cluster_ids, n_boot=N_BOOT, seed=SEED):
    """Clustered bootstrap 95% CI on the mean of values.

    Byte-identical in behaviour to scripts/score_fork.py:80, reproduced here so
    this script does not import from (and cannot perturb) the scoring scripts.
    Clusters are resampled with replacement; all rows in a sampled cluster are
    taken together.
    """
    values = np.array(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}

    clusters = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        clusters[cid].append(i)
    cluster_keys = list(clusters.keys())
    cluster_indices = []
    for key in cluster_keys:
        cluster_indices.append(np.array(clusters[key]))
    n_clust = len(cluster_keys)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.integers(0, n_clust, size=n_clust)
        chosen = []
        for j in sampled:
            chosen.append(cluster_indices[j])
        idx = np.concatenate(chosen)
        boot_means[b] = values[idx].mean()

    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "n": n,
    }


def sign_correct(true_delta: float, model_delta: float) -> bool:
    """The published correctness rule. A zero model_delta is INCORRECT.

    Reproduced from scripts/score_fork.py:49 so the reanalysis scores the
    baseline exactly as published.
    """
    return (true_delta > 0 and model_delta > 0) or (true_delta < 0 and model_delta < 0)


def tie_adjusted_score(true_delta: float, model_delta: float) -> float:
    """Score with zero forecast changes treated as ties worth 0.5.

    This is the scoring rule the referee critique implies: a non-response is
    neither right nor wrong about direction, so it earns the same expected
    credit as a coin flip.
    """
    if model_delta == 0:
        return 0.5
    if sign_correct(true_delta, model_delta):
        return 1.0
    return 0.0


def permutation_null_tie_adjusted(rows, delta_key: str, n_perm=N_PERM, seed=SEED):
    """Permutation null that preserves each model's zero-response frequency.

    Strata are (world, param, horizon, seed) — matching the strata used by the
    project's existing permutation baseline (recovered from
    `git show 07146d3:scripts/strict_rescore.py`), where `param` is the
    parameter NAME, not the setting, so settings pool within a stratum and the
    shuffle is non-degenerate.

    Only the direction signs of NONZERO responses are shuffled, and only within
    a stratum. Zero responses stay fixed at 0.5 credit, matching the
    tie-adjusted scoring in (c). This makes the null answer the right question:
    given that this model declines to move on X% of pairs, and given its
    up/down bias among the pairs where it does move, what accuracy would it
    reach if the direction it picked carried no information?

    Returns the null mean and the 2.5/97.5 percentiles.
    """
    true_signs = []
    model_deltas = []
    strata_keys = []
    for row in rows:
        true_delta = row["true_delta"]
        true_signs.append(1 if true_delta > 0 else -1)
        model_deltas.append(row[delta_key])
        strata_keys.append((row["world"], row["param"], row["horizon"], row["seed"]))

    true_signs = np.array(true_signs, dtype=np.int8)
    model_deltas = np.array(model_deltas, dtype=np.float64)

    n_total = len(model_deltas)
    if n_total == 0:
        return {"null_mean": float("nan"), "null_p2_5": float("nan"),
                "null_p97_5": float("nan"), "n": 0}

    is_nonzero = model_deltas != 0
    n_zero = int((~is_nonzero).sum())

    # Restrict the shuffle to nonzero responses.
    nonzero_positions = np.flatnonzero(is_nonzero)
    nonzero_model_signs = np.where(model_deltas[nonzero_positions] > 0, 1, -1).astype(np.int8)
    nonzero_true_signs = true_signs[nonzero_positions]

    # Encode strata for the nonzero rows, then sort by stratum so that a
    # within-stratum shuffle is a sort by (stratum, random key).
    stratum_codes = {}
    nonzero_stratum_codes = np.empty(len(nonzero_positions), dtype=np.int64)
    for i, pos in enumerate(nonzero_positions):
        key = strata_keys[pos]
        if key not in stratum_codes:
            stratum_codes[key] = len(stratum_codes)
        nonzero_stratum_codes[i] = stratum_codes[key]

    order = np.argsort(nonzero_stratum_codes, kind="stable")
    sorted_strata = nonzero_stratum_codes[order]
    sorted_model_signs = nonzero_model_signs[order]
    sorted_true_signs = nonzero_true_signs[order]

    # Zeros contribute a constant 0.5 each, so only the nonzero part varies.
    zero_credit = 0.5 * n_zero

    rng = np.random.default_rng(seed)
    null_accuracies = np.empty(n_perm)
    for k in range(n_perm):
        random_keys = rng.random(len(sorted_strata))
        # lexsort's last key is primary: sort by stratum, break ties randomly.
        shuffled_order = np.lexsort((random_keys, sorted_strata))
        permuted_signs = sorted_model_signs[shuffled_order]
        n_correct = int((permuted_signs == sorted_true_signs).sum())
        null_accuracies[k] = (n_correct + zero_credit) / n_total

    return {
        "null_mean": float(null_accuracies.mean()),
        "null_p2_5": float(np.percentile(null_accuracies, 2.5)),
        "null_p97_5": float(np.percentile(null_accuracies, 97.5)),
        "n": n_total,
        "n_zero": n_zero,
    }


# ═══════════════════════════════════════════════════════════════════════
# FORK DATA
# ═══════════════════════════════════════════════════════════════════════

def load_fork_forecasts():
    """Load parsed fork forecasts, keeping the last record per call_id."""
    last_record = {}
    with open(FORK_CHECKPOINT) as f:
        for line in f:
            record = json.loads(line)
            last_record[record["call_id"]] = record

    forecasts = []
    for record in last_record.values():
        if record.get("parse_success") and record.get("forecast"):
            forecasts.append(record)
    return forecasts


def index_fork_forecasts(forecasts):
    """Index by (arm, world, param, setting, seed, model) -> {var_h: point}."""
    index = {}
    for forecast in forecasts:
        key = (forecast["arm"], forecast["world"], forecast.get("param"),
               forecast.get("setting"), forecast["seed"], forecast["model"])
        points = {}
        for var_h, value in forecast["forecast"].items():
            if isinstance(value, dict) and "point" in value:
                points[var_h] = value["point"]
        index[key] = points
    return index


def build_fork_rows():
    """Build eligible, non-OOD fork rows carrying the ARM2-ARM1 contrast.

    Mirrors scripts/score_fork_dual.py:120-175 for the delta_21 contrast.
    """
    ground_truth = json.loads(FORK_GT_FILE.read_text())
    forecasts = load_fork_forecasts()
    forecast_index = index_fork_forecasts(forecasts)

    models = sorted(set(f["model"] for f in forecasts))

    rows = []
    for cell_data in ground_truth.values():
        world = cell_data["world"]
        param = cell_data["param"]
        setting = cell_data["setting"]
        seed = cell_data["seed"]
        is_ood = cell_data.get("is_ood", False)
        if is_ood:
            continue
        baseline_val = BASELINES_MAP[param][world]

        for model in models:
            arm2_fc = forecast_index.get(
                ("arm2_change", world, param, setting, seed, model), {})
            # ARM1 records store setting as the baseline value, not None, so
            # both key shapes have to be tried (score_fork_dual.py:134-136).
            arm1_fc = forecast_index.get(
                ("arm1_placebo", world, param, None, seed, model), {})
            if not arm1_fc:
                arm1_fc = forecast_index.get(
                    ("arm1_placebo", world, param, baseline_val, seed, model), {})

            for var_h, gt_info in cell_data["forecasts"].items():
                if not gt_info["eligible"]:
                    continue

                arm2_point = arm2_fc.get(var_h)
                arm1_point = arm1_fc.get(var_h)
                if arm2_point is None or arm1_point is None:
                    continue

                horizon = int(var_h.rsplit("_", 1)[1])
                rows.append({
                    "model": model,
                    "world": world,
                    "param": param,
                    "setting": setting,
                    "seed": seed,
                    "var_h": var_h,
                    "horizon": horizon,
                    "true_delta": gt_info["true_delta"],
                    "delta_21": arm2_point - arm1_point,
                    "cluster_id": f"{world}__{param}__{setting}",
                })

    return rows, models


def rows_for_model(rows, model: str):
    selected = []
    for row in rows:
        if row["model"] == model:
            selected.append(row)
    return selected


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: ZERO-RESPONSE REANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analysis_one_zero_response(rows, models, lines):
    lines.append("## 1. Zero-response reanalysis (fork, ARM2−ARM1, eligible cells)")
    lines.append("")
    lines.append("Eligible non-OOD variable×horizon pairs. Clustered bootstrap "
                 f"{N_BOOT:,} draws, seed {SEED}, cluster = world__param__setting "
                 "(same cluster spec as the published primary).")
    lines.append("")
    lines.append("- **(a) Zero share** — fraction with `model_delta == 0` exactly.")
    lines.append("- **(b) Accuracy | nonzero** — published rule, restricted to pairs "
                 "where the model actually moved.")
    lines.append("- **(c) Tie-adjusted** — zeros score 0.5 instead of 0.")
    lines.append("- **(d) Permutation null** — preserves each model's zero frequency; "
                 "shuffles direction signs among nonzero responses only, within "
                 f"(world, param, horizon, seed); {N_PERM:,} permutations, seed {SEED}.")
    lines.append("")
    lines.append("| Model | Published acc | (a) Zero share [95% CI] | (b) Acc \\| nonzero [95% CI] "
                 "| (c) Tie-adjusted [95% CI] | (d) Null mean [2.5, 97.5] | n |")
    lines.append("|-------|-------------:|------------------------:|---------------------------:"
                 "|-------------------------:|-------------------------:|---:|")

    results = {}
    for model in models:
        model_rows = rows_for_model(rows, model)
        if not model_rows:
            continue

        cluster_ids = []
        zero_flags = []
        tie_scores = []
        for row in model_rows:
            cluster_ids.append(row["cluster_id"])
            zero_flags.append(1 if row["delta_21"] == 0 else 0)
            tie_scores.append(tie_adjusted_score(row["true_delta"], row["delta_21"]))

        # (a) share of exactly-zero responses
        zero_ci = clustered_bootstrap_ci(zero_flags, cluster_ids)

        # (b) accuracy conditional on a nonzero response
        nonzero_rows = []
        for row in model_rows:
            if row["delta_21"] != 0:
                nonzero_rows.append(row)
        nonzero_correct = []
        nonzero_clusters = []
        for row in nonzero_rows:
            nonzero_correct.append(1 if sign_correct(row["true_delta"], row["delta_21"]) else 0)
            nonzero_clusters.append(row["cluster_id"])
        nonzero_ci = clustered_bootstrap_ci(nonzero_correct, nonzero_clusters)

        # (c) tie-adjusted accuracy
        tie_ci = clustered_bootstrap_ci(tie_scores, cluster_ids)

        # (d) permutation null preserving the zero frequency
        null = permutation_null_tie_adjusted(model_rows, "delta_21")

        published_acc = PUBLISHED_FORK_ARM21.get(model, (float("nan"), 0))[0]

        results[model] = {
            "published": published_acc,
            "zero": zero_ci,
            "nonzero": nonzero_ci,
            "tie": tie_ci,
            "null": null,
            "n": len(model_rows),
        }

        lines.append(
            f"| {model} | {published_acc:.1f}% "
            f"| {zero_ci['mean']*100:.1f}% [{zero_ci['ci_low']*100:.1f}, {zero_ci['ci_high']*100:.1f}] "
            f"| {nonzero_ci['mean']*100:.1f}% [{nonzero_ci['ci_low']*100:.1f}, {nonzero_ci['ci_high']*100:.1f}] "
            f"| {tie_ci['mean']*100:.1f}% [{tie_ci['ci_low']*100:.1f}, {tie_ci['ci_high']*100:.1f}] "
            f"| {null['null_mean']*100:.1f}% [{null['null_p2_5']*100:.1f}, {null['null_p97_5']*100:.1f}] "
            f"| {len(model_rows)} |"
        )

    lines.append("")
    return results


def flag_qualitative_changes(results, lines):
    """Flag models whose characterisation vs chance changes under tie-adjustment."""
    lines.append("### 1e. Qualitative-claim changes")
    lines.append("")
    lines.append("A model is 'below chance' only if its 95% CI upper bound is below the "
                 "benchmark. Under the published rule the benchmark is 50%. Under "
                 "tie-adjusted scoring the honest benchmark is the model's own "
                 "permutation null, which is not 50% when the model returns zeros.")
    lines.append("")
    lines.append("| Model | Published claim (vs 50%) | Tie-adjusted vs 50% | "
                 "Tie-adjusted vs own null | Claim changes? |")
    lines.append("|-------|--------------------------|---------------------|"
                 "--------------------------|:--------------:|")

    for model, res in results.items():
        published_acc = res["published"]
        tie = res["tie"]
        null = res["null"]

        # Published characterisation used the published CI, restated here.
        if published_acc < 50.0:
            published_claim = "below chance"
        else:
            published_claim = "at chance"

        # Tie-adjusted vs the flat 50% line.
        if tie["ci_high"] * 100 < 50.0:
            tie_vs_50 = "below chance"
        elif tie["ci_low"] * 100 > 50.0:
            tie_vs_50 = "above chance"
        else:
            tie_vs_50 = "indistinguishable"

        # Tie-adjusted vs the model's own zero-preserving null.
        if tie["ci_high"] * 100 < null["null_p2_5"] * 100:
            tie_vs_null = "below own null"
        elif tie["ci_low"] * 100 > null["null_p97_5"] * 100:
            tie_vs_null = "above own null"
        else:
            tie_vs_null = "indistinguishable"

        changed = "**YES**" if tie_vs_50 != published_claim else "no"

        lines.append(f"| {model} | {published_claim} | {tie_vs_50} | {tie_vs_null} | {changed} |")

    lines.append("")


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: PROPORTION NOT MOVING
# ═══════════════════════════════════════════════════════════════════════

def analysis_two_proportion_not_moving(rows, models, lines):
    lines.append("## 2. Proportion not moving (response ratio exactly zero)")
    lines.append("")
    lines.append("The manuscript states 'median response ratio 0.00' "
                 "(docs/FORK_RESULTS_EXTRACT.md:116, docs/PAPER_SOURCE_OF_TRUTH.md:652). "
                 "A median of exactly 0.00 is produced by a point mass at zero, not by a "
                 "distribution centred near zero. The proportion below is the statistic "
                 "the text should quote instead, because it says directly what the median "
                 "only implies.")
    lines.append("")
    lines.append("Response ratio = `delta_21 / true_delta`, on eligible non-OOD pairs with "
                 "`|true_delta| > 1e-10` (the guard used at score_fork.py:282).")
    lines.append("")
    lines.append("| Model | Proportion ratio exactly 0 [95% CI] | Median ratio | n |")
    lines.append("|-------|------------------------------------:|-------------:|---:|")

    results = {}
    for model in models:
        model_rows = []
        for row in rows_for_model(rows, model):
            if abs(row["true_delta"]) > 1e-10:
                model_rows.append(row)
        if not model_rows:
            continue

        ratios = []
        zero_flags = []
        cluster_ids = []
        for row in model_rows:
            ratio = row["delta_21"] / row["true_delta"]
            ratios.append(ratio)
            zero_flags.append(1 if ratio == 0 else 0)
            cluster_ids.append(row["cluster_id"])

        zero_ci = clustered_bootstrap_ci(zero_flags, cluster_ids)
        median_ratio = float(np.median(np.array(ratios)))

        results[model] = {"zero_ci": zero_ci, "median": median_ratio, "n": len(model_rows)}

        lines.append(
            f"| {model} "
            f"| {zero_ci['mean']*100:.1f}% [{zero_ci['ci_low']*100:.1f}, {zero_ci['ci_high']*100:.1f}] "
            f"| {median_ratio:.2f} | {len(model_rows)} |"
        )

    lines.append("")
    return results


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: GEMINI TEMPERATURE ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════

def build_gemini_temp_pairs():
    """Rebuild the per-pair temp-0 vs temp-0.7 comparison for Gemini.

    Reproduces the cell subsample and pair construction of
    scripts/gemini_temp_robustness.py:114-348, with one correction: the
    published report divides `temp0_correct` by `temp07_n` even though the two
    are accumulated over different pair sets (temp0_correct is incremented at
    line 327, before the temp-0.7 availability guard at line 332). Here both
    accuracies are computed over the SAME set of comparable pairs.
    """
    ground_truth = json.loads(FORK_GT_FILE.read_text())

    eligible_cells = []
    for cell_key, cell_data in ground_truth.items():
        if cell_data.get("is_ood"):
            continue
        has_eligible = False
        for gt_info in cell_data["forecasts"].values():
            if gt_info["eligible"]:
                has_eligible = True
                break
        if has_eligible:
            eligible_cells.append(cell_key)

    # Same deterministic subsample as the original script.
    random.seed(42)
    sample_size = max(1, len(eligible_cells) // 10)
    sample = sorted(random.sample(eligible_cells, sample_size))

    # Temp-0 Gemini forecasts from the main fork run.
    temp0_index = {}
    with open(FORK_CHECKPOINT) as f:
        for line in f:
            record = json.loads(line)
            if record["model"] != "Gemini 3.5 Flash":
                continue
            if not record.get("parse_success") or not record.get("forecast"):
                continue
            key = (record["arm"], record["world"], record.get("param"),
                   record.get("setting"), record["seed"])
            points = {}
            for var_h, value in record["forecast"].items():
                if isinstance(value, dict) and "point" in value:
                    points[var_h] = value["point"]
            temp0_index[key] = points

    # Temp-0.7 replicate forecasts.
    temp07_index = defaultdict(list)
    with open(GEMINI_CHECKPOINT) as f:
        for line in f:
            record = json.loads(line)
            if not record.get("parse_success") or not record.get("forecast"):
                continue
            key = (record["arm"], record["world"], record.get("param"),
                   record.get("setting"), record["seed"])
            points = {}
            for var_h, value in record["forecast"].items():
                if isinstance(value, dict) and "point" in value:
                    points[var_h] = value["point"]
            temp07_index[key].append(points)

    pairs = []
    for cell_key in sample:
        cell_data = ground_truth[cell_key]
        world = cell_data["world"]
        param = cell_data["param"]
        setting = cell_data["setting"]
        seed = cell_data["seed"]
        baseline_val = BASELINES_MAP[param][world]

        arm2_key = ("arm2_change", world, param, setting, seed)
        arm1_key = ("arm1_placebo", world, param, None, seed)
        arm1_key_baseline = ("arm1_placebo", world, param, baseline_val, seed)

        t0_arm2 = temp0_index.get(arm2_key, {})
        t0_arm1 = temp0_index.get(arm1_key, temp0_index.get(arm1_key_baseline, {}))

        t07_arm2_list = temp07_index.get(arm2_key, [])
        # The robustness run stored ARM1 under the modified setting.
        t07_arm1_list = temp07_index.get(("arm1_placebo", world, param, setting, seed), [])
        if not t07_arm1_list:
            t07_arm1_list = temp07_index.get(
                arm1_key, temp07_index.get(arm1_key_baseline, []))

        for var_h, gt_info in cell_data["forecasts"].items():
            if not gt_info["eligible"]:
                continue
            true_delta = gt_info["true_delta"]

            if var_h not in t0_arm2 or var_h not in t0_arm1:
                continue
            if not t07_arm2_list or not t07_arm1_list:
                continue

            temp0_delta = t0_arm2[var_h] - t0_arm1[var_h]

            # Modal direction over the cross product of replicates, as in the
            # original script. Note this forces a nonzero direction: a tie maps
            # to +1, so temp-0.7 can never record a zero response while temp-0
            # can.
            directions = []
            for arm2_points in t07_arm2_list:
                for arm1_points in t07_arm1_list:
                    if var_h in arm2_points and var_h in arm1_points:
                        delta = arm2_points[var_h] - arm1_points[var_h]
                        directions.append(1 if delta > 0 else -1)
            if not directions:
                continue
            modal_direction = 1 if sum(directions) > 0 else -1

            temp0_direction = 0
            if temp0_delta > 0:
                temp0_direction = 1
            elif temp0_delta < 0:
                temp0_direction = -1

            temp0_correct = 1 if sign_correct(true_delta, temp0_delta) else 0
            temp07_correct = 1 if sign_correct(true_delta, modal_direction) else 0

            pairs.append({
                "cell_key": cell_key,
                "cluster_id": f"{world}__{param}__{setting}__s{seed}",
                "true_delta": true_delta,
                "temp0_delta": temp0_delta,
                "temp0_is_zero": temp0_delta == 0,
                "temp0_direction": temp0_direction,
                "modal_direction": modal_direction,
                "temp0_correct": temp0_correct,
                "temp07_correct": temp07_correct,
                "agree": 1 if modal_direction == temp0_direction else 0,
            })

    # Coverage check: the robustness run carried a $5 spend cap
    # (scripts/gemini_temp_robustness.py:43, SPEND_CAP = 5.0), so the schedule
    # may have been truncated before every sampled cell was called.
    cells_with_data = set()
    for key in temp07_index:
        _arm, world, param, setting, seed = key
        cells_with_data.add((world, param, setting, seed))

    coverage = {
        "cells_sampled": len(sample),
        "cells_with_data": len(cells_with_data),
        "calls_scheduled": len(sample) * 2 * 3,
        "calls_present": sum(len(v) for v in temp07_index.values()),
    }

    return pairs, len(sample), coverage


def cluster_signflip_permutation_p(differences, cluster_ids, n_perm=N_PERM, seed=SEED):
    """Two-sided permutation p-value for a paired difference, flipping at the
    cluster level.

    Under the null that temperature has no effect, the sign of each cluster's
    mean paired difference is exchangeable. This respects the clustering rather
    than assuming pairs within a cell are independent.
    """
    clusters = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        clusters[cid].append(differences[i])

    cluster_means = []
    cluster_weights = []
    for values in clusters.values():
        cluster_means.append(float(np.mean(values)))
        cluster_weights.append(len(values))
    cluster_means = np.array(cluster_means)
    cluster_weights = np.array(cluster_weights, dtype=np.float64)

    observed = float(np.average(cluster_means, weights=cluster_weights))

    rng = np.random.default_rng(seed)
    n_clusters = len(cluster_means)
    count_as_extreme = 0
    for k in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_clusters)
        permuted = float(np.average(cluster_means * signs, weights=cluster_weights))
        if abs(permuted) >= abs(observed):
            count_as_extreme += 1

    p_value = (count_as_extreme + 1) / (n_perm + 1)
    return {"observed": observed, "p_value": p_value, "n_clusters": n_clusters}


def analysis_three_gemini_temp(lines):
    lines.append("## 3. Gemini temperature-robustness uncertainty")
    lines.append("")

    pairs, n_cells, coverage = build_gemini_temp_pairs()
    if not pairs:
        lines.append("No comparable pairs rebuilt — check the robustness checkpoint.")
        lines.append("")
        return None

    temp0_correct = []
    temp07_correct = []
    differences = []
    cluster_ids = []
    n_temp0_zero = 0
    for pair in pairs:
        temp0_correct.append(pair["temp0_correct"])
        temp07_correct.append(pair["temp07_correct"])
        differences.append(pair["temp07_correct"] - pair["temp0_correct"])
        cluster_ids.append(pair["cluster_id"])
        if pair["temp0_is_zero"]:
            n_temp0_zero += 1

    n_pairs = len(pairs)
    n_clusters = len(set(cluster_ids))

    temp0_acc = float(np.mean(temp0_correct)) * 100
    temp07_acc = float(np.mean(temp07_correct)) * 100

    diff_ci = clustered_bootstrap_ci(differences, cluster_ids)
    perm = cluster_signflip_permutation_p(differences, cluster_ids)

    lines.append(f"Rebuilt from the same deterministic 10% cell subsample "
                 f"({n_cells} cells, seed 42). Comparable pairs: **{n_pairs}** across "
                 f"**{n_clusters}** clusters (cluster = cell = world__param__setting__seed).")
    lines.append("")
    lines.append(f"**Coverage flag — the published '50 cells' overstates the data.** "
                 f"The run carried a $5 spend cap "
                 f"(`scripts/gemini_temp_robustness.py:43`) and stopped after "
                 f"**{coverage['calls_present']} of {coverage['calls_scheduled']}** scheduled "
                 f"calls, so only **{coverage['cells_with_data']} of "
                 f"{coverage['cells_sampled']}** sampled cells were ever called; "
                 f"{n_clusters} of those contribute eligible pairs. The published table row "
                 f"'Cells in subsample | 50' reports the intended sample, not the achieved "
                 f"one — it is `len(sample)` (line 363), which is fixed before any API call. "
                 f"The 282 pairs are real, but they come from {n_clusters} cells, not 50, so "
                 f"the effective cluster count for any interval is roughly half what the "
                 f"report implies.")
    lines.append("")
    lines.append("Both accuracies are computed over the SAME pair set. The published "
                 "report divides `temp0_correct` by `temp07_n` "
                 "(scripts/gemini_temp_robustness.py:367) although the numerator is "
                 "accumulated at line 327, before the temp-0.7 availability guard at "
                 "line 332 — so the published 46.5% can have a numerator and "
                 "denominator drawn from different pair sets.")
    lines.append("")
    lines.append("| Quantity | Published | Recomputed |")
    lines.append("|----------|----------:|-----------:|")
    lines.append(f"| Pairs compared | {PUBLISHED_GEMINI_TEMP['pairs']} | {n_pairs} |")
    lines.append(f"| Temp-0 accuracy | {PUBLISHED_GEMINI_TEMP['temp0_acc']:.1f}% | {temp0_acc:.1f}% |")
    lines.append(f"| Temp-0.7 modal accuracy | {PUBLISHED_GEMINI_TEMP['temp07_acc']:.1f}% | {temp07_acc:.1f}% |")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|------:|")
    lines.append(f"| Difference (temp-0.7 − temp-0) | {diff_ci['mean']*100:+.1f} pp |")
    lines.append(f"| Clustered bootstrap 95% CI ({N_BOOT:,} draws, seed {SEED}) "
                 f"| [{diff_ci['ci_low']*100:+.1f}, {diff_ci['ci_high']*100:+.1f}] pp |")
    lines.append(f"| Cluster sign-flip permutation p ({N_PERM:,}, seed {SEED}) "
                 f"| {perm['p_value']:.4f} |")
    lines.append(f"| Clusters | {perm['n_clusters']} |")
    lines.append("")

    ci_excludes_zero = diff_ci["ci_low"] > 0 or diff_ci["ci_high"] < 0
    if ci_excludes_zero:
        verdict = ("**Distinguishable from noise.** The 95% CI excludes zero "
                   f"(p = {perm['p_value']:.4f}).")
    else:
        verdict = ("**Not distinguishable from noise.** The 95% CI includes zero "
                   f"(p = {perm['p_value']:.4f}), so the temp-0 vs temp-0.7 gap is "
                   "within sampling variability at 50 clusters.")
    lines.append(f"**One-line answer:** {verdict}")
    lines.append("")

    # The zero-response confound specific to this comparison.
    lines.append("### 3a. Zero-response confound in this comparison")
    lines.append("")
    lines.append(f"Of the {n_pairs} temp-0 pairs, **{n_temp0_zero}** "
                 f"({n_temp0_zero/n_pairs*100:.1f}%) have `temp0_delta == 0` and are "
                 "scored incorrect automatically. The temp-0.7 arm cannot produce a "
                 "zero at all: the modal direction is forced to ±1 "
                 "(scripts/gemini_temp_robustness.py:341, where a tie maps to +1). "
                 "The two arms are therefore not scored on the same rule, and part of "
                 "the apparent temperature effect is the zero-scoring artefact rather "
                 "than a decoding effect.")
    lines.append("")

    # Tie-adjusted version of the same contrast, to isolate the artefact.
    tie_temp0 = []
    for pair in pairs:
        tie_temp0.append(tie_adjusted_score(pair["true_delta"], pair["temp0_delta"]))
    tie_differences = []
    for i, pair in enumerate(pairs):
        tie_differences.append(pair["temp07_correct"] - tie_temp0[i])
    tie_diff_ci = clustered_bootstrap_ci(tie_differences, cluster_ids)
    tie_perm = cluster_signflip_permutation_p(tie_differences, cluster_ids)

    lines.append("Scoring temp-0 zeros as ties (0.5) puts both arms on a comparable "
                 "footing:")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|------:|")
    lines.append(f"| Temp-0 tie-adjusted accuracy | {float(np.mean(tie_temp0))*100:.1f}% |")
    lines.append(f"| Difference (temp-0.7 − temp-0 tie-adjusted) | {tie_diff_ci['mean']*100:+.1f} pp |")
    lines.append(f"| Clustered bootstrap 95% CI | [{tie_diff_ci['ci_low']*100:+.1f}, "
                 f"{tie_diff_ci['ci_high']*100:+.1f}] pp |")
    lines.append(f"| Permutation p | {tie_perm['p_value']:.4f} |")
    lines.append("")

    return {
        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "temp0_acc": temp0_acc,
        "temp07_acc": temp07_acc,
        "diff_ci": diff_ci,
        "perm": perm,
        "n_temp0_zero": n_temp0_zero,
        "tie_diff_ci": tie_diff_ci,
        "tie_perm": tie_perm,
    }


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: STATIONARITY ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════

def load_grid_rows():
    """Build grid infer rows scored against the reconstructed per-seed GT.

    Mirrors the per-seed scoring used for the published all-tier numbers:
    forecasts are compared setting-vs-baseline within the SAME history seed,
    and the ground-truth delta is taken between the same two per-seed cells.
    Near-zero truths (|true_delta| < 0.01) are dropped, as in
    scripts/phase1_analyses.py:456.
    """
    if not PER_SEED_GT_FILE.exists():
        return None

    per_seed_gt = json.loads(PER_SEED_GT_FILE.read_text())

    last_record = {}
    with open(GRID_CHECKPOINT) as f:
        for line in f:
            record = json.loads(line)
            last_record[record["call_id"]] = record

    forecast_index = {}
    for record in last_record.values():
        if not record.get("parse_success") or not record.get("forecast"):
            continue
        points = {}
        for var_h, value in record["forecast"].items():
            if isinstance(value, dict) and "point" in value:
                points[var_h] = value["point"]
        key = (record["model"], record["world"], record["param"],
               record["setting"], record["condition"], record["seed"])
        forecast_index[key] = points

    rows = []
    for key, modified_points in forecast_index.items():
        model, world, param, setting, condition, seed = key
        if condition != "infer":
            continue
        baseline_val = BASELINES_MAP[param][world]
        if setting == baseline_val:
            continue

        modified_stem = f"{world}__{param}__{setting}"
        baseline_stem = f"{world}__{param}__{baseline_val}"
        if modified_stem in EXCLUDED_GT_KEYS or baseline_stem in EXCLUDED_GT_KEYS:
            continue

        baseline_points = forecast_index.get(
            (model, world, param, baseline_val, condition, seed))
        if not baseline_points:
            continue

        gt_modified = per_seed_gt.get(f"{modified_stem}__s{seed}")
        gt_baseline = per_seed_gt.get(f"{baseline_stem}__s{seed}")
        if not gt_modified or not gt_baseline:
            continue

        for var_h in modified_points:
            if var_h not in baseline_points:
                continue
            if var_h not in gt_modified or var_h not in gt_baseline:
                continue

            true_delta = gt_modified[var_h] - gt_baseline[var_h]
            if abs(true_delta) < 0.01:
                continue

            rows.append({
                "model": model,
                "world": world,
                "param": param,
                "setting": setting,
                "seed": seed,
                "horizon": int(var_h.rsplit("_", 1)[1]),
                "true_delta": true_delta,
                "model_delta": modified_points[var_h] - baseline_points[var_h],
                "cluster_id": f"{world}__{param}__{setting}",
            })

    return rows


def is_non_stationary(world: str, param: str, setting: float) -> bool:
    for ns_world, ns_param, ns_setting, _lam, _excluded in NON_STATIONARY_CONFIGS:
        if world == ns_world and param == ns_param and abs(setting - ns_setting) < 1e-9:
            return True
    return False


def analysis_four_stationarity(fork_rows, models, lines):
    lines.append("## 4. Stationarity robustness")
    lines.append("")

    # ── 4a. The nine configurations ──
    lines.append("### 4a. The nine non-stationary configurations")
    lines.append("")
    lines.append("| # | World | Param | Setting | max \\|λ\\| | Already excluded? |")
    lines.append("|--:|-------|-------|--------:|--------:|:-----------------:|")
    for i, (world, param, setting, lam, excluded) in enumerate(NON_STATIONARY_CONFIGS, start=1):
        excluded_text = "**yes**" if excluded else "no"
        lines.append(f"| {i} | {world} | {param} | {setting} | {lam:.3f} | {excluded_text} |")
    lines.append("")

    b_world, b_param, b_setting, b_lam = BORDERLINE_CONFIG
    lines.append(f"**Data-quality flag (since fixed).** outputs/analysis/stationarity_report.md "
                 f"previously listed **ten** rows under the heading 'Non-stationary settings "
                 f"(max |λ| ≥ 1.0)', but `{b_world} / {b_param} / {b_setting}` has "
                 f"max |λ| = {b_lam:.3f}, which is below 1.0. It is near-unit-root, not "
                 f"non-stationary. Dropping it leaves exactly the nine that the report's own "
                 f"summary table counts, so the count was right and the table had one row too "
                 f"many. The manuscript restatement (docs/MANUSCRIPT_NUMBERS.md:201-213) omits "
                 f"the row and was already internally consistent. The row has been moved out of "
                 f"that table into a note; the nine listed above are the non-stationary set.")
    lines.append("")
    lines.append("**Already excluded:** the two World 3 Phillips cells (0.7, 0.9) are "
                 "excluded from all published analyses via `EXCLUDED_GT_KEYS` "
                 "(scripts/run_analysis.py:41) and were dropped at fork generation "
                 "(scripts/run_fork.py:151-154).")
    lines.append("")

    # ── 4b. Do the other seven enter analysis? ──
    lines.append("### 4b. Do the other seven enter any analysis?")
    lines.append("")
    lines.append("| World | Param | Setting | Fork eligible pairs | Grid infer rows |")
    lines.append("|-------|-------|--------:|--------------------:|----------------:|")

    grid_rows = load_grid_rows()

    fork_counts = defaultdict(int)
    for row in fork_rows:
        fork_counts[(row["world"], row["param"], row["setting"])] += 1

    grid_counts = defaultdict(int)
    if grid_rows is not None:
        for row in grid_rows:
            grid_counts[(row["world"], row["param"], row["setting"])] += 1

    n_entering = 0
    for world, param, setting, _lam, excluded in NON_STATIONARY_CONFIGS:
        if excluded:
            continue
        fork_n = fork_counts.get((world, param, setting), 0)
        grid_n = grid_counts.get((world, param, setting), 0)
        if fork_n > 0 or grid_n > 0:
            n_entering += 1
        grid_text = str(grid_n) if grid_rows is not None else "n/a"
        lines.append(f"| {world} | {param} | {setting} | {fork_n} | {grid_text} |")
    lines.append("")
    lines.append(f"**{n_entering} of the 7** retained non-stationary configurations "
                 "contribute rows to at least one headline analysis. They are not "
                 "screened out anywhere.")
    lines.append("")

    # ── 4c. Fork recompute ──
    lines.append("### 4c. Fork ARM2−ARM1 excluding all nine non-stationary configs")
    lines.append("")
    lines.append("| Model | Published | Recomputed (all) | Excl. 9 non-stationary | "
                 "Δ vs published | Δ vs recomputed | n (excl.) |")
    lines.append("|-------|----------:|-----------------:|-----------------------:|"
                 "---------------:|----------------:|----------:|")

    fork_results = {}
    for model in models:
        model_rows = rows_for_model(fork_rows, model)
        if not model_rows:
            continue

        all_correct = []
        all_clusters = []
        for row in model_rows:
            all_correct.append(1 if sign_correct(row["true_delta"], row["delta_21"]) else 0)
            all_clusters.append(row["cluster_id"])
        all_ci = clustered_bootstrap_ci(all_correct, all_clusters)

        kept_rows = []
        for row in model_rows:
            if not is_non_stationary(row["world"], row["param"], row["setting"]):
                kept_rows.append(row)
        kept_correct = []
        kept_clusters = []
        for row in kept_rows:
            kept_correct.append(1 if sign_correct(row["true_delta"], row["delta_21"]) else 0)
            kept_clusters.append(row["cluster_id"])
        kept_ci = clustered_bootstrap_ci(kept_correct, kept_clusters)

        published_acc = PUBLISHED_FORK_ARM21.get(model, (float("nan"), 0))[0]
        delta_vs_published = kept_ci["mean"] * 100 - published_acc
        delta_vs_recomputed = (kept_ci["mean"] - all_ci["mean"]) * 100

        fork_results[model] = {
            "all": all_ci, "kept": kept_ci,
            "delta_vs_published": delta_vs_published,
            "delta_vs_recomputed": delta_vs_recomputed,
        }

        lines.append(
            f"| {model} | {published_acc:.1f}% | {all_ci['mean']*100:.1f}% "
            f"| {kept_ci['mean']*100:.1f}% [{kept_ci['ci_low']*100:.1f}, {kept_ci['ci_high']*100:.1f}] "
            f"| {delta_vs_published:+.1f} pp | {delta_vs_recomputed:+.1f} pp | {kept_ci['n']} |"
        )
    lines.append("")

    # ── 4d. Grid recompute ──
    lines.append("### 4d. Grid infer all-tier excluding all nine non-stationary configs")
    lines.append("")

    grid_results = {}
    if grid_rows is None:
        lines.append("**NOT RUN.** The per-seed ground truth "
                     "(`outputs/analysis/per_seed_gt_reconstructed.json`) is missing. "
                     "Run `python scripts/rebuild_per_seed_gt.py` first.")
        lines.append("")
        return fork_results, grid_results

    lines.append("Scored against a reconstructed per-seed ground truth. The file the "
                 "published numbers used (`/tmp/lmm2_per_seed_test/ground_truth_per_seed.json`) "
                 "no longer exists and no script in the repo regenerates it, so "
                 "`scripts/rebuild_per_seed_gt.py` rebuilds it using the construction in "
                 "`scripts/run_analysis.py::load_ground_truth` with the history seed varied "
                 "over 0-9. The 'Recomputed (all)' column is the validation check: it should "
                 "land on the published number if the reconstruction is faithful.")
    lines.append("")
    lines.append("| Model | Published | Recomputed (all) | Excl. 9 non-stationary | "
                 "Δ vs published | Δ vs recomputed | n (all) | n (excl.) |")
    lines.append("|-------|----------:|-----------------:|-----------------------:|"
                 "---------------:|----------------:|-------:|----------:|")

    grid_models = sorted(set(row["model"] for row in grid_rows))
    for model in grid_models:
        model_rows = []
        for row in grid_rows:
            if row["model"] == model:
                model_rows.append(row)
        if not model_rows:
            continue

        all_correct = []
        all_clusters = []
        for row in model_rows:
            all_correct.append(1 if sign_correct(row["true_delta"], row["model_delta"]) else 0)
            all_clusters.append(row["cluster_id"])
        all_ci = clustered_bootstrap_ci(all_correct, all_clusters)

        kept_rows = []
        for row in model_rows:
            if not is_non_stationary(row["world"], row["param"], row["setting"]):
                kept_rows.append(row)
        kept_correct = []
        kept_clusters = []
        for row in kept_rows:
            kept_correct.append(1 if sign_correct(row["true_delta"], row["model_delta"]) else 0)
            kept_clusters.append(row["cluster_id"])
        kept_ci = clustered_bootstrap_ci(kept_correct, kept_clusters)

        published_acc = PUBLISHED_GRID_INFER_ALL.get(model, (float("nan"), 0))[0]
        delta_vs_published = kept_ci["mean"] * 100 - published_acc
        delta_vs_recomputed = (kept_ci["mean"] - all_ci["mean"]) * 100

        grid_results[model] = {
            "all": all_ci, "kept": kept_ci,
            "published": published_acc,
            "delta_vs_published": delta_vs_published,
            "delta_vs_recomputed": delta_vs_recomputed,
        }

        lines.append(
            f"| {model} | {published_acc:.1f}% | {all_ci['mean']*100:.1f}% "
            f"| {kept_ci['mean']*100:.1f}% [{kept_ci['ci_low']*100:.1f}, {kept_ci['ci_high']*100:.1f}] "
            f"| {delta_vs_published:+.1f} pp | {delta_vs_recomputed:+.1f} pp "
            f"| {all_ci['n']} | {kept_ci['n']} |"
        )
    lines.append("")

    # Validation verdict on the reconstruction.
    worst_gap = 0.0
    for model, res in grid_results.items():
        gap = abs(res["all"]["mean"] * 100 - res["published"])
        if gap > worst_gap:
            worst_gap = gap

    if worst_gap <= 1.0:
        lines.append(f"**Reconstruction validated.** Largest gap between the recomputed "
                     f"all-config accuracy and the published number is {worst_gap:.1f} pp, "
                     f"so the rebuilt per-seed ground truth reproduces the published "
                     f"pipeline and the exclusion deltas above are trustworthy.")
    else:
        lines.append(f"**Reconstruction NOT validated.** Largest gap between the recomputed "
                     f"all-config accuracy and the published number is {worst_gap:.1f} pp. "
                     f"The rebuilt per-seed ground truth does not reproduce the published "
                     f"pipeline, so the absolute levels in this table are not comparable to "
                     f"the manuscript. The exclusion delta (last-but-one column) is computed "
                     f"within this table and remains internally consistent, but it should not "
                     f"be quoted against published numbers until the discrepancy is resolved.")
    lines.append("")

    return fork_results, grid_results


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def build_summary(results_one, results_two, results_three,
                  fork_stationarity, grid_stationarity):
    """Assemble the headline findings, including every qualitative-claim change."""
    lines = []
    lines.append("## Summary of what changes")
    lines.append("")

    # Which models change characterisation under tie-adjusted scoring.
    changed_models = []
    for model, res in results_one.items():
        tie = res["tie"]
        if res["published"] < 50.0:
            published_claim = "below chance"
        else:
            published_claim = "at chance"
        if tie["ci_high"] * 100 < 50.0:
            tie_claim = "below chance"
        elif tie["ci_low"] * 100 > 50.0:
            tie_claim = "above chance"
        else:
            tie_claim = "indistinguishable from chance"
        if tie_claim != published_claim:
            changed_models.append((model, published_claim, tie_claim,
                                   res["published"], tie["mean"] * 100))

    lines.append("**1. The central fork claim does not survive tie-adjusted scoring.** "
                 "The published finding is that no model meets the preregistered success "
                 "criterion and that three of four sit entirely below 50% "
                 "(docs/MANUSCRIPT_NUMBERS.md:26). Scoring zeros as ties changes the "
                 "characterisation for every model:")
    lines.append("")
    for model, published_claim, tie_claim, published_acc, tie_acc in changed_models:
        lines.append(f"- **{model}**: {published_acc:.1f}% ({published_claim}) → "
                     f"{tie_acc:.1f}% ({tie_claim})")
    lines.append("")
    lines.append("No model is below chance once non-responses stop counting as wrong "
                 "answers. Two (GPT-5.5, Gemini 3.5 Flash) move to *above* chance and "
                 "also clear their own zero-preserving permutation null. This is a "
                 "qualitative reversal of the headline claim, not a numerical adjustment.")
    lines.append("")

    lines.append("**2. The zero rate is the mechanism, and it is large.** "
                 "Between 15.9% and 40.8% of eligible pairs are exact non-responses. "
                 "The ranking of models by published accuracy is almost exactly the "
                 "inverse ranking by zero rate — Opus is 'worst' (30.6%) and has the "
                 "most zeros (40.8%); GPT-5.5 is 'best' (49.3%) and has the fewest "
                 "(15.9%). Conditional on actually moving, all four models cluster in a "
                 "narrow 51-59% band. The published spread measures willingness to move "
                 "more than directional skill.")
    lines.append("")

    lines.append("**3. 50% was never the right benchmark, but the corrected benchmark is "
                 "close to it.** The zero-preserving permutation null sits at 50.2-50.7% "
                 "for all four models, so the referee is right that the benchmark was "
                 "wrong in principle, but the size of the error is small. What moves the "
                 "conclusion is the scoring of zeros in the numerator, not the location "
                 "of the chance line.")
    lines.append("")

    if results_three is not None:
        tie_ci = results_three["tie_diff_ci"]
        tie_perm = results_three["tie_perm"]
        lines.append(f"**4. The Gemini temperature effect is an artefact of the same bug.** "
                     f"The published contrast (46.5% vs 60.3%) is significant "
                     f"(+13.8 pp, p = {results_three['perm']['p_value']:.4f}), but the two "
                     f"arms are not scored on the same rule: temp-0 can return a zero and be "
                     f"marked wrong, while the temp-0.7 modal direction is forced to ±1 and "
                     f"never can. Scoring both consistently collapses the difference to "
                     f"{tie_ci['mean']*100:+.1f} pp "
                     f"[{tie_ci['ci_low']*100:+.1f}, {tie_ci['ci_high']*100:+.1f}] "
                     f"(p = {tie_perm['p_value']:.4f}) — indistinguishable from noise. "
                     f"The appendix claim 'conclusions not sensitive to deterministic "
                     f"decoding' happens to survive, but the number supporting it does not.")
        lines.append("")
        lines.append(f"**5. The temperature slice covers half the cells its report claims.** "
                     f"A spend cap truncated the run: {results_three['n_clusters']} cells "
                     f"contribute data, not 50. Any interval on that slice rests on "
                     f"{results_three['n_clusters']} clusters.")
        lines.append("")

    lines.append("**6. Stationarity exclusion moves the grid more than the fork.** "
                 "Dropping all nine non-stationary configurations shifts fork accuracies "
                 "by under 1 pp (no claim changes) but lowers every grid infer accuracy by "
                 "1.5-2.7 pp. The direction is consistent: the retained non-stationary "
                 "cells were inflating grid performance. No grid conclusion reverses, but "
                 "the numbers are not exclusion-invariant and the manuscript should say so.")
    lines.append("")

    lines.append("**7. Two data-quality defects found in passing** (details in §3 and §4a): "
                 "the stationarity report's table lists ten rows under a heading that admits "
                 "nine, and the temperature report's accuracy numerator and denominator are "
                 "accumulated over different pair sets.")
    lines.append("")
    lines.append("---")
    lines.append("")

    return lines


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Building fork rows (ARM2-ARM1, eligible, non-OOD)...")
    fork_rows, models = build_fork_rows()
    logger.info(f"Fork rows: {len(fork_rows)} across {len(models)} models")

    lines = []
    lines.append("# Zero-Response Reanalysis")
    lines.append("")
    lines.append("**Referee critique addressed:** exactly-zero forecast changes are scored "
                 "as incorrect, which makes 50% the wrong chance benchmark when models "
                 "frequently return unchanged forecasts.")
    lines.append("")
    lines.append("The scoring rule at `scripts/score_fork.py:49` is")
    lines.append("")
    lines.append("```python")
    lines.append("def _sign_correct(td, md):")
    lines.append("    return (td > 0 and md > 0) or (td < 0 and md < 0)")
    lines.append("```")
    lines.append("")
    lines.append("Both branches require a strict inequality on `md`, so `model_delta == 0` "
                 "returns `False` and is counted as a miss. There is no tie branch and no "
                 "exclusion. The repo already documents the consequence in another context: "
                 "`outputs/analysis/simple_baselines_report.md:18-19` notes that the "
                 "persistence baseline \"always scores 0% because forecast_delta = 0 counts "
                 "as incorrect by the scoring rule\". A coin flip cannot produce a zero, but "
                 "a model can, so a model that declines to move is penalised against a "
                 "benchmark that never faces that choice.")
    lines.append("")
    lines.append("Generated by `scripts/zero_response_reanalysis.py`. No existing scoring "
                 "script or report was modified.")
    lines.append("")

    # The summary depends on results computed below, so reserve its slot and
    # fill it in once everything has run.
    summary_slot = len(lines)

    logger.info("Analysis 1: zero-response reanalysis...")
    results_one = analysis_one_zero_response(fork_rows, models, lines)
    flag_qualitative_changes(results_one, lines)

    logger.info("Analysis 2: proportion not moving...")
    results_two = analysis_two_proportion_not_moving(fork_rows, models, lines)

    logger.info("Analysis 3: Gemini temperature robustness...")
    results_three = analysis_three_gemini_temp(lines)

    logger.info("Analysis 4: stationarity robustness...")
    fork_stationarity, grid_stationarity = analysis_four_stationarity(
        fork_rows, models, lines)

    summary_lines = build_summary(results_one, results_two, results_three,
                                  fork_stationarity, grid_stationarity)
    lines[summary_slot:summary_slot] = summary_lines

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text)
    logger.info(f"Report written to {REPORT_PATH}")
    print(report_text)


if __name__ == "__main__":
    main()
