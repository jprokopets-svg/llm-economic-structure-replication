"""Score fork experiment results.

Reads fork_ground_truth.json, checkpoint.jsonl, and prefork_trends.json
to produce the full scoring report:

  1. PRIMARY: directional accuracy (ARM2-ARM3 vs GT sign)
  2. SECONDARY: response ratio (model_delta / true_delta)
  3. PLACEBO CHECK: ARM1-ARM3 distribution
  4. SIGN-REVERSAL TRACKING: paired h4→h8 flips
  5. CONFLICT CLASSIFICATION (exploratory): aligned/conflict/neutral by pre-fork trend
  6. OOD EXTENSION results

Usage:
    python scripts/score_fork.py
"""

import json
import logging
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
GT_FILE = FORK_DIR / "fork_ground_truth.json"
CHECKPOINT_FILE = FORK_DIR / "checkpoint.jsonl"
TRENDS_FILE = FORK_DIR / "prefork_trends.json"
HORIZONS = [1, 4, 8]

# Sign-flip cells: (param, variable) pairs where prereg predicts
# true_delta flips sign between h=4 and h=8.
SIGN_FLIP_PAIRS = [
    ("phillips_slope", "pi"),
    ("phillips_slope", "r"),
    ("taylor_phi_pi", "r"),
    ("wage_gap_slope", "pi"),
    ("wage_gap_slope", "w"),
]


def _sign_correct(td, md):
    return (td > 0 and md > 0) or (td < 0 and md < 0)


def load_forecasts():
    """Load parsed forecasts from checkpoint, deduplicated by call_id."""
    last_record = {}
    with open(CHECKPOINT_FILE) as f:
        for line in f:
            r = json.loads(line)
            last_record[r["call_id"]] = r
    # Only keep successful parses
    forecasts = [r for r in last_record.values()
                 if r.get("parse_success") and r.get("forecast")]
    return forecasts


def index_forecasts(forecasts):
    """Index forecasts by (arm, world, param, setting, seed, model) -> {var_h: point}."""
    idx = {}
    for f in forecasts:
        key = (f["arm"], f["world"], f.get("param"), f.get("setting"),
               f["seed"], f["model"])
        vals = {}
        for var_h, v in f["forecast"].items():
            if isinstance(v, dict) and "point" in v:
                vals[var_h] = v["point"]
        idx[key] = vals
    return idx


def clustered_bootstrap_ci(values, cluster_ids, n_boot=10000, seed=42):
    """Clustered bootstrap 95% CI on the mean of values."""
    values = np.array(values, dtype=np.float64)
    n = len(values)
    if n == 0:
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
        "n": n,
    }


def classify_conflict(trend_8q, true_delta, rolling_sd):
    """Classify a cell as ALIGNED, CONFLICT, or NEUTRAL.

    NEUTRAL: |8q trend| < 0.5 × rolling_sd (trend is flat relative to noise)
    ALIGNED: trend direction == sign(true_delta)
    CONFLICT: trend direction == -sign(true_delta)
    """
    if rolling_sd > 0 and abs(trend_8q) < 0.5 * rolling_sd:
        return "NEUTRAL"
    if true_delta == 0 or trend_8q == 0:
        return "NEUTRAL"
    if (trend_8q > 0) == (true_delta > 0):
        return "ALIGNED"
    return "CONFLICT"


def main():
    gt = json.loads(GT_FILE.read_text())
    forecasts = load_forecasts()
    fc = index_forecasts(forecasts)
    trends = json.loads(TRENDS_FILE.read_text())

    logger.info(f"GT cells: {len(gt)}")
    logger.info(f"Forecasts loaded: {len(forecasts)}")

    # Identify models from data
    models = sorted(set(f["model"] for f in forecasts))
    logger.info(f"Models: {models}")

    # ── Build scored rows ──
    rows = []
    for cell_key, cell_data in gt.items():
        world = cell_data["world"]
        param = cell_data["param"]
        setting = cell_data["setting"]
        seed = cell_data["seed"]
        is_ood = cell_data.get("is_ood", False)
        trend_key = f"{world}__s{seed}"
        cell_trends = trends.get(trend_key, {})

        for model in models:
            # ARM 3 (baseline) forecast
            arm3_key = ("arm3_baseline", world, None, None, seed, model)
            arm3_fc = fc.get(arm3_key, {})

            # ARM 2 (change) forecast
            arm2_key = ("arm2_change", world, param, setting, seed, model)
            arm2_fc = fc.get(arm2_key, {})

            # ARM 1 (placebo) forecast
            arm1_key = ("arm1_placebo", world, param, None, seed, model)
            # ARM1 setting is baseline_val, stored as the setting field
            # Try multiple possible key patterns
            arm1_fc = fc.get(arm1_key, {})
            if not arm1_fc:
                # Try with baseline_val as setting
                BASELINES_MAP = {
                    "phillips_slope": {"world1": 0.4, "world2": 0.4, "world3": 0.5, "world4": 0.2},
                    "taylor_phi_pi": {"world1": 1.5, "world2": 1.5, "world3": 2.0, "world4": 1.5},
                    "is_sensitivity": {"world1": 0.6, "world2": 0.6, "world3": 0.6, "world4": 0.6},
                    "wage_gap_slope": {"world4": 0.5},
                }
                baseline_val = BASELINES_MAP[param][world]
                arm1_key2 = ("arm1_placebo", world, param, baseline_val, seed, model)
                arm1_fc = fc.get(arm1_key2, {})

            for var_h, gt_info in cell_data["forecasts"].items():
                true_delta = gt_info["true_delta"]
                mc_se = gt_info["mc_se"]
                eligible = gt_info["eligible"]

                var = var_h.rsplit("_", 1)[0]
                horizon = int(var_h.rsplit("_", 1)[1])

                # ARM2 - ARM3 delta
                arm2_point = arm2_fc.get(var_h)
                arm3_point = arm3_fc.get(var_h)
                if arm2_point is not None and arm3_point is not None:
                    model_delta = arm2_point - arm3_point
                else:
                    model_delta = None

                # ARM1 - ARM3 delta (placebo)
                arm1_point = arm1_fc.get(var_h)
                if arm1_point is not None and arm3_point is not None:
                    placebo_delta = arm1_point - arm3_point
                else:
                    placebo_delta = None

                # Conflict classification
                var_trend = cell_trends.get(var, {})
                trend_8q = var_trend.get("trend_8q", 0)
                trend_4q = var_trend.get("trend_4q", 0)
                rolling_sd = var_trend.get("rolling_sd", 1.0)
                conflict_class = classify_conflict(trend_8q, true_delta, rolling_sd)

                cluster_id = f"{world}__{param}__{setting}"

                rows.append({
                    "model": model,
                    "world": world,
                    "param": param,
                    "setting": setting,
                    "seed": seed,
                    "var": var,
                    "var_h": var_h,
                    "horizon": horizon,
                    "true_delta": true_delta,
                    "mc_se": mc_se,
                    "eligible": eligible,
                    "model_delta": model_delta,
                    "placebo_delta": placebo_delta,
                    "trend_8q": trend_8q,
                    "trend_4q": trend_4q,
                    "rolling_sd": rolling_sd,
                    "conflict_class": conflict_class,
                    "cluster_id": cluster_id,
                    "is_ood": is_ood,
                })

    logger.info(f"Scored rows: {len(rows)}")

    # ── 1. PRIMARY: directional accuracy ──
    lines = []
    lines.append("# Fork Experiment Scoring Report")
    lines.append("")
    lines.append("## 1. PRIMARY: Directional accuracy (ARM2−ARM3 vs GT)")
    lines.append("")
    lines.append("Eligible cells: |true_delta| > 2×MC_SE. Clustered bootstrap "
                 "10K reps, seed 42, cluster = world__param__setting.")
    lines.append("")

    lines.append("| Model | Accuracy | 95% CI | n eligible | n total |")
    lines.append("|-------|--------:|-------:|----------:|--------:|")

    for model in models:
        eligible_rows = [r for r in rows
                         if r["model"] == model
                         and r["eligible"]
                         and r["model_delta"] is not None
                         and not r["is_ood"]]
        total_rows = [r for r in rows
                      if r["model"] == model
                      and r["model_delta"] is not None
                      and not r["is_ood"]]

        if eligible_rows:
            correct = [1 if _sign_correct(r["true_delta"], r["model_delta"]) else 0
                       for r in eligible_rows]
            cids = [r["cluster_id"] for r in eligible_rows]
            ci = clustered_bootstrap_ci(correct, cids)
            lines.append(
                f"| {model} | {ci['mean']*100:.1f}% "
                f"| [{ci['ci_low']*100:.1f}%, {ci['ci_high']*100:.1f}%] "
                f"| {ci['n']} | {len(total_rows)} |"
            )
        else:
            lines.append(f"| {model} | — | — | 0 | {len(total_rows)} |")

    lines.append("")

    # ── 2. SECONDARY: response ratio ──
    lines.append("## 2. SECONDARY: Response ratio (model_delta / true_delta)")
    lines.append("")
    lines.append("| Model | Median ratio | IQR | Fraction <10% | Fraction >200% | n |")
    lines.append("|-------|------------:|----:|-------------:|---------------:|--:|")

    for model in models:
        eligible_rows = [r for r in rows
                         if r["model"] == model
                         and r["eligible"]
                         and r["model_delta"] is not None
                         and abs(r["true_delta"]) > 1e-10
                         and not r["is_ood"]]
        if eligible_rows:
            ratios = [r["model_delta"] / r["true_delta"] for r in eligible_rows]
            ratios_arr = np.array(ratios)
            median = float(np.median(ratios_arr))
            q25 = float(np.percentile(ratios_arr, 25))
            q75 = float(np.percentile(ratios_arr, 75))
            frac_micro = float(np.mean(np.abs(ratios_arr) < 0.1))
            frac_over = float(np.mean(np.abs(ratios_arr) > 2.0))
            lines.append(
                f"| {model} | {median:.2f} "
                f"| [{q25:.2f}, {q75:.2f}] "
                f"| {frac_micro*100:.1f}% | {frac_over*100:.1f}% "
                f"| {len(eligible_rows)} |"
            )
    lines.append("")

    # ── 3. PLACEBO CHECK ──
    lines.append("## 3. PLACEBO CHECK: ARM1−ARM3 distribution")
    lines.append("")
    lines.append("| Model | Mean |placebo| | SD | Fraction |placebo| > |true_delta|/2 | n |")
    lines.append("|-------|------------------:|---:|-------------------------------------------:|--:|")

    for model in models:
        placebo_rows = [r for r in rows
                        if r["model"] == model
                        and r["placebo_delta"] is not None
                        and r["eligible"]
                        and not r["is_ood"]]
        if placebo_rows:
            pdeltas = np.array([r["placebo_delta"] for r in placebo_rows])
            tdeltas = np.array([abs(r["true_delta"]) for r in placebo_rows])
            mean_abs = float(np.mean(np.abs(pdeltas)))
            sd = float(np.std(pdeltas))
            frac_material = float(np.mean(np.abs(pdeltas) > tdeltas / 2))
            lines.append(
                f"| {model} | {mean_abs:.4f} | {sd:.4f} "
                f"| {frac_material*100:.1f}% | {len(placebo_rows)} |"
            )
    lines.append("")

    # ── 4. SIGN-REVERSAL TRACKING ──
    lines.append("## 4. SIGN-REVERSAL TRACKING (paired h4→h8)")
    lines.append("")
    lines.append("For (param, variable) pairs where GT flips sign between h=4 and h=8.")
    lines.append("")
    lines.append("| Model | Param | Variable | GT flips | Model flips | Match rate | n |")
    lines.append("|-------|-------|----------|--------:|:-----------:|----------:|--:|")

    for model in models:
        for param, var in SIGN_FLIP_PAIRS:
            # Find paired h=4 and h=8 cells
            h4_rows = {(r["world"], r["setting"], r["seed"]): r
                       for r in rows
                       if r["model"] == model and r["param"] == param
                       and r["var"] == var and r["horizon"] == 4
                       and r["model_delta"] is not None
                       and not r["is_ood"]}
            h8_rows = {(r["world"], r["setting"], r["seed"]): r
                       for r in rows
                       if r["model"] == model and r["param"] == param
                       and r["var"] == var and r["horizon"] == 8
                       and r["model_delta"] is not None
                       and not r["is_ood"]}

            common = set(h4_rows.keys()) & set(h8_rows.keys())
            gt_flips = 0
            model_flips = 0
            matches = 0
            for ck in common:
                r4 = h4_rows[ck]
                r8 = h8_rows[ck]
                gt_flip = (r4["true_delta"] > 0) != (r8["true_delta"] > 0)
                if gt_flip:
                    gt_flips += 1
                    model_flip = (r4["model_delta"] > 0) != (r8["model_delta"] > 0)
                    if model_flip:
                        model_flips += 1
                        matches += 1

            if gt_flips > 0:
                match_rate = matches / gt_flips * 100
                lines.append(
                    f"| {model} | {param} | {var} "
                    f"| {gt_flips} | {model_flips} "
                    f"| {match_rate:.0f}% | {len(common)} |"
                )
    lines.append("")

    # ── 5. CONFLICT CLASSIFICATION (EXPLORATORY) ──
    lines.append("## 5. CONFLICT CLASSIFICATION (exploratory)")
    lines.append("")
    lines.append("Pre-fork trend direction vs sign(true_delta). "
                 "NEUTRAL: |8q trend| < 0.5×rolling_sd. "
                 "This analysis is **exploratory** — not part of the "
                 "confirmatory primary outcome.")
    lines.append("")
    lines.append("### 5a. Cell counts by classification")
    lines.append("")
    for cls in ["ALIGNED", "CONFLICT", "NEUTRAL"]:
        n = sum(1 for r in rows if r["conflict_class"] == cls
                and r["eligible"] and not r["is_ood"])
        lines.append(f"- {cls}: {n}")
    lines.append("")

    lines.append("### 5b. Directional accuracy by conflict class")
    lines.append("")
    lines.append("| Model | Class | Accuracy | 95% CI | n |")
    lines.append("|-------|-------|--------:|-------:|--:|")

    for model in models:
        for cls in ["ALIGNED", "CONFLICT", "NEUTRAL"]:
            cls_rows = [r for r in rows
                        if r["model"] == model
                        and r["conflict_class"] == cls
                        and r["eligible"]
                        and r["model_delta"] is not None
                        and not r["is_ood"]]
            if cls_rows:
                correct = [1 if _sign_correct(r["true_delta"], r["model_delta"]) else 0
                           for r in cls_rows]
                cids = [r["cluster_id"] for r in cls_rows]
                ci = clustered_bootstrap_ci(correct, cids)
                lines.append(
                    f"| {model} | {cls} "
                    f"| {ci['mean']*100:.1f}% "
                    f"| [{ci['ci_low']*100:.1f}%, {ci['ci_high']*100:.1f}%] "
                    f"| {ci['n']} |"
                )
    lines.append("")

    lines.append("### 5c. Placebo movement by conflict class")
    lines.append("")
    lines.append("| Model | Class | Mean |placebo| | n |")
    lines.append("|-------|-------|------------------:|--:|")

    for model in models:
        for cls in ["ALIGNED", "CONFLICT", "NEUTRAL"]:
            cls_rows = [r for r in rows
                        if r["model"] == model
                        and r["conflict_class"] == cls
                        and r["eligible"]
                        and r["placebo_delta"] is not None
                        and not r["is_ood"]]
            if cls_rows:
                mean_abs = float(np.mean([abs(r["placebo_delta"]) for r in cls_rows]))
                lines.append(
                    f"| {model} | {cls} | {mean_abs:.4f} | {len(cls_rows)} |"
                )
    lines.append("")

    # ── 6. OOD EXTENSION ──
    lines.append("## 6. OOD EXTENSION (W1 phillips_slope)")
    lines.append("")

    ood_rows = [r for r in rows if r["is_ood"] and r["eligible"]
                and r["model_delta"] is not None]
    if ood_rows:
        lines.append("| Model | Setting | Accuracy | n |")
        lines.append("|-------|--------:|--------:|--:|")
        for model in models:
            for setting in sorted(set(r["setting"] for r in ood_rows)):
                sub = [r for r in ood_rows
                       if r["model"] == model and r["setting"] == setting]
                if sub:
                    acc = sum(1 for r in sub
                              if _sign_correct(r["true_delta"], r["model_delta"])) / len(sub)
                    lines.append(f"| {model} | {setting} | {acc*100:.1f}% | {len(sub)} |")
    else:
        lines.append("No eligible OOD cells scored.")
    lines.append("")

    # Write report
    report_text = "\n".join(lines)
    out_path = FORK_DIR / "fork_scoring_report.md"
    out_path.write_text(report_text)
    logger.info(f"Report written to {out_path}")
    print(report_text)


if __name__ == "__main__":
    main()
