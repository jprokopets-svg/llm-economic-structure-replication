"""Dual-contrast fork scoring: ARM2−ARM3 (preregistered) + ARM2−ARM1 (placebo-controlled).

Produces outputs/fork_run/fork_scoring_dual_report.md with both contrasts
side by side for all registered outcomes.

Usage:
    python scripts/score_fork_dual.py
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

BASELINES_MAP = {
    "phillips_slope": {"world1": 0.4, "world2": 0.4, "world3": 0.5, "world4": 0.2},
    "taylor_phi_pi": {"world1": 1.5, "world2": 1.5, "world3": 2.0, "world4": 1.5},
    "is_sensitivity": {"world1": 0.6, "world2": 0.6, "world3": 0.6, "world4": 0.6},
    "wage_gap_slope": {"world4": 0.5},
}

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
    last_record = {}
    with open(CHECKPOINT_FILE) as f:
        for line in f:
            r = json.loads(line)
            last_record[r["call_id"]] = r
    return [r for r in last_record.values()
            if r.get("parse_success") and r.get("forecast")]


def index_forecasts(forecasts):
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

    models = sorted(set(f["model"] for f in forecasts))
    logger.info(f"GT cells: {len(gt)}, Forecasts: {len(forecasts)}, Models: {models}")

    # ── Build scored rows with both contrasts ──
    rows = []
    for cell_key, cell_data in gt.items():
        world = cell_data["world"]
        param = cell_data["param"]
        setting = cell_data["setting"]
        seed = cell_data["seed"]
        is_ood = cell_data.get("is_ood", False)
        trend_key = f"{world}__s{seed}"
        cell_trends = trends.get(trend_key, {})
        baseline_val = BASELINES_MAP[param][world]

        for model in models:
            arm3_fc = fc.get(("arm3_baseline", world, None, None, seed, model), {})
            arm2_fc = fc.get(("arm2_change", world, param, setting, seed, model), {})
            arm1_fc = fc.get(("arm1_placebo", world, param, None, seed, model), {})
            if not arm1_fc:
                arm1_fc = fc.get(("arm1_placebo", world, param, baseline_val, seed, model), {})

            for var_h, gt_info in cell_data["forecasts"].items():
                true_delta = gt_info["true_delta"]
                mc_se = gt_info["mc_se"]
                eligible = gt_info["eligible"]
                var = var_h.rsplit("_", 1)[0]
                horizon = int(var_h.rsplit("_", 1)[1])

                a2 = arm2_fc.get(var_h)
                a3 = arm3_fc.get(var_h)
                a1 = arm1_fc.get(var_h)

                # Contrast 1: ARM2 - ARM3 (preregistered)
                delta_23 = (a2 - a3) if (a2 is not None and a3 is not None) else None
                # Contrast 2: ARM2 - ARM1 (placebo-controlled)
                delta_21 = (a2 - a1) if (a2 is not None and a1 is not None) else None
                # Placebo: ARM1 - ARM3
                placebo = (a1 - a3) if (a1 is not None and a3 is not None) else None

                var_trend = cell_trends.get(var, {})
                trend_8q = var_trend.get("trend_8q", 0)
                trend_4q = var_trend.get("trend_4q", 0)
                rolling_sd = var_trend.get("rolling_sd", 1.0)
                conflict_class = classify_conflict(trend_8q, true_delta, rolling_sd)

                rows.append({
                    "model": model, "world": world, "param": param,
                    "setting": setting, "seed": seed,
                    "var": var, "var_h": var_h, "horizon": horizon,
                    "true_delta": true_delta, "mc_se": mc_se,
                    "eligible": eligible,
                    "delta_23": delta_23, "delta_21": delta_21,
                    "placebo": placebo,
                    "trend_8q": trend_8q, "trend_4q": trend_4q,
                    "rolling_sd": rolling_sd,
                    "conflict_class": conflict_class,
                    "cluster_id": f"{world}__{param}__{setting}",
                    "is_ood": is_ood,
                })

    logger.info(f"Scored rows: {len(rows)}")

    # Count rows with data for each contrast
    has_23 = sum(1 for r in rows if r["delta_23"] is not None and r["eligible"] and not r["is_ood"])
    has_21 = sum(1 for r in rows if r["delta_21"] is not None and r["eligible"] and not r["is_ood"])
    logger.info(f"Eligible rows with ARM2-ARM3: {has_23}")
    logger.info(f"Eligible rows with ARM2-ARM1: {has_21}")

    # ── Build report ──
    lines = []
    lines.append("# Fork Experiment Dual-Contrast Scoring Report")
    lines.append("")
    lines.append("Two contrasts: **ARM2−ARM3** (preregistered original) and "
                 "**ARM2−ARM1** (placebo-controlled, post-hoc motivated by "
                 "the placebo check failure). Both on eligible cells "
                 "(|true_delta| > 2×MC_SE).")
    lines.append("")

    # ── 1. Placebo summary ──
    lines.append("## 1. Placebo check summary")
    lines.append("")
    lines.append("| Model | Mean |ARM1−ARM3| | Share |placebo| > |td|/2 | n |")
    lines.append("|-------|-------------------:|-------------------------------------------:|---:|")
    for model in models:
        p_rows = [r for r in rows if r["model"] == model
                  and r["placebo"] is not None and r["eligible"]
                  and not r["is_ood"]]
        if p_rows:
            mean_abs = float(np.mean([abs(r["placebo"]) for r in p_rows]))
            frac = float(np.mean([1 if abs(r["placebo"]) > abs(r["true_delta"]) / 2 else 0
                                  for r in p_rows]))
            lines.append(f"| {model} | {mean_abs:.4f} | {frac*100:.1f}% | {len(p_rows)} |")
    lines.append("")

    # ── 2. Primary directional accuracy (both contrasts) ──
    lines.append("## 2. Directional accuracy: ARM2−ARM3 vs ARM2−ARM1")
    lines.append("")
    lines.append("| Model | ARM2−ARM3 acc [95% CI] | n | ARM2−ARM1 acc [95% CI] | n |")
    lines.append("|-------|----------------------:|---:|----------------------:|---:|")

    for model in models:
        # ARM2-ARM3
        r23 = [r for r in rows if r["model"] == model and r["delta_23"] is not None
               and r["eligible"] and not r["is_ood"]]
        if r23:
            c23 = [1 if _sign_correct(r["true_delta"], r["delta_23"]) else 0 for r in r23]
            ci23 = clustered_bootstrap_ci(c23, [r["cluster_id"] for r in r23])
            s23 = f"{ci23['mean']*100:.1f}% [{ci23['ci_low']*100:.1f}, {ci23['ci_high']*100:.1f}]"
            n23 = ci23["n"]
        else:
            s23 = "—"
            n23 = 0

        # ARM2-ARM1
        r21 = [r for r in rows if r["model"] == model and r["delta_21"] is not None
               and r["eligible"] and not r["is_ood"]]
        if r21:
            c21 = [1 if _sign_correct(r["true_delta"], r["delta_21"]) else 0 for r in r21]
            ci21 = clustered_bootstrap_ci(c21, [r["cluster_id"] for r in r21])
            s21 = f"{ci21['mean']*100:.1f}% [{ci21['ci_low']*100:.1f}, {ci21['ci_high']*100:.1f}]"
            n21 = ci21["n"]
        else:
            s21 = "—"
            n21 = 0

        lines.append(f"| {model} | {s23} | {n23} | {s21} | {n21} |")
    lines.append("")

    # ── 3. Response ratio (both contrasts) ──
    lines.append("## 3. Response ratio: both contrasts")
    lines.append("")
    lines.append("| Model | Contrast | Median | IQR | |ratio|<0.1 | |ratio|>2 | n |")
    lines.append("|-------|----------|-------:|----:|----------:|----------:|---:|")

    for model in models:
        for contrast_name, delta_key in [("ARM2−ARM3", "delta_23"), ("ARM2−ARM1", "delta_21")]:
            sub = [r for r in rows if r["model"] == model and r[delta_key] is not None
                   and r["eligible"] and not r["is_ood"] and abs(r["true_delta"]) > 1e-10]
            if sub:
                ratios = np.array([r[delta_key] / r["true_delta"] for r in sub])
                med = float(np.median(ratios))
                q25 = float(np.percentile(ratios, 25))
                q75 = float(np.percentile(ratios, 75))
                frac_small = float(np.mean(np.abs(ratios) < 0.1))
                frac_big = float(np.mean(np.abs(ratios) > 2.0))
                lines.append(
                    f"| {model} | {contrast_name} | {med:.2f} "
                    f"| [{q25:.2f}, {q75:.2f}] "
                    f"| {frac_small*100:.1f}% | {frac_big*100:.1f}% | {len(sub)} |")
    lines.append("")

    # ── 4. Sign-reversal tracking (both contrasts) ──
    lines.append("## 4. Sign-reversal tracking (h4→h8 flips)")
    lines.append("")
    lines.append("| Model | Param | Var | Contrast | GT flips | Model flips | Match % | n |")
    lines.append("|-------|-------|-----|----------|--------:|:-----------:|-------:|---:|")

    for model in models:
        for param, var in SIGN_FLIP_PAIRS:
            for contrast_name, delta_key in [("23", "delta_23"), ("21", "delta_21")]:
                h4 = {(r["world"], r["setting"], r["seed"]): r
                      for r in rows if r["model"] == model and r["param"] == param
                      and r["var"] == var and r["horizon"] == 4
                      and r[delta_key] is not None and not r["is_ood"]}
                h8 = {(r["world"], r["setting"], r["seed"]): r
                      for r in rows if r["model"] == model and r["param"] == param
                      and r["var"] == var and r["horizon"] == 8
                      and r[delta_key] is not None and not r["is_ood"]}
                common = set(h4.keys()) & set(h8.keys())
                gt_flips = 0
                model_flips = 0
                for ck in common:
                    if (h4[ck]["true_delta"] > 0) != (h8[ck]["true_delta"] > 0):
                        gt_flips += 1
                        if (h4[ck][delta_key] > 0) != (h8[ck][delta_key] > 0):
                            model_flips += 1
                if gt_flips > 0:
                    lines.append(
                        f"| {model} | {param} | {var} | {contrast_name} "
                        f"| {gt_flips} | {model_flips} "
                        f"| {model_flips/gt_flips*100:.0f}% | {len(common)} |")
    lines.append("")

    # ── 5. Conflict classification (both contrasts) ──
    lines.append("## 5. Conflict classification (exploratory, both contrasts)")
    lines.append("")

    # Cell counts
    for cls in ["ALIGNED", "CONFLICT", "NEUTRAL"]:
        n = sum(1 for r in rows if r["conflict_class"] == cls
                and r["eligible"] and not r["is_ood"])
        lines.append(f"- {cls}: {n}")
    lines.append("")

    lines.append("| Model | Class | ARM2−ARM3 acc [CI] | n | ARM2−ARM1 acc [CI] | n |")
    lines.append("|-------|-------|-------------------:|---:|-------------------:|---:|")

    for model in models:
        for cls in ["ALIGNED", "CONFLICT", "NEUTRAL"]:
            # ARM2-ARM3
            r23 = [r for r in rows if r["model"] == model and r["conflict_class"] == cls
                   and r["delta_23"] is not None and r["eligible"] and not r["is_ood"]]
            if r23:
                c23 = [1 if _sign_correct(r["true_delta"], r["delta_23"]) else 0 for r in r23]
                ci23 = clustered_bootstrap_ci(c23, [r["cluster_id"] for r in r23])
                s23 = f"{ci23['mean']*100:.1f}% [{ci23['ci_low']*100:.1f}, {ci23['ci_high']*100:.1f}]"
                n23 = ci23["n"]
            else:
                s23 = "—"
                n23 = 0

            # ARM2-ARM1
            r21 = [r for r in rows if r["model"] == model and r["conflict_class"] == cls
                   and r["delta_21"] is not None and r["eligible"] and not r["is_ood"]]
            if r21:
                c21 = [1 if _sign_correct(r["true_delta"], r["delta_21"]) else 0 for r in r21]
                ci21 = clustered_bootstrap_ci(c21, [r["cluster_id"] for r in r21])
                s21 = f"{ci21['mean']*100:.1f}% [{ci21['ci_low']*100:.1f}, {ci21['ci_high']*100:.1f}]"
                n21 = ci21["n"]
            else:
                s21 = "—"
                n21 = 0

            lines.append(f"| {model} | {cls} | {s23} | {n23} | {s21} | {n21} |")
    lines.append("")

    # ── 6. OOD (both contrasts) ──
    lines.append("## 6. OOD extension (both contrasts)")
    lines.append("")
    lines.append("| Model | Setting | ARM2−ARM3 | ARM2−ARM1 | n |")
    lines.append("|-------|--------:|--------:|--------:|---:|")
    for model in models:
        for setting in [0.05, 1.0]:
            r23 = [r for r in rows if r["model"] == model and r["setting"] == setting
                   and r["is_ood"] and r["eligible"] and r["delta_23"] is not None]
            r21 = [r for r in rows if r["model"] == model and r["setting"] == setting
                   and r["is_ood"] and r["eligible"] and r["delta_21"] is not None]
            a23 = sum(1 for r in r23 if _sign_correct(r["true_delta"], r["delta_23"])) / len(r23) * 100 if r23 else float("nan")
            a21 = sum(1 for r in r21 if _sign_correct(r["true_delta"], r["delta_21"])) / len(r21) * 100 if r21 else float("nan")
            n = max(len(r23), len(r21))
            lines.append(f"| {model} | {setting} | {a23:.1f}% | {a21:.1f}% | {n} |")
    lines.append("")

    # Write
    report_text = "\n".join(lines)
    out_path = FORK_DIR / "fork_scoring_dual_report.md"
    out_path.write_text(report_text)
    logger.info(f"Report written to {out_path}")
    print(report_text)


if __name__ == "__main__":
    main()
