"""Relabel-minus-infer paired clustered-bootstrap CIs.

Appends a section to strict_rescore_report.md showing the paired
difference for each model, at each tier, with the same cluster
resamples used for both conditions.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


V2_ROOT = Path(__file__).resolve().parent.parent
V1_ROOT = V2_ROOT
sys.path.insert(0, str(V1_ROOT / "src"))
sys.path.insert(0, str(V2_ROOT / "src"))

_env_data = os.environ.get("LMM2_DATA_ROOT")
if _env_data:
    V2_ROOT = Path(_env_data).resolve()
_env_v1 = os.environ.get("LMM2_V1_ROOT")
if _env_v1:
    V1_ROOT = Path(_env_v1).resolve()
    sys.path.insert(0, str(V1_ROOT / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


from strict_rescore import (
    MODELS, HISTORY_SEEDS, EXCLUDED_GT_KEYS,
    compute_cell_stability, build_rows, compute_var_forecasts_per_seed,
    filter_by_tier,
)
from run_analysis import load_forecasts
from score_relabel import _load_relabel_records


def paired_diff_ci(rows_a, rows_b, n_boot=10000, seed=42,
                    near_zero=0.01):
    """CI on accuracy(a) - accuracy(b) via cluster-paired bootstrap.

    Filters near-zero rows first. Uses union of clusters across both
    conditions and resamples them together.
    """
    def _prep(rows):
        filt = [r for r in rows if abs(r["td_per_seed"]) >= near_zero]
        corr = np.array(
            [1 if ((r["td_per_seed"] > 0 and r["model_delta"] > 0)
                   or (r["td_per_seed"] < 0 and r["model_delta"] < 0)) else 0
             for r in filt], dtype=np.int8,
        )
        clust = defaultdict(list)
        for i, r in enumerate(filt):
            clust[r["cluster_id"]].append(i)
        return filt, corr, {c: np.array(v) for c, v in clust.items()}

    filt_a, corr_a, idx_a = _prep(rows_a)
    filt_b, corr_b, idx_b = _prep(rows_b)
    if len(corr_a) == 0 or len(corr_b) == 0:
        return {"acc_a": float("nan"), "acc_b": float("nan"),
                "diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_a": len(corr_a),
                "n_b": len(corr_b)}

    obs_a = float(corr_a.mean())
    obs_b = float(corr_b.mean())
    obs_diff = obs_a - obs_b
    all_clusters = sorted(set(idx_a) | set(idx_b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        s = rng.integers(0, len(all_clusters), size=len(all_clusters))
        picks_a, picks_b = [], []
        for j in s:
            c = all_clusters[j]
            if c in idx_a:
                picks_a.append(idx_a[c])
            if c in idx_b:
                picks_b.append(idx_b[c])
        if picks_a and picks_b:
            diffs[k] = (corr_a[np.concatenate(picks_a)].mean()
                        - corr_b[np.concatenate(picks_b)].mean())
        else:
            diffs[k] = np.nan
    diffs = diffs[~np.isnan(diffs)]
    return {
        "acc_a": obs_a, "acc_b": obs_b,
        "diff": obs_diff,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_a": len(corr_a), "n_b": len(corr_b),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel-run",
                       default=str(V2_ROOT / "data/processed"))
    parser.add_argument("--per-seed-gt",
                       default=str(V2_ROOT / "data/processed/ground_truth_per_seed.json"))
    parser.add_argument("--seed42-gt", default=None)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    gt_ps = json.loads(Path(args.per_seed_gt).read_text())
    gt42_path = args.seed42_gt or str(V2_ROOT / "data/processed/ground_truth.json")
    gt42 = json.loads(Path(gt42_path).read_text())

    told_infer = load_forecasts()
    told_infer = [r for r in told_infer if r.get("condition") != "relabel"]
    relabel_recs = _load_relabel_records(Path(args.relabel_run))
    forecasts = told_infer + relabel_recs

    stability = compute_cell_stability(gt_ps)
    var_forecasts = compute_var_forecasts_per_seed()

    import strict_rescore as sr
    sr.CONDITIONS = ["told", "infer", "relabel"]
    all_rows = build_rows(forecasts, gt42, gt_ps, var_forecasts, stability)

    tiers = {
        "all": all_rows,
        "moderate": filter_by_tier(all_rows, "moderate"),
        "strict": filter_by_tier(all_rows, "strict"),
    }

    lines = []
    lines.append("\n## Relabel − Infer (paired cluster-bootstrap CIs)")
    lines.append("")
    lines.append(f"Per model, per tier. Bootstrap {args.n_boot:,} reps, "
                 "clusters = `{world}__{param}__{setting}`, paired "
                 "resampling across relabel and infer.")
    lines.append("")
    lines.append("| Model | Tier | Relabel acc | Infer acc | "
                 "Relabel − Infer [95% CI] | n relabel | n infer |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    fmt = lambda x: "n/a" if x != x else f"{x*100:.1f}%"

    for model in MODELS:
        for tier in ["all", "moderate", "strict"]:
            trows = tiers[tier]
            rows_r = [r for r in trows
                      if r["source"] == model and r["condition"] == "relabel"]
            rows_i = [r for r in trows
                      if r["source"] == model and r["condition"] == "infer"]
            r = paired_diff_ci(rows_r, rows_i, n_boot=args.n_boot, seed=42)
            diff_s = ("n/a" if r["diff"] != r["diff"] else
                      f"{r['diff']*100:+.1f} pp "
                      f"[{r['ci_low']*100:+.1f}, {r['ci_high']*100:+.1f}]")
            lines.append(f"| {model} | {tier} | {fmt(r['acc_a'])} "
                         f"| {fmt(r['acc_b'])} | {diff_s} "
                         f"| {r['n_a']} | {r['n_b']} |")

    lines.append("")

    # Compact interpretation footnote.
    lines.append("**Interpretation.** A positive difference with a CI that "
                 "excludes zero means the RELABEL condition beats INFER for "
                 "that model on that tier — the model does *better* when "
                 "the economic vocabulary is stripped out and replaced with "
                 "neutral system-dynamics tokens (X1..X7, coupling A/B/C, "
                 "controller). A negative difference means it does worse.")
    lines.append("")

    report_path = V2_ROOT / "data/processed/strict_rescore_report.md"
    existing = report_path.read_text()
    marker = "## Relabel − Infer (paired cluster-bootstrap CIs)"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    report_path.write_text(existing + "\n".join(lines))
    logger.info(f"Appended relabel-minus-infer to {report_path}")


if __name__ == "__main__":
    main()
