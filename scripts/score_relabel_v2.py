"""Score the v4 (v2 template) relabel run through the per-seed pipeline
and produce the paired relabel-minus-infer table with world/param
breakdown.

Appends a "Relabel v2 (final)" section to strict_rescore_report.md.
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

sys.path.insert(0, str(V2_ROOT / "src"))

_env_data = os.environ.get("LMM2_DATA_ROOT")
if _env_data:
    V2_ROOT = Path(_env_data).resolve()
_env_v1 = os.environ.get("LMM2_V1_ROOT")
if _env_v1:
    V1_ROOT = Path(_env_v1).resolve()
    

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


from lmm2.relabel_template import RELABEL_VARIABLE_INVERSE
from strict_rescore import (
    MODELS, HISTORY_SEEDS, EXCLUDED_GT_KEYS, PARAMS, BASELINES,
    compute_cell_stability, build_rows, compute_var_forecasts_per_seed,
    filter_by_tier,
    clustered_bootstrap_ci, permutation_baseline, fmt_pct, fmt_ci,
)
from run_analysis import load_forecasts


V4_MODELS = ["Claude Sonnet 4.6", "GPT-5.5", "Gemini 3.5 Flash"]
CONDITIONS = ["told", "infer", "relabel"]


def _load_relabel_records(relabel_run_dir: Path) -> list[dict]:
    """Load v4 checkpoint, un-relabel forecast keys."""
    cp = relabel_run_dir / "checkpoint.jsonl"
    records = []
    with open(cp) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error"):
                continue
            if not r.get("parse_success"):
                continue
            fc = r.get("forecast") or {}
            new_fc = {}
            for k, v in fc.items():
                if "_" not in k:
                    continue
                head, tail = k.rsplit("_", 1)
                if head in RELABEL_VARIABLE_INVERSE:
                    new_fc[f"{RELABEL_VARIABLE_INVERSE[head]}_{tail}"] = v
            r["forecast"] = new_fc
            records.append(r)
    return records


def _parse_stats(relabel_run_dir: Path) -> dict:
    cp = relabel_run_dir / "checkpoint.jsonl"
    stats = defaultdict(lambda: {"ok": 0, "fail": 0, "error": 0,
                                 "cost": 0.0, "tokens_in": 0, "tokens_out": 0})
    with open(cp) as f:
        for line in f:
            r = json.loads(line)
            m = r.get("model", "?")
            if r.get("error"):
                stats[m]["error"] += 1
            elif r.get("parse_success"):
                stats[m]["ok"] += 1
            else:
                stats[m]["fail"] += 1
            stats[m]["cost"] += r.get("cost_usd", 0.0) or 0.0
            stats[m]["tokens_in"] += r.get("tokens_in", 0) or 0
            stats[m]["tokens_out"] += r.get("tokens_out", 0) or 0
    return dict(stats)


def _paired_diff(rows_a, rows_b, n_boot=10000, seed=42,
                  near_zero=0.01):
    """CI on accuracy(a) - accuracy(b). Cluster-paired bootstrap."""
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
                "ci_high": float("nan"), "n_a": len(corr_a), "n_b": len(corr_b)}
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
        "acc_a": obs_a, "acc_b": obs_b, "diff": obs_diff,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_a": len(corr_a), "n_b": len(corr_b),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel-run", default=str(V2_ROOT / "data/processed"))
    parser.add_argument("--per-seed-gt",
                       default=str(V2_ROOT / "data/processed/ground_truth_per_seed.json"))
    parser.add_argument("--seed42-gt", default=None)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()

    gt_ps = json.loads(Path(args.per_seed_gt).read_text())
    gt42_path = args.seed42_gt or str(V2_ROOT / "data/processed/ground_truth.json")
    gt42 = json.loads(Path(gt42_path).read_text())

    told_infer = load_forecasts()
    told_infer = [r for r in told_infer if r.get("condition") != "relabel"]
    relabel = _load_relabel_records(Path(args.relabel_run))
    logger.info(f"Loaded {len(told_infer)} told/infer + {len(relabel)} relabel")

    stability = compute_cell_stability(gt_ps)
    var_forecasts = compute_var_forecasts_per_seed()

    # Extend CONDITIONS in strict_rescore for build_rows.
    import strict_rescore as sr
    sr.CONDITIONS = CONDITIONS
    all_rows = build_rows(told_infer + relabel, gt42, gt_ps,
                          var_forecasts, stability)
    logger.info(f"Assembled {len(all_rows)} rows")

    tiers = {
        "all": all_rows,
        "moderate": filter_by_tier(all_rows, "moderate"),
        "strict": filter_by_tier(all_rows, "strict"),
    }
    for k, v in tiers.items():
        logger.info(f"  tier={k}: {len(v)} rows")

    # === Three-condition acc + permutation ===
    stats_by_tier = {}
    perm_by_tier = {}
    for tier, trs in tiers.items():
        stats_by_tier[tier] = {}
        perm_by_tier[tier] = {}
        for model in V4_MODELS:
            for cond in CONDITIONS:
                sub = [r for r in trs
                       if r["source"] == model and r["condition"] == cond]
                if not sub:
                    continue
                ci = clustered_bootstrap_ci(
                    sub, gt_field="td_per_seed",
                    n_boot=args.n_boot, seed=42,
                )
                perm = permutation_baseline(
                    sub, gt_field="td_per_seed",
                    n_perm=args.n_perm, seed=42,
                )
                stats_by_tier[tier][(model, cond)] = ci
                perm_by_tier[tier][(model, cond)] = perm
        logger.info(f"tier={tier} scored")

    # === Paired relabel-infer with CIs ===
    paired_by_tier = {}
    paired_by_wp = defaultdict(list)  # per (model, world, param, tier)
    for tier, trs in tiers.items():
        paired_by_tier[tier] = {}
        for model in V4_MODELS:
            rows_r = [r for r in trs if r["source"] == model and r["condition"] == "relabel"]
            rows_i = [r for r in trs if r["source"] == model and r["condition"] == "infer"]
            paired_by_tier[tier][model] = _paired_diff(
                rows_r, rows_i, n_boot=args.n_boot, seed=42,
            )
        # World/param breakdown (all tier only, to keep n manageable).
        if tier != "all":
            continue
        for model in V4_MODELS:
            for world in ["world1", "world2", "world3", "world4"]:
                for param in PARAMS.keys():
                    if world not in PARAMS[param]:
                        continue
                    rows_r = [r for r in trs
                              if r["source"] == model and r["condition"] == "relabel"
                              and r["world"] == world and r["param"] == param]
                    rows_i = [r for r in trs
                              if r["source"] == model and r["condition"] == "infer"
                              and r["world"] == world and r["param"] == param]
                    if not rows_r or not rows_i:
                        continue
                    paired_by_wp[model].append((
                        world, param,
                        _paired_diff(rows_r, rows_i, n_boot=args.n_boot,
                                     seed=42),
                    ))

    parse_stats = _parse_stats(Path(args.relabel_run))

    # === Report ===
    lines = []
    lines.append("\n## Relabel v2 (final): three-condition table + paired "
                 "differences")
    lines.append("")
    lines.append(f"Run: `{Path(args.relabel_run).name}` (v4). Uses "
                 "`lmm2.relabel_template.build_relabel_prompt` — hand-written "
                 "static templates + whitelist verifier, no substitution "
                 "pass. Bootstrap 10K reps, permutation 10K reps.")
    lines.append("")

    # Parse rates + spend.
    lines.append("### Parse rates and spend (v4 run)")
    lines.append("")
    lines.append("| Model | ok | fail | error | parse rate | tokens_in | "
                 "tokens_out | cost |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    total_cost = 0.0
    for m in V4_MODELS:
        s = parse_stats.get(m, {"ok": 0, "fail": 0, "error": 0, "cost": 0,
                               "tokens_in": 0, "tokens_out": 0})
        denom = s["ok"] + s["fail"]
        rate = s["ok"] / denom * 100 if denom else 0.0
        lines.append(
            f"| {m} | {s['ok']} | {s['fail']} | {s['error']} | {rate:.1f}% "
            f"| {s['tokens_in']:,} | {s['tokens_out']:,} | ${s['cost']:.2f} |"
        )
        total_cost += s["cost"]
    lines.append(f"| **Total** | | | | | | | **${total_cost:.2f}** |")
    lines.append("")

    # Three-condition tables per tier.
    lines.append("### Accuracy per condition per tier (per-seed GT, 95% CIs)")
    lines.append("")
    for tier in ["all", "moderate", "strict"]:
        lines.append(f"#### Tier: {tier}")
        lines.append("")
        lines.append("| Model | Told acc [CI] | Infer acc [CI] | "
                     "Relabel acc [CI] | Told n | Infer n | Relabel n |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for model in V4_MODELS:
            row = []
            for cond in CONDITIONS:
                ci = stats_by_tier[tier].get((model, cond))
                if ci is None:
                    row.append(("n/a", 0))
                else:
                    row.append((
                        f"{fmt_pct(ci['observed'])} "
                        f"{fmt_ci(ci['ci_low'], ci['ci_high'])}",
                        ci["n"],
                    ))
            lines.append(
                f"| {model} | {row[0][0]} | {row[1][0]} | {row[2][0]} "
                f"| {row[0][1]} | {row[1][1]} | {row[2][1]} |"
            )
        lines.append("")

    # Paired relabel-infer per tier.
    lines.append("### Paired Relabel − Infer (same-cluster resampling, 95% CI)")
    lines.append("")
    lines.append("| Model | Tier | Relabel acc | Infer acc | Diff [95% CI] | n |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for model in V4_MODELS:
        for tier in ["all", "moderate", "strict"]:
            d = paired_by_tier[tier][model]
            diff_s = (
                "n/a" if d["diff"] != d["diff"] else
                f"{d['diff']*100:+.1f} pp "
                f"[{d['ci_low']*100:+.1f}, {d['ci_high']*100:+.1f}]"
            )
            lines.append(
                f"| {model} | {tier} | {fmt_pct(d['acc_a'])} "
                f"| {fmt_pct(d['acc_b'])} | {diff_s} | {d['n_a']} |"
            )
    lines.append("")

    # World/param breakdown (all tier).
    lines.append("### World × Param breakdown of Relabel − Infer (all tier)")
    lines.append("")
    lines.append("| Model | World | Param | Relabel acc | Infer acc | Diff [95% CI] | n |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for model in V4_MODELS:
        for world, param, d in paired_by_wp[model]:
            diff_s = (
                "n/a" if d["diff"] != d["diff"] else
                f"{d['diff']*100:+.1f} pp "
                f"[{d['ci_low']*100:+.1f}, {d['ci_high']*100:+.1f}]"
            )
            lines.append(
                f"| {model} | {world} | {param} | {fmt_pct(d['acc_a'])} "
                f"| {fmt_pct(d['acc_b'])} | {diff_s} | {d['n_a']} |"
            )
    lines.append("")

    # Permutation baselines.
    lines.append("### Relabel-condition permutation baselines")
    lines.append("")
    lines.append("| Model | Tier | Observed | Perm mean | Perm 95% CI | n |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for tier in ["all", "moderate", "strict"]:
        for model in V4_MODELS:
            perm = perm_by_tier[tier].get((model, "relabel"))
            if perm is None:
                continue
            lines.append(
                f"| {model} | {tier} | {fmt_pct(perm['observed'])} "
                f"| {fmt_pct(perm['perm_mean'])} "
                f"| {fmt_ci(perm['perm_ci_low'], perm['perm_ci_high'])} "
                f"| {perm['n']} |"
            )
    lines.append("")

    report_path = V2_ROOT / "data/processed/strict_rescore_report.md"
    existing = report_path.read_text()
    marker = "## Relabel v2 (final): three-condition table + paired"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    report_path.write_text(existing + "\n".join(lines))
    logger.info(f"Appended to {report_path}")

    # JSON dump.
    (V2_ROOT / "data/processed/relabel_v2_final.json").write_text(json.dumps({
        "parse_stats": parse_stats,
        "total_cost": total_cost,
        "paired_by_tier": {
            tier: {m: v for m, v in paired.items()}
            for tier, paired in paired_by_tier.items()
        },
        "paired_by_wp": {
            m: [{"world": w, "param": p, **d} for w, p, d in v]
            for m, v in paired_by_wp.items()
        },
        "per_tier": {
            tier: {
                f"{m}__{c}": {
                    "accuracy": ci["observed"], "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"], "n": ci["n"],
                    "perm_mean": perm_by_tier[tier].get((m, c), {}).get("perm_mean"),
                    "perm_ci_low": perm_by_tier[tier].get((m, c), {}).get("perm_ci_low"),
                    "perm_ci_high": perm_by_tier[tier].get((m, c), {}).get("perm_ci_high"),
                }
                for (m, c), ci in stats_by_tier[tier].items()
            }
            for tier in tiers
        },
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
