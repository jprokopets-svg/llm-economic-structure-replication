"""Independent rescore of the v2 relabel run.

Deliberately does NOT import score_relabel_v2 or strict_rescore's
scoring functions. Instead, reads the v4 checkpoint, per-seed GT, and
runs its own sign-scoring + bootstrap. Reports:

  - Independent per-model relabel accuracy at each tier
  - Independent paired relabel-minus-infer per model at each tier
  - Comparison to score_relabel_v2's output (from
    outputs/analysis/relabel_v2_final.json) — flags any deviation
    beyond 0.5pp.
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
    

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Local minimal constants (deliberately duplicated so we do NOT import).
HORIZONS = [1, 4, 8]
HISTORY_SEEDS = list(range(10))
NEAR_ZERO = 0.01
STRICT_THRESHOLD = 0.8
MODERATE_THRESHOLD = 0.6
V4_MODELS = ["Claude Sonnet 4.6", "GPT-5.5", "Gemini 3.5 Flash"]

# Variable inverse mapping — must match RELABEL_VARIABLE_MAP.
_RELABEL_INVERSE = {
    "X1": "y", "X2": "pi", "X3": "r", "X4": "e",
    "X5": "u", "X6": "w", "X7": "u_natural",
}

EXCLUDED = {"world3__phillips_slope__0.7", "world3__phillips_slope__0.9"}

BASELINES = {
    "phillips_slope": {"world1": 0.4, "world2": 0.4, "world3": 0.5, "world4": 0.2},
    "taylor_phi_pi": {"world1": 1.5, "world2": 1.5, "world3": 2.0, "world4": 1.5},
    "is_sensitivity": {"world1": 0.6, "world2": 0.6, "world3": 0.6, "world4": 0.6},
    "wage_gap_slope": {"world4": 0.5},
}
PARAMS = {
    "phillips_slope": {"world1": [0.1,0.2,0.4,0.6,0.8], "world2": [0.1,0.2,0.4,0.6,0.8],
                       "world3": [0.1,0.3,0.5,0.7,0.9], "world4": [0.05,0.1,0.2,0.3,0.5]},
    "taylor_phi_pi": {"world1": [1.1,1.3,1.5,2.0,2.5], "world2": [1.1,1.3,1.5,2.0,2.5],
                      "world3": [1.1,1.5,2.0,2.5,3.0], "world4": [1.1,1.3,1.5,2.0,2.5]},
    "is_sensitivity": {"world1": [0.2,0.4,0.6,0.8,1.0], "world2": [0.2,0.4,0.6,0.8,1.0],
                       "world3": [0.2,0.4,0.6,0.8,1.0], "world4": [0.2,0.4,0.6,0.8,1.0]},
    "wage_gap_slope": {"world4": [0.1,0.3,0.5,0.7,1.0]},
}


def _load_checkpoint(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def _index_forecasts(records, model_filter=None, condition_filter=None,
                     unrelabel=False):
    """Return {(model, world, param, setting, condition, seed) -> {var_h: point}}."""
    fc = {}
    for r in records:
        if r.get("error") or not r.get("parse_success"):
            continue
        model = r.get("model")
        if model_filter and model not in model_filter:
            continue
        cond = r.get("condition")
        if condition_filter and cond not in condition_filter:
            continue
        seed = r.get("seed")
        if seed is None:
            continue
        key = (model, r["world"], r["param"], r["setting"], cond, int(seed))
        pts = {}
        for var_h, v in (r.get("forecast") or {}).items():
            if not isinstance(v, dict) or "point" not in v:
                continue
            if unrelabel:
                # Convert X_k → orig_k
                if "_" not in var_h:
                    continue
                head, tail = var_h.rsplit("_", 1)
                if head in _RELABEL_INVERSE:
                    pts[f"{_RELABEL_INVERSE[head]}_{tail}"] = v["point"]
            else:
                pts[var_h] = v["point"]
        fc[key] = pts
    return fc


def _compute_stability(gt_ps):
    stab = {}
    for param, worlds in PARAMS.items():
        for world, settings in worlds.items():
            base = BASELINES[param][world]
            base_stem = f"{world}__{param}__{base}"
            if base_stem in EXCLUDED:
                continue
            for setting in settings:
                if setting == base:
                    continue
                mod_stem = f"{world}__{param}__{setting}"
                if mod_stem in EXCLUDED:
                    continue
                probe = gt_ps.get(f"{mod_stem}__s0", {})
                for var_h in probe:
                    deltas = []
                    for s in HISTORY_SEEDS:
                        gm = gt_ps.get(f"{mod_stem}__s{s}", {}).get(var_h)
                        gb = gt_ps.get(f"{base_stem}__s{s}", {}).get(var_h)
                        if gm is None or gb is None:
                            continue
                        deltas.append(gm - gb)
                    if len(deltas) != len(HISTORY_SEEDS):
                        continue
                    pos = sum(1 for d in deltas if d > 0)
                    stab[(world, param, setting, var_h)] = max(pos, len(deltas)-pos) / len(deltas)
    return stab


def _build_rows(fc_infer, fc_relabel, gt_ps, stability):
    """Build per-row records with td_per_seed, model_delta, correct."""
    rows = []
    for param, worlds in PARAMS.items():
        for world, settings in worlds.items():
            base = BASELINES[param][world]
            base_stem = f"{world}__{param}__{base}"
            if base_stem in EXCLUDED:
                continue
            for setting in settings:
                if setting == base:
                    continue
                mod_stem = f"{world}__{param}__{setting}"
                if mod_stem in EXCLUDED:
                    continue
                for model in V4_MODELS:
                    for cond, fc_by_cond in [
                        ("infer", fc_infer),
                        ("relabel", fc_relabel),
                    ]:
                        for seed in HISTORY_SEEDS:
                            fbase = fc_by_cond.get(
                                (model, world, param, base, cond, seed)
                            )
                            fmod = fc_by_cond.get(
                                (model, world, param, setting, cond, seed)
                            )
                            if fbase is None or fmod is None:
                                continue
                            gt_base = gt_ps.get(f"{base_stem}__s{seed}", {})
                            gt_mod = gt_ps.get(f"{mod_stem}__s{seed}", {})
                            for var_h in fbase:
                                if var_h not in fmod:
                                    continue
                                if var_h not in gt_base or var_h not in gt_mod:
                                    continue
                                cell = (world, param, setting, var_h)
                                if cell not in stability:
                                    continue
                                td = gt_mod[var_h] - gt_base[var_h]
                                if abs(td) < NEAR_ZERO:
                                    continue
                                md = fmod[var_h] - fbase[var_h]
                                rows.append({
                                    "model": model, "condition": cond,
                                    "world": world, "param": param,
                                    "setting": setting, "seed": seed,
                                    "var_h": var_h, "td": td, "md": md,
                                    "majority_frac": stability[cell],
                                    "cluster_id": f"{world}__{param}__{setting}",
                                    "correct": int(
                                        (td > 0 and md > 0) or (td < 0 and md < 0)
                                    ),
                                })
    return rows


def _clustered_ci(rows, n_boot=10000, seed=42):
    if not rows:
        return {"acc": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    corr = np.array([r["correct"] for r in rows], dtype=np.int8)
    clust = defaultdict(list)
    for i, r in enumerate(rows):
        clust[r["cluster_id"]].append(i)
    ids = list(clust)
    idxs = [np.array(clust[c]) for c in ids]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        s = rng.integers(0, len(ids), size=len(ids))
        idx = np.concatenate([idxs[j] for j in s])
        boot[k] = corr[idx].mean()
    return {
        "acc": float(corr.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n": len(rows),
    }


def _paired_diff(rows_a, rows_b, n_boot=10000, seed=42):
    if not rows_a or not rows_b:
        return {"acc_a": float("nan"), "acc_b": float("nan"),
                "diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_a": 0, "n_b": 0}
    corr_a = np.array([r["correct"] for r in rows_a], dtype=np.int8)
    corr_b = np.array([r["correct"] for r in rows_b], dtype=np.int8)
    clust_a = defaultdict(list)
    for i, r in enumerate(rows_a):
        clust_a[r["cluster_id"]].append(i)
    clust_b = defaultdict(list)
    for i, r in enumerate(rows_b):
        clust_b[r["cluster_id"]].append(i)
    idx_a = {c: np.array(v) for c, v in clust_a.items()}
    idx_b = {c: np.array(v) for c, v in clust_b.items()}
    all_c = sorted(set(idx_a) | set(idx_b))
    obs_a, obs_b = float(corr_a.mean()), float(corr_b.mean())
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        s = rng.integers(0, len(all_c), size=len(all_c))
        pa, pb = [], []
        for j in s:
            c = all_c[j]
            if c in idx_a:
                pa.append(idx_a[c])
            if c in idx_b:
                pb.append(idx_b[c])
        if pa and pb:
            diffs[k] = corr_a[np.concatenate(pa)].mean() - corr_b[np.concatenate(pb)].mean()
        else:
            diffs[k] = np.nan
    diffs = diffs[~np.isnan(diffs)]
    return {
        "acc_a": obs_a, "acc_b": obs_b, "diff": obs_a - obs_b,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_a": len(rows_a), "n_b": len(rows_b),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel-run",
                       default=str(V2_ROOT / "outputs/relabel_run_v4"))
    parser.add_argument("--per-seed-gt",
                       default="/tmp/lmm2_per_seed_test/ground_truth_per_seed.json")
    parser.add_argument("--reference-json",
                       default=str(V2_ROOT / "outputs/analysis/relabel_v2_final.json"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--tolerance-pp", type=float, default=0.5)
    args = parser.parse_args()

    # Load per-seed GT.
    gt_ps = json.loads(Path(args.per_seed_gt).read_text())

    # Load infer/told from the main checkpoints directly (no dependency
    # on run_analysis.load_forecasts).
    infer_records = []
    for cp_rel in [
        "outputs/structural_tracking_v2/checkpoint.jsonl",
        "outputs/opus_run/checkpoint.jsonl",
    ]:
        cp = V2_ROOT / cp_rel
        if not cp.exists():
            continue
        # Dedup by call_id (last wins).
        last = {}
        with open(cp) as f:
            for line in f:
                r = json.loads(line)
                last[r["call_id"]] = r
        infer_records.extend(last.values())
    infer_records = [r for r in infer_records
                     if r.get("model") in V4_MODELS
                     and r.get("condition") == "infer"]

    # Load v4 relabel.
    v4 = _load_checkpoint(Path(args.relabel_run) / "checkpoint.jsonl")
    v4 = [r for r in v4 if r.get("model") in V4_MODELS]

    logger.info(f"loaded {len(infer_records)} infer + {len(v4)} v4 relabel")

    fc_infer = _index_forecasts(infer_records, condition_filter={"infer"})
    fc_relabel = _index_forecasts(v4, condition_filter={"relabel"},
                                   unrelabel=True)

    stability = _compute_stability(gt_ps)
    rows = _build_rows(fc_infer, fc_relabel, gt_ps, stability)
    logger.info(f"assembled {len(rows)} rows")

    def _tier_filter(rs, tier):
        if tier == "all":
            return rs
        thresh = STRICT_THRESHOLD if tier == "strict" else MODERATE_THRESHOLD
        return [r for r in rs if r["majority_frac"] >= thresh]

    tiers = {"all": rows,
             "moderate": _tier_filter(rows, "moderate"),
             "strict": _tier_filter(rows, "strict")}

    # Compute independent numbers.
    independent = {tier: {} for tier in tiers}
    for tier, trs in tiers.items():
        for model in V4_MODELS:
            r_r = [r for r in trs if r["model"] == model
                   and r["condition"] == "relabel"]
            r_i = [r for r in trs if r["model"] == model
                   and r["condition"] == "infer"]
            paired = _paired_diff(r_r, r_i, n_boot=args.n_boot, seed=42)
            ci_r = _clustered_ci(r_r, n_boot=args.n_boot, seed=42)
            ci_i = _clustered_ci(r_i, n_boot=args.n_boot, seed=42)
            independent[tier][model] = {
                "relabel_acc": ci_r["acc"], "relabel_n": ci_r["n"],
                "infer_acc": ci_i["acc"], "infer_n": ci_i["n"],
                "diff": paired["diff"], "diff_ci_low": paired["ci_low"],
                "diff_ci_high": paired["ci_high"],
            }

    # Compare to score_relabel_v2 output.
    ref = json.loads(Path(args.reference_json).read_text())
    ref_paired = ref["paired_by_tier"]

    deviations = []
    for tier, per_model in independent.items():
        for model, ours in per_model.items():
            theirs = ref_paired.get(tier, {}).get(model, {})
            if not theirs:
                continue
            for field, label in [
                ("acc_a", "relabel_acc"),
                ("acc_b", "infer_acc"),
                ("diff", "diff"),
            ]:
                a = theirs.get(field)
                b = ours.get(label)
                if a is None or b is None:
                    continue
                pp = abs(a - b) * 100
                if pp > args.tolerance_pp:
                    deviations.append({
                        "tier": tier, "model": model, "field": label,
                        "reference": a, "independent": b, "pp_diff": pp,
                    })

    lines = []
    lines.append("# Independent rescore of v2 relabel run")
    lines.append("")
    lines.append(f"Reference: `{args.reference_json}` "
                 f"(score_relabel_v2.py output).")
    lines.append(f"Tolerance: **{args.tolerance_pp} pp** absolute difference "
                 "on accuracy or paired difference.")
    lines.append("")
    lines.append("## Independent numbers (this script's computation)")
    lines.append("")
    lines.append("| Tier | Model | Relabel acc | n (r) | Infer acc | n (i) "
                 "| Paired diff [95% CI] |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for tier in ["all", "moderate", "strict"]:
        for model in V4_MODELS:
            d = independent[tier].get(model)
            if not d:
                continue
            diff_s = (
                f"{d['diff']*100:+.1f} pp "
                f"[{d['diff_ci_low']*100:+.1f}, {d['diff_ci_high']*100:+.1f}]"
                if d["diff"] == d["diff"] else "n/a"
            )
            lines.append(
                f"| {tier} | {model} "
                f"| {d['relabel_acc']*100:.1f}% | {d['relabel_n']} "
                f"| {d['infer_acc']*100:.1f}% | {d['infer_n']} | {diff_s} |"
            )
    lines.append("")
    lines.append("## Agreement with score_relabel_v2")
    lines.append("")
    if not deviations:
        lines.append(f"**No deviations above {args.tolerance_pp} pp.** "
                     "Independent rescore agrees with score_relabel_v2 "
                     "within tolerance for every model at every tier.")
    else:
        lines.append(f"**FLAGGED: {len(deviations)} field(s) differ by more "
                     f"than {args.tolerance_pp} pp.**")
        lines.append("")
        lines.append("| Tier | Model | Field | Reference | Independent | Δ pp |")
        lines.append("|---|---|---|---:|---:|---:|")
        for d in deviations:
            lines.append(
                f"| {d['tier']} | {d['model']} | {d['field']} "
                f"| {d['reference']*100:.2f}% | {d['independent']*100:.2f}% "
                f"| {d['pp_diff']:.2f} |"
            )
    lines.append("")

    out_path = V2_ROOT / "outputs/analysis/relabel_v2_verify_report.md"
    out_path.write_text("\n".join(lines))
    logger.info(f"Wrote {out_path}")

    print()
    print("=" * 70)
    if not deviations:
        print(f"AGREEMENT PASS — all fields within {args.tolerance_pp} pp")
    else:
        print(f"AGREEMENT FLAGGED — {len(deviations)} field(s) exceed "
              f"{args.tolerance_pp} pp")
    print("=" * 70)


if __name__ == "__main__":
    main()
