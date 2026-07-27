"""
Pre-registered analysis: sign-and-slope structural tracking test.

Implements EXACTLY the analysis plan from PREREGISTRATION.md (commit a30b17b):
  - Floor: % correct sign across testable cells
  - Tracks: per-horizon regression with clustered bootstrap
  - TOLD-INFER gap
  - INFER-RELABEL gap (W1-W2/P1 subset)
  - Horizon story
  - Baseline comparison

NO post-hoc additions. NO pooling across horizons.
"""

import importlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

V2_ROOT = Path(__file__).resolve().parent.parent
V1_ROOT = V2_ROOT

sys.path.insert(0, str(V2_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [1, 4, 8]
MODELS = ['Claude Sonnet 4.6', 'GPT-5.5', 'Gemini 3.5 Flash', 'DeepSeek V4 Pro']

# World 3 simulator becomes dynamically unstable at extreme Phillips slopes.
# These settings produce ground-truth values in the tens of thousands (e.g.,
# exchange rate = 183,421). Scoring against these is meaningless.
# Exclude from ALL analysis, consistently.
EXCLUDED_GT_KEYS = {
    "world3__phillips_slope__0.7",
    "world3__phillips_slope__0.9",
}

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

def load_forecasts():
    """Load parsed forecasts from checkpoint, deduplicated.

    The checkpoint may contain duplicate call_ids from session restarts.
    We keep only the LAST record per call_id (the most recent attempt).
    Parse rate is computed as: successful parses / (successful + failed),
    excluding API errors from the denominator.
    """
    # Deduplicate: last record per call_id wins
    last_record = {}
    # Default: read checked-in data from data/processed/.
    with open(V2_ROOT / "data/processed/checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            last_record[r['call_id']] = r

    forecasts = []
    stats = defaultdict(lambda: {'ok': 0, 'fail': 0, 'error': 0})
    for r in last_record.values():
        model = r.get('model', '?')
        if r.get('error'):
            stats[model]['error'] += 1
        elif r.get('parse_success') and r.get('forecast'):
            stats[model]['ok'] += 1
            forecasts.append(r)
        else:
            stats[model]['fail'] += 1

    logger.info(f"Checkpoint: {len(last_record)} unique call_ids")
    logger.info("Parse rates (OK / (OK + fail), excluding API errors):")
    for model in sorted(stats.keys()):
        s = stats[model]
        denom = s['ok'] + s['fail']
        rate = s['ok'] / denom * 100 if denom > 0 else 0
        logger.info(f"  {model:<25} {s['ok']:>5} ok / {denom:>5} attempted = {rate:.1f}%  ({s['error']} API errors)")

    return forecasts


def load_ground_truth():
    """Compute MC ground truth for each (world, param, setting).
    Returns dict: (world, param_key, setting) -> {var_h: mc_mean}
    """
    from llmmatrix.monte_carlo import run_counterfactual
    from llmmatrix.shocks import demand_shock

    WORLDS = {
        "world1": {"config": str(V1_ROOT / "config/ball_baseline.yaml"),
                    "sim_module": "llmmatrix.simulator", "sim_class": "Sim",
                    "mc_vars": ["y","pi","r","e","u"], "shock_mag": 2.0},
        "world2": {"config": str(V1_ROOT / "config/world2_closed_economy.yaml"),
                    "sim_module": "llmmatrix.world2_simulator", "sim_class": "ClosedSim",
                    "mc_vars": ["y","pi","r","e","u"], "shock_mag": 2.0},
        "world3": {"config": str(V1_ROOT / "config/world3_emerging_market.yaml"),
                    "sim_module": "llmmatrix.world3_simulator", "sim_class": "EmergingSim",
                    "mc_vars": ["y","pi","r","e","u"], "shock_mag": 3.0},
        "world4": {"config": str(V1_ROOT / "config/world4_labor_hysteresis.yaml"),
                    "sim_module": "llmmatrix.world4_simulator", "sim_class": "HysteresisSim",
                    "mc_vars": ["y","pi","r","e","u","u_natural","w"], "shock_mag": 2.0},
    }

    PARAMS = {
        "phillips_slope": {
            "world1": {"key": "phillips_curve.output_slope", "settings": [0.1,0.2,0.4,0.6,0.8]},
            "world2": {"key": "phillips_curve.output_slope", "settings": [0.1,0.2,0.4,0.6,0.8]},
            "world3": {"key": "phillips_curve.output_slope", "settings": [0.1,0.3,0.5,0.7,0.9]},
            "world4": {"key": "price_phillips.output_slope", "settings": [0.05,0.1,0.2,0.3,0.5]},
        },
        "taylor_phi_pi": {
            "world1": {"key": "taylor_rule.inflation_coefficient", "settings": [1.1,1.3,1.5,2.0,2.5]},
            "world2": {"key": "taylor_rule.inflation_coefficient", "settings": [1.1,1.3,1.5,2.0,2.5]},
            "world3": {"key": "taylor_rule.inflation_coefficient", "settings": [1.1,1.5,2.0,2.5,3.0]},
            "world4": {"key": "taylor_rule.inflation_coefficient", "settings": [1.1,1.3,1.5,2.0,2.5]},
        },
        "is_sensitivity": {
            "world1": {"key": "is_curve.interest_sensitivity", "settings": [0.2,0.4,0.6,0.8,1.0]},
            "world2": {"key": "is_curve.interest_sensitivity", "settings": [0.2,0.4,0.6,0.8,1.0]},
            "world3": {"key": "is_curve.interest_sensitivity", "settings": [0.2,0.4,0.6,0.8,1.0]},
            "world4": {"key": "is_curve.interest_sensitivity", "settings": [0.2,0.4,0.6,0.8,1.0]},
        },
        "wage_gap_slope": {
            "world4": {"key": "wage_phillips.unemployment_gap_slope", "settings": [0.1,0.3,0.5,0.7,1.0]},
        },
    }

    # Default: read checked-in data from data/processed/.
    gt_cache_path = V2_ROOT / "data/processed/ground_truth.json"
    if gt_cache_path.exists():
        logger.info("Loading cached ground truth...")
        with open(gt_cache_path) as f:
            return json.load(f)

    logger.info("Computing ground truth (MC simulation)...")
    ground_truth = {}

    for param_name, param_worlds in PARAMS.items():
        for world_name, wcfg in param_worlds.items():
            wc = WORLDS[world_name]
            sim_mod = importlib.import_module(wc["sim_module"])
            sim_class = getattr(sim_mod, wc["sim_class"])
            var_index = {v: i for i, v in enumerate(wc["mc_vars"])}

            for setting in wcfg["settings"]:
                gt_key = f"{world_name}__{param_name}__{setting}"
                if gt_key in ground_truth:
                    continue

                # Build modified config
                with open(wc["config"]) as f:
                    config = yaml.safe_load(f)
                keys = wcfg["key"].split(".")
                target = config
                for k in keys[:-1]:
                    target = target[k]
                target[keys[-1]] = setting

                import tempfile
                tmp = Path(tempfile.gettempdir()) / "lmm2_gt"
                tmp.mkdir(exist_ok=True)
                tmp_path = tmp / f"gt_{world_name}_{param_name}_{setting}.yaml"
                with open(tmp_path, "w") as f:
                    yaml.dump(config, f)

                # Generate history
                sim = sim_class(str(tmp_path), seed=42)
                traj = sim.run(60, sim.get_initial_state())
                history_df = sim.to_dataframe(traj)
                shock_period = int(history_df.iloc[-1]["period"]) + 1

                # MC with shock
                from llmmatrix.shocks import demand_shock
                shock_paths = run_counterfactual(
                    str(tmp_path), history_df,
                    lambda: demand_shock(magnitude=wc["shock_mag"], period=shock_period),
                    1000, max(HORIZONS), 1000, sim_class, wc["mc_vars"])
                # MC without shock (baseline)
                base_paths = run_counterfactual(
                    str(tmp_path), history_df,
                    lambda: {}, 1000, max(HORIZONS), 1000, sim_class, wc["mc_vars"])

                gt = {}
                for var, vi in var_index.items():
                    for h in HORIZONS:
                        # The "forecast" ground truth is the MC mean under no-shock
                        # (what the economy does with this param setting, no new shock)
                        gt[f"{var}_{h}"] = float(np.mean(base_paths[:, h-1, vi]))
                ground_truth[gt_key] = gt

    with open(gt_cache_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    logger.info(f"Ground truth cached: {len(ground_truth)} cells")
    return ground_truth


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def compute_sign_test(forecasts, ground_truth):
    """
    For each (model, param, condition, variable, horizon), check if the
    model's forecast moves in the same direction as the ground truth when
    the parameter changes from baseline.
    """
    BASELINES = {
        "phillips_slope": {"world1": 0.4, "world2": 0.4, "world3": 0.5, "world4": 0.2},
        "taylor_phi_pi": {"world1": 1.5, "world2": 1.5, "world3": 2.0, "world4": 1.5},
        "is_sensitivity": {"world1": 0.6, "world2": 0.6, "world3": 0.6, "world4": 0.6},
        "wage_gap_slope": {"world4": 0.5},
    }

    # Group forecasts by (model, world, param, setting, condition) -> avg forecast
    grouped = defaultdict(lambda: defaultdict(list))
    for f in forecasts:
        key = (f['model'], f['world'], f['param'], f['setting'], f['condition'])
        for var_h, vals in f['forecast'].items():
            if isinstance(vals, dict) and 'point' in vals:
                grouped[key][var_h].append(vals['point'])

    # Average across seeds
    avg_forecasts = {}
    for key, var_dict in grouped.items():
        avg_forecasts[key] = {vh: np.mean(vals) for vh, vals in var_dict.items()}

    results = []  # list of dicts for each testable cell

    for param_name, baselines in BASELINES.items():
        for world_name, baseline_val in baselines.items():
            gt_baseline_key = f"{world_name}__{param_name}__{baseline_val}"
            if gt_baseline_key in EXCLUDED_GT_KEYS:
                continue
            gt_baseline = ground_truth.get(gt_baseline_key)
            if not gt_baseline:
                continue

            # Get all non-baseline settings
            for model in MODELS:
                for condition in ['told', 'infer', 'relabel']:
                    baseline_fkey = (model, world_name, param_name, baseline_val, condition)
                    baseline_forecast = avg_forecasts.get(baseline_fkey)
                    if not baseline_forecast:
                        continue

                    # Get non-baseline settings that this param has
                    non_baseline_settings = [
                        s for s in avg_forecasts
                        if s[0] == model and s[1] == world_name and s[2] == param_name
                        and s[4] == condition and s[3] != baseline_val
                    ]

                    for setting_key in non_baseline_settings:
                        setting_val = setting_key[3]
                        gt_modified_key = f"{world_name}__{param_name}__{setting_val}"
                        if gt_modified_key in EXCLUDED_GT_KEYS:
                            continue
                        modified_forecast = avg_forecasts[setting_key]
                        gt_modified = ground_truth.get(gt_modified_key)
                        if not gt_modified:
                            continue

                        for var_h in baseline_forecast:
                            if var_h not in modified_forecast or var_h not in gt_baseline or var_h not in gt_modified:
                                continue

                            true_delta = gt_modified[var_h] - gt_baseline[var_h]
                            model_delta = modified_forecast[var_h] - baseline_forecast[var_h]

                            if abs(true_delta) < 0.01:  # flat true response
                                continue

                            var = var_h.rsplit('_', 1)[0]
                            horizon = int(var_h.rsplit('_', 1)[1])
                            correct = (true_delta > 0 and model_delta > 0) or (true_delta < 0 and model_delta < 0)

                            results.append({
                                'model': model, 'world': world_name,
                                'param': param_name, 'setting': setting_val,
                                'condition': condition, 'variable': var,
                                'horizon': horizon,
                                'true_delta': true_delta,
                                'model_delta': model_delta,
                                'correct': correct,
                                'cluster_id': f"{world_name}__{param_name}__{setting_val}",
                            })

    return results


def run_regressions(sign_results):
    """
    Per-horizon sign-and-slope regressions with clustered bootstrap.
    Per (model, param, condition, horizon).
    """
    from scipy import stats as scipy_stats

    # Group
    groups = defaultdict(list)
    for r in sign_results:
        key = (r['model'], r['param'], r['condition'], r['horizon'])
        groups[key].append(r)

    regression_results = []

    for key, obs in groups.items():
        model, param, condition, horizon = key
        true_deltas = np.array([o['true_delta'] for o in obs])
        model_deltas = np.array([o['model_delta'] for o in obs])
        cluster_ids = [o['cluster_id'] for o in obs]

        n = len(obs)
        if n < 3 or np.std(true_deltas) < 1e-10:
            regression_results.append({
                'model': model, 'param': param, 'condition': condition,
                'horizon': horizon, 'n': n,
                'beta': float('nan'), 'beta_se_ols': float('nan'),
                'p_ols': float('nan'), 'significant_ols': False,
                'beta_ci_iid': [float('nan'), float('nan')],
                'beta_ci_clust': [float('nan'), float('nan')],
                'significant_iid': False, 'significant_clust': False,
            })
            continue

        # OLS
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(true_deltas, model_deltas)

        # Bootstrap
        rng = np.random.default_rng(42)
        n_boot = 10000

        # i.i.d. bootstrap
        iid_betas = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            st = true_deltas[idx]
            sm = model_deltas[idx]
            if np.std(st) < 1e-10:
                continue
            b, _, _, _, _ = scipy_stats.linregress(st, sm)
            iid_betas.append(b)
        iid_betas = np.array(iid_betas)
        iid_ci = [float(np.percentile(iid_betas, 2.5)), float(np.percentile(iid_betas, 97.5))] if len(iid_betas) > 0 else [float('nan'), float('nan')]

        # Clustered bootstrap
        unique_clusters = list(set(cluster_ids))
        cluster_indices = defaultdict(list)
        for i, cid in enumerate(cluster_ids):
            cluster_indices[cid].append(i)
        cluster_arr = np.array(unique_clusters)
        n_clust = len(unique_clusters)

        clust_betas = []
        for _ in range(n_boot):
            sampled = rng.choice(cluster_arr, size=n_clust, replace=True)
            idx = np.concatenate([cluster_indices[c] for c in sampled])
            st = true_deltas[idx]
            sm = model_deltas[idx]
            if np.std(st) < 1e-10:
                continue
            b, _, _, _, _ = scipy_stats.linregress(st, sm)
            clust_betas.append(b)
        clust_betas = np.array(clust_betas)
        clust_ci = [float(np.percentile(clust_betas, 2.5)), float(np.percentile(clust_betas, 97.5))] if len(clust_betas) > 0 else [float('nan'), float('nan')]

        regression_results.append({
            'model': model, 'param': param, 'condition': condition,
            'horizon': horizon, 'n': n,
            'beta': float(slope), 'beta_se_ols': float(std_err),
            'p_ols': float(p_value),
            'significant_ols': bool(p_value < 0.05 and slope > 0),
            'beta_ci_iid': iid_ci,
            'beta_ci_clust': clust_ci,
            'significant_iid': bool(len(iid_betas) > 0 and iid_ci[0] > 0 and np.mean(iid_betas) > 0),
            'significant_clust': bool(len(clust_betas) > 0 and clust_ci[0] > 0 and np.mean(clust_betas) > 0),
        })

    return regression_results


def compute_baseline_sign_test(ground_truth):
    """Compute sign test for Naive/AR1/VAR baselines."""
    from lmm2.baselines import naive_baseline, ar1_baseline, var_baseline

    BASELINES_MAP = {
        "phillips_slope": {"world1": {"key": "phillips_curve.output_slope", "baseline": 0.4, "settings": [0.1,0.2,0.4,0.6,0.8]},
            "world2": {"key": "phillips_curve.output_slope", "baseline": 0.4, "settings": [0.1,0.2,0.4,0.6,0.8]},
            "world3": {"key": "phillips_curve.output_slope", "baseline": 0.5, "settings": [0.1,0.3,0.5,0.7,0.9]},
            "world4": {"key": "price_phillips.output_slope", "baseline": 0.2, "settings": [0.05,0.1,0.2,0.3,0.5]}},
        "taylor_phi_pi": {"world1": {"key": "taylor_rule.inflation_coefficient", "baseline": 1.5, "settings": [1.1,1.3,1.5,2.0,2.5]},
            "world2": {"key": "taylor_rule.inflation_coefficient", "baseline": 1.5, "settings": [1.1,1.3,1.5,2.0,2.5]},
            "world3": {"key": "taylor_rule.inflation_coefficient", "baseline": 2.0, "settings": [1.1,1.5,2.0,2.5,3.0]},
            "world4": {"key": "taylor_rule.inflation_coefficient", "baseline": 1.5, "settings": [1.1,1.3,1.5,2.0,2.5]}},
        "is_sensitivity": {"world1": {"key": "is_curve.interest_sensitivity", "baseline": 0.6, "settings": [0.2,0.4,0.6,0.8,1.0]},
            "world2": {"key": "is_curve.interest_sensitivity", "baseline": 0.6, "settings": [0.2,0.4,0.6,0.8,1.0]},
            "world3": {"key": "is_curve.interest_sensitivity", "baseline": 0.6, "settings": [0.2,0.4,0.6,0.8,1.0]},
            "world4": {"key": "is_curve.interest_sensitivity", "baseline": 0.6, "settings": [0.2,0.4,0.6,0.8,1.0]}},
        "wage_gap_slope": {"world4": {"key": "wage_phillips.unemployment_gap_slope", "baseline": 0.5, "settings": [0.1,0.3,0.5,0.7,1.0]}},
    }

    WORLDS = {
        "world1": {"config": str(V1_ROOT / "config/ball_baseline.yaml"),
                    "sim_module": "llmmatrix.simulator", "sim_class": "Sim",
                    "variables": ["y","pi","r","e","u"]},
        "world2": {"config": str(V1_ROOT / "config/world2_closed_economy.yaml"),
                    "sim_module": "llmmatrix.world2_simulator", "sim_class": "ClosedSim",
                    "variables": ["y","pi","r","u"]},
        "world3": {"config": str(V1_ROOT / "config/world3_emerging_market.yaml"),
                    "sim_module": "llmmatrix.world3_simulator", "sim_class": "EmergingSim",
                    "variables": ["y","pi","r","e","u"]},
        "world4": {"config": str(V1_ROOT / "config/world4_labor_hysteresis.yaml"),
                    "sim_module": "llmmatrix.world4_simulator", "sim_class": "HysteresisSim",
                    "variables": ["y","pi","r","u","w","u_natural"]},
    }

    results = {}
    for bl_name in ['naive', 'ar1', 'var']:
        correct = 0
        total = 0
        for param_name, param_worlds in BASELINES_MAP.items():
            for world_name, wcfg in param_worlds.items():
                wc = WORLDS[world_name]
                baseline_val = wcfg["baseline"]

                gt_base_key = f"{world_name}__{param_name}__{baseline_val}"
                if gt_base_key in EXCLUDED_GT_KEYS:
                    continue

                for setting in wcfg["settings"]:
                    if setting == baseline_val:
                        continue

                    gt_mod_key = f"{world_name}__{param_name}__{setting}"
                    if gt_mod_key in EXCLUDED_GT_KEYS:
                        continue

                    # Get baseline forecasts
                    gt_base = ground_truth.get(gt_base_key, {})
                    gt_mod = ground_truth.get(gt_mod_key, {})

                    # Generate histories
                    sim_mod = importlib.import_module(wc["sim_module"])
                    sim_class = getattr(sim_mod, wc["sim_class"])

                    for config_setting, gt_key in [(baseline_val, gt_base_key), (setting, gt_mod_key)]:
                        pass  # We already have GT

                    # Get baseline forecast from statistical model
                    # The baseline predicts from its history — same for all settings
                    # (it doesn't know about the parameter)
                    # Build history for each setting
                    for is_baseline in [True, False]:
                        s = baseline_val if is_baseline else setting
                        with open(wc["config"]) as f:
                            config = yaml.safe_load(f)
                        keys_path = wcfg["key"].split(".")
                        t = config
                        for k in keys_path[:-1]: t = t[k]
                        t[keys_path[-1]] = s
                        import tempfile
                        tmp = Path(tempfile.gettempdir()) / "lmm2_bl"
                        tmp.mkdir(exist_ok=True)
                        tp = tmp / f"bl_{world_name}_{param_name}_{s}.yaml"
                        with open(tp, "w") as ff: yaml.dump(config, ff)
                        sim = sim_class(str(tp), seed=42)
                        hist = sim.to_dataframe(sim.run(60, sim.get_initial_state()))

                        if bl_name == 'naive':
                            from lmm2.baselines import naive_baseline
                            bl_forecast = naive_baseline(hist, wc["variables"], max(HORIZONS))
                        elif bl_name == 'ar1':
                            from lmm2.baselines import ar1_baseline
                            bl_forecast = ar1_baseline(hist, wc["variables"], max(HORIZONS))
                        else:
                            from lmm2.baselines import var_baseline
                            try:
                                bl_forecast = var_baseline(hist, wc["variables"], max(HORIZONS))
                            except:
                                bl_forecast = None

                        if is_baseline:
                            bl_base = bl_forecast
                        else:
                            bl_mod = bl_forecast

                    if bl_base is None or bl_mod is None:
                        continue

                    for var_h in bl_base:
                        if var_h not in bl_mod or var_h not in gt_base or var_h not in gt_mod:
                            continue
                        true_delta = gt_mod[var_h] - gt_base[var_h]
                        model_delta = bl_mod[var_h]['point'] - bl_base[var_h]['point']
                        if abs(true_delta) < 0.01:
                            continue
                        c = (true_delta > 0 and model_delta > 0) or (true_delta < 0 and model_delta < 0)
                        if c: correct += 1
                        total += 1

        results[bl_name] = {'correct': correct, 'total': total,
                            'rate': correct/total*100 if total > 0 else 0}
    return results


def main():
    # Default: write to data/processed/ (or override via env).
    output_dir = V2_ROOT / "data/processed"

    # Load data
    logger.info("Loading forecasts...")
    forecasts = load_forecasts()
    logger.info(f"Loaded {len(forecasts)} parsed forecasts")

    logger.info("Loading/computing ground truth...")
    ground_truth = load_ground_truth()
    logger.info(f"Ground truth: {len(ground_truth)} cells")

    # Sign test
    logger.info("Computing sign test...")
    logger.info(f"Excluded GT keys: {sorted(EXCLUDED_GT_KEYS)}")
    sign_results = compute_sign_test(forecasts, ground_truth)
    logger.info(f"Sign test: {len(sign_results)} testable cells")

    # Verify exclusion: max true_delta should be sane (not in thousands)
    max_td = max(abs(r['true_delta']) for r in sign_results) if sign_results else 0
    logger.info(f"Max |true_delta| after exclusion: {max_td:.2f}")
    if max_td > 100:
        logger.warning(f"WARNING: max |true_delta| = {max_td:.2f} — extreme values may remain")

    # Floor: % correct per (model, condition)
    print("\n" + "="*80)
    print("FLOOR: DIRECTIONAL ACCURACY")
    print("="*80)
    floor = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in sign_results:
        key = (r['model'], r['condition'])
        floor[key]['total'] += 1
        if r['correct']:
            floor[key]['correct'] += 1

    print(f"\n{'Model':<25} {'Condition':<10} {'Correct':>8} {'Total':>6} {'Rate':>7} {'Floor':>7}")
    print("-"*70)
    for (model, condition) in sorted(floor.keys()):
        f = floor[(model, condition)]
        rate = f['correct'] / f['total'] * 100 if f['total'] > 0 else 0
        passes = "PASS" if rate > 50 else "FAIL"
        print(f"{model:<25} {condition:<10} {f['correct']:>8} {f['total']:>6} {rate:>6.1f}% {passes:>7}")

    # Floor per horizon
    print(f"\n{'Model':<25} {'Cond':<7} {'h=1':>8} {'h=4':>8} {'h=8':>8}")
    print("-"*60)
    floor_h = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in sign_results:
        key = (r['model'], r['condition'], r['horizon'])
        floor_h[key]['total'] += 1
        if r['correct']:
            floor_h[key]['correct'] += 1

    for model in MODELS:
        for condition in ['told', 'infer']:
            parts = []
            for h in HORIZONS:
                f = floor_h.get((model, condition, h), {'correct':0,'total':0})
                rate = f['correct']/f['total']*100 if f['total']>0 else 0
                parts.append(f"{rate:.0f}%")
            print(f"{model:<25} {condition:<7} {'  '.join(f'{p:>6}' for p in parts)}")

    # Regressions
    logger.info("Running per-horizon regressions with clustered bootstrap (10K)...")
    regressions = run_regressions(sign_results)

    print("\n" + "="*80)
    print("TRACKS: SIGN-AND-SLOPE REGRESSION (per-horizon)")
    print("="*80)
    print(f"\n{'Model':<22} {'Param':<16} {'Cond':<7} {'h':>2} {'n':>5} {'beta':>7} {'CI_iid':>18} {'CI_clust':>18} {'Sig':>5}")
    print("-"*105)

    for reg in sorted(regressions, key=lambda x: (x['model'], x['param'], x['condition'], x['horizon'])):
        ci_iid = f"[{reg['beta_ci_iid'][0]:+.3f},{reg['beta_ci_iid'][1]:+.3f}]" if not np.isnan(reg['beta_ci_iid'][0]) else "N/A"
        ci_clust = f"[{reg['beta_ci_clust'][0]:+.3f},{reg['beta_ci_clust'][1]:+.3f}]" if not np.isnan(reg['beta_ci_clust'][0]) else "N/A"
        sig = "YES" if reg['significant_clust'] else "no"
        beta_str = f"{reg['beta']:+.3f}" if not np.isnan(reg['beta']) else "N/A"
        print(f"{reg['model']:<22} {reg['param']:<16} {reg['condition']:<7} {reg['horizon']:>2} {reg['n']:>5} {beta_str:>7} {ci_iid:>18} {ci_clust:>18} {sig:>5}")

    # TOLD-INFER gap
    print("\n" + "="*80)
    print("TOLD - INFER GAP")
    print("="*80)
    told_floor = {(m, c): f for (m, c), f in floor.items() if c == 'told'}
    infer_floor = {(m, c): f for (m, c), f in floor.items() if c == 'infer'}

    print(f"\n{'Model':<25} {'Told%':>7} {'Infer%':>7} {'Gap':>7}")
    print("-"*50)
    for model in MODELS:
        t = told_floor.get((model, 'told'), {'correct':0,'total':1})
        i = infer_floor.get((model, 'infer'), {'correct':0,'total':1})
        t_rate = t['correct']/t['total']*100 if t['total']>0 else 0
        i_rate = i['correct']/i['total']*100 if i['total']>0 else 0
        gap = t_rate - i_rate
        print(f"{model:<25} {t_rate:>6.1f}% {i_rate:>6.1f}% {gap:>+6.1f}%")

    # TOLD-INFER gap per horizon (beta)
    print(f"\n{'Model':<22} {'Param':<16} {'h':>2} {'beta_T':>7} {'beta_I':>7} {'gap':>7}")
    print("-"*60)
    reg_dict = {(r['model'],r['param'],r['condition'],r['horizon']): r for r in regressions}
    for model in MODELS:
        for param in ['phillips_slope','taylor_phi_pi','is_sensitivity','wage_gap_slope']:
            for h in HORIZONS:
                t = reg_dict.get((model, param, 'told', h))
                i = reg_dict.get((model, param, 'infer', h))
                if not t or not i:
                    continue
                bt = t['beta'] if not np.isnan(t['beta']) else 0
                bi = i['beta'] if not np.isnan(i['beta']) else 0
                print(f"{model:<22} {param:<16} {h:>2} {bt:>+7.3f} {bi:>+7.3f} {bt-bi:>+7.3f}")

    # INFER-RELABEL gap
    print("\n" + "="*80)
    print("INFER - RELABEL GAP (W1-W2, phillips_slope only)")
    print("="*80)
    relabel_results = [r for r in sign_results if r['condition'] == 'relabel']
    infer_sub = [r for r in sign_results if r['condition'] == 'infer'
                 and r['param'] == 'phillips_slope' and r['world'] in ['world1','world2']]

    print(f"\n{'Model':<25} {'Infer%':>8} {'Relabel%':>9} {'Gap':>7} {'n_infer':>8} {'n_relab':>8}")
    print("-"*70)
    for model in MODELS:
        inf = [r for r in infer_sub if r['model'] == model]
        rel = [r for r in relabel_results if r['model'] == model]
        inf_rate = sum(r['correct'] for r in inf)/len(inf)*100 if inf else 0
        rel_rate = sum(r['correct'] for r in rel)/len(rel)*100 if rel else 0
        gap = inf_rate - rel_rate
        print(f"{model:<25} {inf_rate:>7.1f}% {rel_rate:>8.1f}% {gap:>+6.1f}% {len(inf):>8} {len(rel):>8}")

    # Baselines
    logger.info("Computing baseline sign tests...")
    baseline_results = compute_baseline_sign_test(ground_truth)

    print("\n" + "="*80)
    print("BASELINES")
    print("="*80)
    print(f"\n{'Baseline':<15} {'Correct':>8} {'Total':>6} {'Rate':>7} {'Floor':>7}")
    print("-"*50)
    for bl_name, bl in baseline_results.items():
        passes = "PASS" if bl['rate'] > 50 else "FAIL"
        print(f"{bl_name:<15} {bl['correct']:>8} {bl['total']:>6} {bl['rate']:>6.1f}% {passes:>7}")

    # Save all results
    all_results = {
        'floor': {f"{m}_{c}": {'correct': f['correct'], 'total': f['total'],
                                'rate': f['correct']/f['total']*100 if f['total']>0 else 0}
                  for (m,c), f in floor.items()},
        'regressions': regressions,
        'baselines': baseline_results,
    }
    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}/analysis_results.json")


if __name__ == "__main__":
    main()
