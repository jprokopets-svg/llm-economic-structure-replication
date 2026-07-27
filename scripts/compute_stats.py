"""
Compute sign-and-slope statistics and TOLD-vs-INFER gap from experiment results.

Usage:
    python scripts/compute_stats.py outputs/run_2026_06_20_143000/
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lmm2.scoring import run_sign_and_slope_regression, compute_told_infer_gap

logger = logging.getLogger(__name__)


def load_results(run_dir: str) -> list[dict]:
    """Load results.json from a run directory."""
    results_path = Path(run_dir) / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json found in {run_dir}")
    with open(results_path) as f:
        return json.load(f)


def compute_sign_test(results: list[dict]) -> dict:
    """
    Compute directional accuracy (sign test) per (model, parameter, variable).

    For each combination, count how many non-baseline settings had the model
    forecast moving in the correct direction.

    Returns:
        Dict with sign-test results per (model, parameter, variable).
    """
    # Group results by (model, parameter, condition, variable)
    groups = defaultdict(list)

    for record in results:
        if "parse_failure" in record.get("directional_accuracy", {}):
            continue

        model_id = record["model_id"]
        parameter = record["parameter"]
        condition = record["condition"]

        for var_key, var_result in record["directional_accuracy"].items():
            key = (model_id, parameter, condition, var_result["variable"])
            groups[key].append(var_result)

    sign_test_results = {}
    for (model_id, parameter, condition, variable), entries in groups.items():
        n_correct = sum(1 for e in entries if e["correct"])
        n_total = len(entries)
        accuracy = n_correct / n_total if n_total > 0 else 0
        passes_floor = accuracy > 0.5

        sign_test_results[f"{model_id}__{parameter}__{condition}__{variable}"] = {
            "model_id": model_id,
            "parameter": parameter,
            "condition": condition,
            "variable": variable,
            "n_correct": n_correct,
            "n_total": n_total,
            "accuracy": accuracy,
            "passes_floor": passes_floor,
        }

    return sign_test_results


def compute_slope_tests(results: list[dict]) -> dict:
    """
    Run sign-and-slope regressions per (model, parameter, condition, variable).

    Collects (true_delta, model_delta) pairs across settings, then regresses.

    Returns:
        Dict with regression results.
    """
    # Group deltas by (model, parameter, condition, variable, horizon)
    groups = defaultdict(lambda: {"true": [], "model": []})

    for record in results:
        if "parse_failure" in record.get("directional_accuracy", {}):
            continue

        model_id = record["model_id"]
        parameter = record["parameter"]
        condition = record["condition"]

        for var_key, var_result in record["directional_accuracy"].items():
            key = (
                model_id, parameter, condition,
                var_result["variable"], var_result["horizon"],
            )
            groups[key]["true"].append(var_result["true_delta"])
            groups[key]["model"].append(var_result["model_delta"])

    slope_results = {}
    for (model_id, param, cond, variable, horizon), deltas in groups.items():
        regression = run_sign_and_slope_regression(
            deltas["true"], deltas["model"],
        )
        key = f"{model_id}__{param}__{cond}__{variable}_h{horizon}"
        slope_results[key] = {
            "model_id": model_id,
            "parameter": param,
            "condition": cond,
            "variable": variable,
            "horizon": horizon,
            **regression,
        }

    return slope_results


def compute_gaps(sign_tests: dict, slope_tests: dict) -> dict:
    """
    Compute TOLD-minus-INFER gaps for sign tests and slope tests.

    Returns:
        Dict with gap results.
    """
    gaps = {}

    # Sign-test gaps: match told/infer pairs
    told_sign = {
        k: v for k, v in sign_tests.items() if v["condition"] == "told"
    }
    infer_sign = {
        k: v for k, v in sign_tests.items() if v["condition"] == "infer"
    }

    for told_key, told_val in told_sign.items():
        # Find matching infer key
        infer_key = told_key.replace("__told__", "__infer__")
        if infer_key in infer_sign:
            infer_val = infer_sign[infer_key]
            gap_key = told_key.replace("__told__", "__gap__")
            gaps[gap_key] = {
                "model_id": told_val["model_id"],
                "parameter": told_val["parameter"],
                "variable": told_val["variable"],
                "accuracy_told": told_val["accuracy"],
                "accuracy_infer": infer_val["accuracy"],
                "accuracy_gap": told_val["accuracy"] - infer_val["accuracy"],
            }

    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute sign-and-slope stats from experiment results."
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to the run directory (e.g., outputs/run_2026_06_20_143000/).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    results = load_results(args.run_dir)
    logger.info(f"Loaded {len(results)} result records.")

    # Compute stats
    sign_tests = compute_sign_test(results)
    slope_tests = compute_slope_tests(results)
    gaps = compute_gaps(sign_tests, slope_tests)

    # Save stats
    run_dir = Path(args.run_dir)

    stats = {
        "sign_tests": sign_tests,
        "slope_tests": slope_tests,
        "told_infer_gaps": gaps,
    }

    stats_path = run_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\nStats saved to {stats_path}")

    # Print summary table
    print("\n=== SIGN TEST SUMMARY ===")
    print(f"{'Model':<30} {'Parameter':<30} {'Cond':<6} {'Var':<5} {'Acc':>6} {'Floor':>6}")
    print("-" * 85)
    for key, val in sorted(sign_tests.items()):
        floor_label = "PASS" if val["passes_floor"] else "FAIL"
        print(
            f"{val['model_id']:<30} {val['parameter']:<30} "
            f"{val['condition']:<6} {val['variable']:<5} "
            f"{val['accuracy']:>6.2f} {floor_label:>6}"
        )

    print("\n=== TOLD-INFER GAPS ===")
    print(f"{'Model':<30} {'Parameter':<30} {'Var':<5} {'Told':>6} {'Infer':>6} {'Gap':>6}")
    print("-" * 85)
    for key, val in sorted(gaps.items()):
        print(
            f"{val['model_id']:<30} {val['parameter']:<30} "
            f"{val['variable']:<5} {val['accuracy_told']:>6.2f} "
            f"{val['accuracy_infer']:>6.2f} {val['accuracy_gap']:>+6.2f}"
        )


if __name__ == "__main__":
    main()
