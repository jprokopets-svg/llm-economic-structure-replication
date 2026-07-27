"""Independent verification of fork experiment scoring.

Reimplements scoring logic independently (no imports from score_fork.py).
Compares results against fork_scoring_report.md. Tolerance: 0.5pp.

Usage:
    python scripts/verify_fork.py
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

V2_ROOT = Path(__file__).resolve().parent.parent
# Default: read checked-in data from data/processed/.
FORK_DIR = V2_ROOT / "data" / "processed"
HORIZONS = [1, 4, 8]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    gt = json.loads((FORK_DIR / "fork_ground_truth.json").read_text())
    trends = json.loads((FORK_DIR / "prefork_trends.json").read_text())

    # Load checkpoint independently
    last_record = {}
    with open(FORK_DIR / "checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            last_record[r["call_id"]] = r

    # Index by (arm, world, param, setting, seed, model)
    fc = {}
    for r in last_record.values():
        if not r.get("parse_success") or not r.get("forecast"):
            continue
        key = (r["arm"], r["world"], r.get("param"), r.get("setting"),
               r["seed"], r["model"])
        vals = {}
        for vh, v in r["forecast"].items():
            if isinstance(v, dict) and "point" in v:
                vals[vh] = v["point"]
        fc[key] = vals

    models = sorted(set(r["model"] for r in last_record.values()
                        if r.get("parse_success")))

    # Score: directional accuracy on eligible non-OOD cells
    model_results = {}
    for model in models:
        correct = 0
        total = 0
        for cell_key, cell_data in gt.items():
            if cell_data.get("is_ood"):
                continue
            world = cell_data["world"]
            param = cell_data["param"]
            setting = cell_data["setting"]
            seed = cell_data["seed"]

            arm3 = fc.get(("arm3_baseline", world, None, None, seed, model), {})
            arm2 = fc.get(("arm2_change", world, param, setting, seed, model), {})

            for vh, gt_info in cell_data["forecasts"].items():
                if not gt_info["eligible"]:
                    continue
                a2 = arm2.get(vh)
                a3 = arm3.get(vh)
                if a2 is None or a3 is None:
                    continue
                md = a2 - a3
                td = gt_info["true_delta"]
                total += 1
                if (td > 0 and md > 0) or (td < 0 and md < 0):
                    correct += 1

        acc = correct / total if total > 0 else float("nan")
        model_results[model] = {"accuracy": acc, "n": total,
                                "correct": correct}
        logger.info(f"{model}: {acc*100:.1f}% ({correct}/{total})")

    # Compare against score_fork report
    report_path = FORK_DIR / "fork_scoring_report.md"
    if report_path.exists():
        logger.info("Comparing against fork_scoring_report.md...")
        report = report_path.read_text()
        all_ok = True
        for model, res in model_results.items():
            # Search for model's accuracy in report
            for line in report.split("\n"):
                if model in line and "%" in line:
                    # Extract first percentage
                    import re
                    pcts = re.findall(r"(\d+\.\d+)%", line)
                    if pcts:
                        reported = float(pcts[0])
                        computed = res["accuracy"] * 100
                        diff = abs(reported - computed)
                        status = "OK" if diff <= 0.5 else "MISMATCH"
                        logger.info(
                            f"  {model}: reported={reported:.1f}% "
                            f"computed={computed:.1f}% diff={diff:.2f}pp "
                            f"[{status}]"
                        )
                        if diff > 0.5:
                            all_ok = False
                        break

        if all_ok:
            logger.info("VERIFICATION PASSED: all numbers within 0.5pp")
        else:
            logger.error("VERIFICATION FAILED: mismatches found")
    else:
        logger.info("No scoring report to compare against.")

    # Write verification report
    out = FORK_DIR / "fork_verification_report.md"
    lines = ["# Fork Experiment Independent Verification", ""]
    lines.append("| Model | Accuracy | n |")
    lines.append("|-------|--------:|--:|")
    for model in models:
        r = model_results[model]
        lines.append(f"| {model} | {r['accuracy']*100:.1f}% | {r['n']} |")
    lines.append("")
    out.write_text("\n".join(lines))
    logger.info(f"Verification report: {out}")


if __name__ == "__main__":
    main()
