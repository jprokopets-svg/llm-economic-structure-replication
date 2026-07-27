"""Generate manuscript figures from final post-b15ca4d data.

Figure 1: Fork history + MC paths + model forecasts (one representative cell)
Figure 2: Fork primary accuracy (ARM2-ARM1) per model with CI bars
Figure 3: Old-grid infer accuracy per model + VAR + baselines

Output: paper/figures/fig1_fork_paths.pdf, fig2_fork_accuracy.pdf, fig3_oldgrid.pdf
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

V2_ROOT = Path(__file__).resolve().parent.parent
FORK_DIR = V2_ROOT / "outputs" / "fork_run"
FIG_DIR = V2_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "GPT-5.5": "#E69F00",        # orange
    "Claude Sonnet 4.6": "#56B4E9",  # sky blue
    "Claude Opus 4.8": "#009E73",    # bluish green
    "Gemini 3.5 Flash": "#CC79A7",   # reddish purple
    "VAR": "#0072B2",             # blue
    "true_base": "#999999",       # grey
    "true_mod": "#000000",        # black
    "chance": "#D55E00",          # vermilion
}

MODEL_ORDER = ["GPT-5.5", "Gemini 3.5 Flash", "Claude Sonnet 4.6", "Claude Opus 4.8"]
MODEL_SHORT = {"GPT-5.5": "GPT-5.5", "Gemini 3.5 Flash": "Gemini 3.5",
               "Claude Sonnet 4.6": "Sonnet 4.6", "Claude Opus 4.8": "Opus 4.8"}


def load_fork_data():
    gt = json.loads((FORK_DIR / "fork_ground_truth.json").read_text())
    last_record = {}
    with open(FORK_DIR / "checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            last_record[r["call_id"]] = r
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
    return gt, fc


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Fork paths
# ═══════════════════════════════════════════════════════════════════════

def figure1():
    """Fork history + true MC paths + model ARM2 forecast paths.

    Cell: world2 / is_sensitivity = 0.2 / seed 3 / variable = pi
    Chosen because: large true effect (-2.4 at h=4, -5.3 at h=8),
    precisely estimated (SE 0.013/0.093), all 4 models have data,
    3/4 wrong at h=4, GPT-5.5 tracks correctly, Opus fails completely.
    World 2 (closed economy) is the simplest model — cleanest visual.
    """
    gt, fc = load_fork_data()
    cell = gt["world2__is_sensitivity__0.2__s3"]
    var = "pi"
    horizons = [1, 4, 8]

    # Load history for the narrative context
    hist_path = FORK_DIR / "histories" / "world2_s3.csv"
    import pandas as pd
    hist = pd.read_csv(hist_path)
    hist_periods = hist[hist["period"] > 0]["period"].values
    hist_values = hist[hist["period"] > 0][var].values

    # True MC means: baseline and modified branches
    base_means = [cell["forecasts"]["%s_%d" % (var, h)]["base_mean"] for h in horizons]
    mod_means = [cell["forecasts"]["%s_%d" % (var, h)]["mod_mean"] for h in horizons]

    # Model ARM2 forecasts (absolute level, not delta)
    model_forecasts = {}
    for model in MODEL_ORDER:
        a2 = fc.get(("arm2_change", "world2", "is_sensitivity", 0.2, 3, model), {})
        vals = []
        for h in horizons:
            vh = "%s_%d" % (var, h)
            vals.append(a2.get(vh))
        model_forecasts[model] = vals

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))

    # History line (quarters 1-60)
    ax.plot(hist_periods, hist_values, color="#666666", linewidth=0.8,
            alpha=0.7, zorder=1)

    # Fork point
    fork_q = 60
    fork_val = hist_values[-1]
    ax.axvline(fork_q, color="#AAAAAA", linestyle="--", linewidth=0.6,
               alpha=0.5, zorder=0)

    # True branches
    fwd_periods = [fork_q + h for h in horizons]
    ax.plot([fork_q] + fwd_periods, [fork_val] + base_means,
            color=COLORS["true_base"], linewidth=2, linestyle="--",
            marker="s", markersize=4, label="True (baseline param)", zorder=3)
    ax.plot([fork_q] + fwd_periods, [fork_val] + mod_means,
            color=COLORS["true_mod"], linewidth=2,
            marker="s", markersize=4, label="True (modified param)", zorder=3)

    # Model forecasts
    for model in MODEL_ORDER:
        vals = model_forecasts[model]
        if any(v is None for v in vals):
            continue
        ax.plot(fwd_periods, vals, color=COLORS[model],
                linewidth=1.5, marker="o", markersize=4,
                label=MODEL_SHORT[model], zorder=2, alpha=0.85)

    ax.set_xlabel("Quarter", fontsize=9)
    ax.set_ylabel("Inflation rate (pp)", fontsize=9)
    ax.set_xlim(40, 72)
    ax.tick_params(labelsize=8)

    # Legend
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9,
              edgecolor="#CCCCCC")

    # Annotation
    ax.annotate("Fork", xy=(fork_q, fork_val), xytext=(fork_q + 1, fork_val + 1.5),
                fontsize=7, color="#888888",
                arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.6))

    fig.tight_layout()
    out = FIG_DIR / "fig1_fork_paths.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Figure 1 saved to %s" % out)
    print("  Cell: world2 / is_sensitivity=0.2 / seed=3 / pi")
    print("  Rationale: large true effect, 3/4 models wrong at h=4, clean visual")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Fork primary accuracy bars
# ═══════════════════════════════════════════════════════════════════════

def figure2():
    """Fork primary accuracy (ARM2-ARM1) per model with 95% CI bars."""
    # Numbers from fork_scoring_dual_report.md Section 2
    data = {
        "GPT-5.5":             {"acc": 49.3, "ci_lo": 46.3, "ci_hi": 52.4},
        "Gemini 3.5 Flash":    {"acc": 41.5, "ci_lo": 39.0, "ci_hi": 44.1},
        "Claude Sonnet 4.6":   {"acc": 37.6, "ci_lo": 35.4, "ci_hi": 39.8},
        "Claude Opus 4.8":     {"acc": 30.6, "ci_lo": 28.6, "ci_hi": 32.7},
    }

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    x_positions = np.arange(len(MODEL_ORDER))
    bar_width = 0.6

    accs = [data[m]["acc"] for m in MODEL_ORDER]
    ci_los = [data[m]["acc"] - data[m]["ci_lo"] for m in MODEL_ORDER]
    ci_his = [data[m]["ci_hi"] - data[m]["acc"] for m in MODEL_ORDER]
    colors = [COLORS[m] for m in MODEL_ORDER]

    bars = ax.bar(x_positions, accs, width=bar_width, color=colors,
                  edgecolor="white", linewidth=0.5, zorder=2)
    ax.errorbar(x_positions, accs, yerr=[ci_los, ci_his],
                fmt="none", ecolor="#333333", capsize=3, capthick=1,
                linewidth=1, zorder=3)

    # Chance line
    ax.axhline(50.0, color=COLORS["chance"], linestyle="--",
               linewidth=1, zorder=1, label="Chance (50%)")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=8)
    ax.set_ylabel("Directional accuracy (%)", fontsize=9)
    ax.set_ylim(20, 60)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper right")

    # Value labels inside bars
    for i, acc in enumerate(accs):
        ax.text(i, acc - 2.5, "%.1f%%" % acc, ha="center", va="top",
                fontsize=7, fontweight="bold", color="white")

    fig.tight_layout()
    out = FIG_DIR / "fig2_fork_accuracy.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Figure 2 saved to %s" % out)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Old-grid infer accuracy + VAR + baselines
# ═══════════════════════════════════════════════════════════════════════

def figure3():
    """Old-grid infer accuracy per model + VAR + simple baselines at all tier."""
    # Numbers from strict_rescore_report.md
    data = [
        ("VAR",        87.3, 84.8, 89.7, 53.0, 51.6, 54.3),
        ("GPT-5.5",    73.1, 69.6, 76.5, 52.1, 50.9, 53.3),
        ("Gemini 3.5", 64.1, 59.8, 68.1, 51.2, 49.9, 52.4),
        ("Opus 4.8",   60.6, 56.5, 64.5, 51.0, 49.9, 52.0),
        ("Sonnet 4.6", 54.9, 52.5, 57.2, 50.4, 49.3, 51.5),
        ("DeepSeek V4", 53.9, 49.4, 58.7, 49.9, 47.5, 52.3),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    x = np.arange(len(data))
    names = [d[0] for d in data]
    accs = [d[1] for d in data]
    ci_los = [d[1] - d[2] for d in data]
    ci_his = [d[3] - d[1] for d in data]
    perm_means = [d[4] for d in data]
    perm_los = [d[4] - d[5] for d in data]
    perm_his = [d[6] - d[4] for d in data]

    bar_colors = [COLORS.get({"VAR": "VAR", "GPT-5.5": "GPT-5.5",
                               "Gemini 3.5": "Gemini 3.5 Flash",
                               "Opus 4.8": "Claude Opus 4.8",
                               "Sonnet 4.6": "Claude Sonnet 4.6",
                               "DeepSeek V4": "chance"}.get(n, "chance"), "#888888")
                  for n in names]

    # Accuracy bars
    bars = ax.bar(x, accs, width=0.6, color=bar_colors,
                  edgecolor="white", linewidth=0.5, zorder=2)
    ax.errorbar(x, accs, yerr=[ci_los, ci_his],
                fmt="none", ecolor="#333333", capsize=3, capthick=1,
                linewidth=1, zorder=3)

    # Permutation band
    for i in range(len(data)):
        ax.fill_between([i - 0.35, i + 0.35],
                        [perm_means[i] - perm_los[i]] * 2,
                        [perm_means[i] + perm_his[i]] * 2,
                        color="#DDDDDD", alpha=0.6, zorder=0)
        ax.plot([i - 0.35, i + 0.35], [perm_means[i]] * 2,
                color="#AAAAAA", linewidth=0.8, zorder=1)

    # Sign-frequency baseline
    ax.axhline(51.7, color=COLORS["chance"], linestyle=":",
               linewidth=0.8, zorder=0)
    ax.text(0.5, 52.5, "Sign-freq baseline",
            fontsize=6, color=COLORS["chance"], ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5, rotation=15, ha="right")
    ax.set_ylabel("Directional accuracy (%)", fontsize=9)
    ax.set_ylim(40, 95)
    ax.tick_params(labelsize=8)

    # Value labels
    for i, acc in enumerate(accs):
        y_pos = acc + ci_his[i] + 1.5
        ax.text(i, y_pos, "%.1f%%" % acc, ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    fig.tight_layout()
    out = FIG_DIR / "fig3_oldgrid.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Figure 3 saved to %s" % out)


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "pdf.fonttype": 42,  # TrueType — embeds fonts
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure1()
    figure2()
    figure3()

    # Delete stale old figure
    old_fig = V2_ROOT / "paper" / "figures" / "fig_main_accuracy.py"
    if old_fig.exists():
        print("Note: stale fig_main_accuracy.py still exists at %s" % old_fig)
        print("  (Not deleting — it's a script, not an output. Superseded by fig3.)")
