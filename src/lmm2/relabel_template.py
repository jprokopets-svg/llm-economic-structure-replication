"""RELABEL condition — hand-written static templates (v2).

Design goals (contrast with the substitution-based v1 approach):
  - No pass-through of the infer narrative text. Everything the model
    sees is generated from constants defined in THIS file plus numeric
    literals lifted verbatim from the history DataFrame.
  - Every non-numeric word in every produced prompt appears in
    ``RELABEL_WHITELIST`` (built from the templates below, saved as
    data). Any other word in the output fails ``verify_whitelist``.
  - Structure and word count match the infer narrative closely: same
    number of sections (5), same numeric anchor per period, same
    "coupling / causal / pool" trailing paragraphs. Target ±15% word
    count vs infer.
  - Variable mapping is fixed and documented; the un-mapping at parse
    time uses ``RELABEL_VARIABLE_INVERSE``.
  - Numeric anchor lines use the SAME formatting values as infer (each
    numeric printed with the exact same precision/sign). Only the label
    around the number changes.

The old substitution map, cleanup pass, and blacklist guard live in
``lmm2.narrative`` and are kept for historical reproducibility of
earlier runs but are NOT used by this module.
"""

from __future__ import annotations

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# Variable mapping (fixed and documented)
# ═══════════════════════════════════════════════════════════════════════

# Every economic variable name maps to an X-token. The un-mapping is used
# at parse time in score_relabel.py to convert X1_1 → y_1 etc.
RELABEL_VARIABLE_MAP: dict[str, str] = {
    "y":          "X1",   # output gap  → aggregate deviation
    "pi":         "X2",   # inflation   → rate of index Y2
    "r":          "X3",   # policy rate → control signal
    "e":          "X4",   # exchange rate → coupled index
    "u":          "X5",   # unemployment → pool fraction
    "w":          "X6",   # wage inflation → secondary rate
    "u_natural":  "X7",   # NAIRU        → pool baseline
}
RELABEL_VARIABLE_INVERSE: dict[str, str] = {
    v: k for k, v in RELABEL_VARIABLE_MAP.items()
}


# ═══════════════════════════════════════════════════════════════════════
# System-prompt-like framing (prepended)
# ═══════════════════════════════════════════════════════════════════════

_SYSTEM_PREAMBLE = (
    "You are forecasting an abstract dynamical system. "
    "The system has multiple observable quantities labelled X1, X2, X3, "
    "and so on, and a controller that adjusts a control signal in "
    "response to observed drift. The trace below reports quantities "
    "period by period.\n\n"
)


# ═══════════════════════════════════════════════════════════════════════
# Overview text (per world type)
# ═══════════════════════════════════════════════════════════════════════

_OVERVIEW_OPEN = (
    "The following log describes 60 successive periods of an open coupled "
    "system with a free-adjusting quantity X4. The controller monitors an "
    "index Y2 and adjusts control signal X3 to hold Y2 near the reference "
    "value {tgt} units per period. This trace covers 60 successive periods."
)

_OVERVIEW_CLOSED = (
    "The following log describes 60 successive periods of a closed "
    "coupled system. The controller monitors an index Y2 and adjusts "
    "control signal X3 to hold Y2 near the reference value {tgt} units "
    "per period. This trace covers 60 successive periods."
)


# ═══════════════════════════════════════════════════════════════════════
# Quantity definition blocks — one per world variable
# ═══════════════════════════════════════════════════════════════════════

_QDEF_X1 = (
    "- **Quantity X1**: signed deviation of aggregate value from baseline, "
    "in units. Positive = above baseline. Observed range: {lo} to {hi}."
)
_QDEF_X2 = (
    "- **Quantity X2**: rate of change of index Y2 in units per period. "
    "Observed range: {lo} to {hi}. Reference value: {tgt} units per "
    "period."
)
_QDEF_X3 = (
    "- **Signal X3**: real value of control signal in units per period. "
    "Observed range: {lo} to {hi}."
)
_QDEF_X4 = (
    "- **Quantity X4**: index of coupled quantity relative to equilibrium. "
    "Positive = above equilibrium; negative = below equilibrium. "
    "Equilibrium value: 0.0. Observed range: {lo} to {hi}. Example: a "
    "value of {lo} is the lowest observed reading; a value of {hi} is "
    "the highest."
)
_QDEF_X5 = (
    "- **Quantity X5**: fraction of pool P, in units. Observed range: "
    "{lo} to {hi}. Baseline pool level: {baseline} units."
)
_QDEF_X6 = (
    "- **Quantity X6**: secondary rate of change, in units per period. "
    "Observed range: {lo} to {hi}. In steady state, X6 equals X2."
)
_QDEF_X7 = (
    "- **Quantity X7**: baseline reference for X5, in units. Observed "
    "range: {lo} to {hi}. X7 can drift over time due to persistence in "
    "the pool."
)


# ═══════════════════════════════════════════════════════════════════════
# Qualitative descriptors (paragraph one of each section)
# ═══════════════════════════════════════════════════════════════════════

def _upfirst(s: str) -> str:
    """Capitalize just the first character. Preserves case of the rest —
    unlike str.capitalize() which downcases the rest and would destroy
    X-tokens like ``X2`` (turning them into ``x2``)."""
    return s[0].upper() + s[1:] if s else s


def _describe_x1(y: float) -> str:
    """Qualitative sentence describing X1 (aggregate deviation)."""
    if y > 1.5:
        return "quantity X1 was running well above baseline"
    if y > 0.5:
        return "quantity X1 was above baseline"
    if y > 0.1:
        return "X1 was slightly above baseline"
    if y > -0.1:
        return "quantity X1 was near baseline"
    if y > -0.5:
        return "quantity X1 was slightly below baseline"
    if y > -1.5:
        return "quantity X1 was below baseline"
    return "quantity X1 was well below baseline"


def _describe_x2(pi_current: float, pi_prev: float) -> str:
    """Qualitative sentence describing X2 (rate of change of Y2)."""
    delta = pi_current - pi_prev
    if pi_current > 3.0:
        if delta > 0.3:
            return "X2 continued to rise, moving further above reference"
        return "X2 remained elevated relative to the controller reference"
    if pi_current > 2.3:
        if delta > 0.2:
            return "X2 pressures were building"
        return "X2 was running slightly above the reference"
    if pi_current > 1.7:
        return "X2 hovered near the controller reference value"
    if pi_current > 1.0:
        if delta < -0.2:
            return "downward drift in X2 was evident"
        return "X2 was running below reference"
    return "downside risks in X2 were emerging"


def _describe_x5(u: float, delta_u: float) -> str:
    """Qualitative sentence describing X5 (pool fraction)."""
    if u > 6.0:
        if delta_u > 0.2:
            return "pool fraction X5 rose further"
        return "X5 remained elevated"
    if u > 5.3:
        if delta_u > 0.1:
            return "X5 rose modestly"
        return "X5 showed some elevation"
    if u > 4.7:
        return "pool fraction X5 was near baseline"
    if u > 4.0:
        if delta_u < -0.1:
            return "X5 continued to compress"
        return "X5 was compressed"
    return "X5 was strongly compressed"


def _describe_x3(r: float, pi: float) -> str:
    """Qualitative sentence describing the controller's signal stance."""
    real_stance = r - pi
    if real_stance > 1.0:
        return "signal X3 was in restrictive mode"
    if real_stance > 0.0:
        return "signal X3 was mildly restrictive"
    if real_stance > -0.5:
        return "signal X3 was broadly neutral"
    if real_stance > -1.5:
        return "signal X3 was supportive"
    return "signal X3 was strongly supportive"


# ═══════════════════════════════════════════════════════════════════════
# Coupling-front sentence (paragraph three of each section)
# ═══════════════════════════════════════════════════════════════════════

def _describe_x4(e_val: float, delta_e: float) -> str:
    """Qualitative sentence describing X4 (coupled index)."""
    if delta_e > 1.0:
        return "X4 rose sharply against the reference basket"
    if delta_e > 0.3:
        return "X4 rose against the reference basket"
    if delta_e > -0.3:
        return "X4 held broadly steady against the reference basket"
    if delta_e > -1.0:
        return "X4 fell against the reference basket"
    return "X4 fell sharply against the reference basket"


# ═══════════════════════════════════════════════════════════════════════
# Causal-observation templates (paragraph four of each section)
# ═══════════════════════════════════════════════════════════════════════

def _causal_signal_to_x1(r_delta: float, y_delta: float) -> str:
    verb = "increase" if r_delta > 0 else "decrease"
    reaction = "softened" if y_delta < 0 else "firmed"
    return (f"Following the signal {verb} in the prior period, X1 "
            f"{reaction} over the subsequent periods.")


def _causal_x4_to_x2(e_delta: float, pi_delta: float) -> str:
    direction = "upward" if e_delta > 0 else "downward"
    pressure = "downward" if pi_delta < 0 else "upward"
    return (f"The {direction} move in X4 contributed to {pressure} "
            f"pressure on X2 through the coupling channel.")


def _causal_x1_to_x5(y_val: float, u_delta: float) -> str:
    side = "above" if y_val > 0 else "below"
    verb = "eased" if u_delta < 0 else "drifted higher"
    return (f"With X1 {side} baseline, quantity X5 {verb} in line with "
            f"observed persistence.")


def _causal_x2_to_signal(pi_delta: float, r_delta: float) -> str:
    trend = "rising" if pi_delta > 0 else "falling"
    action = "raised" if r_delta > 0 else "lowered"
    return (f"In response to {trend} X2, the controller {action} signal "
            f"X3 in a measured fashion.")


def _causal_combined(y_val: float, pi_val: float) -> str:
    y_side = "above" if y_val > 0 else "below"
    pi_side = "above" if pi_val > 2.0 else "below"
    if (y_val > 0) != (pi_val > 2.0):
        judgement = "mixed"
    else:
        judgement = "consistent"
    return (f"The combination of X1 {y_side} baseline and X2 {pi_side} "
            f"reference presented a {judgement} configuration for the "
            f"controller.")


_CAUSAL_TEMPLATES = [
    _causal_signal_to_x1,
    _causal_x4_to_x2,
    _causal_x1_to_x5,
    _causal_x2_to_signal,
    _causal_combined,
]


# ═══════════════════════════════════════════════════════════════════════
# Pool-sector closing sentence (paragraph five of each section)
# ═══════════════════════════════════════════════════════════════════════

def _describe_pool(u: float, delta_u: float) -> str:
    if u > 6.0:
        if delta_u > 0.2:
            return "pool fraction rose further"
        return "pool fraction remained elevated"
    if u > 5.3:
        if delta_u > 0.1:
            return "pool fraction rose modestly"
        return "pool fraction showed some elevation"
    if u > 4.7:
        return "pool fraction was near baseline"
    if u > 4.0:
        if delta_u < -0.1:
            return "pool fraction continued to compress"
        return "pool fraction was compressed"
    return "pool fraction was strongly compressed"


# ═══════════════════════════════════════════════════════════════════════
# NUMERIC-ANCHOR BUILDER (byte-identical numeric formatting to infer)
# ═══════════════════════════════════════════════════════════════════════

def _anchor_for_period(row: pd.Series, has_e: bool, has_w: bool,
                       has_x7: bool, q: int) -> str:
    """One period's line — mirrors the infer per-quarter anchor line.

    Numeric formatting is the same as the infer narrative:
      - y: 1 decimal, unsigned (with above/below wording)
      - pi, r, u, w, u_natural: 1 decimal, one place after point
      - e: signed 1 decimal (e.g. +1.0, -4.0)

    Only the surrounding labels change (all economic vocabulary
    replaced with X-tokens and neutral language).
    """
    y_val = row["y"]
    if y_val >= 0:
        y_str = f"X1 was {y_val:.1f} above baseline"
    else:
        y_str = f"X1 was {abs(y_val):.1f} below baseline"
    pi_str = f"index Y2 rose {row['pi']:.1f} units per period"
    r_str = f"signal X3 stood at {row['r']:.1f}"
    u_str = f"quantity X5 was at {row['u']:.1f}"

    parts = [y_str, pi_str, r_str, u_str]

    if has_e:
        e_val = row["e"]
        if abs(e_val) < 0.3:
            e_str = f"X4 index was {e_val:+.1f} near equilibrium"
        elif e_val > 0:
            e_str = f"X4 index was {e_val:+.1f} above equilibrium"
        else:
            e_str = f"X4 index was {e_val:+.1f} below equilibrium"
        parts.append(e_str)

    if has_w:
        parts.append(f"quantity X6 was {row['w']:.1f} units per period")

    if has_x7:
        parts.append(f"reference X7 was {row['u_natural']:.1f}")

    if len(parts) > 1:
        return f"In P{q}, " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."
    return f"In P{q}, " + parts[0] + "."


# ═══════════════════════════════════════════════════════════════════════
# SECTION BUILDER
# ═══════════════════════════════════════════════════════════════════════

def _write_section(section_index: int, quarters: list[int],
                    history_df: pd.DataFrame, has_e: bool,
                    has_w: bool, has_x7: bool) -> str:
    first_q, last_q = quarters[0], quarters[-1]
    year_start = 1 + (first_q - 1) // 4
    year_end = 1 + (last_q - 1) // 4
    if year_start == year_end:
        header = f"### Block {year_start} (P{first_q}-P{last_q})"
    else:
        header = f"### Blocks {year_start}-{year_end} (P{first_q}-P{last_q})"

    paragraphs = [header]

    # Opening: qualitative descriptors
    mid_q = quarters[len(quarters) // 2]
    mid_row = history_df[history_df["period"] == mid_q].iloc[0]
    prev_q = max(first_q - 1, 0)
    prev_pi = history_df[history_df["period"] == prev_q].iloc[0]["pi"]
    opening_x1 = _describe_x1(mid_row["y"])
    opening_x2 = _describe_x2(mid_row["pi"], prev_pi)
    opening_x3 = _describe_x3(mid_row["r"], mid_row["pi"])
    paragraphs.append(
        f"During this block, {opening_x1}. "
        f"{_upfirst(opening_x2)}. "
        f"{_upfirst(opening_x3)}."
    )

    # Numeric anchors
    anchor_lines = []
    for q in quarters:
        row = history_df[history_df["period"] == q].iloc[0]
        anchor_lines.append(_anchor_for_period(row, has_e, has_w, has_x7, q))
    paragraphs.append(" ".join(anchor_lines))

    # Coupling-front sentence
    first_row = history_df[history_df["period"] == first_q].iloc[0]
    last_row = history_df[history_df["period"] == last_q].iloc[0]
    e_delta = last_row["e"] - first_row["e"]
    coupling = _describe_x4(last_row["e"], e_delta)
    paragraphs.append(f"On the coupling channel, {coupling}.")

    # Causal observation
    template_idx = section_index % len(_CAUSAL_TEMPLATES)
    if template_idx == 0:
        r_delta = last_row["r"] - first_row["r"]
        y_delta = last_row["y"] - first_row["y"]
        causal = _CAUSAL_TEMPLATES[0](r_delta, y_delta)
    elif template_idx == 1:
        pi_delta = last_row["pi"] - first_row["pi"]
        causal = _CAUSAL_TEMPLATES[1](e_delta, pi_delta)
    elif template_idx == 2:
        u_delta = last_row["u"] - first_row["u"]
        causal = _CAUSAL_TEMPLATES[2](mid_row["y"], u_delta)
    elif template_idx == 3:
        pi_delta = last_row["pi"] - first_row["pi"]
        r_delta = last_row["r"] - first_row["r"]
        causal = _CAUSAL_TEMPLATES[3](pi_delta, r_delta)
    else:
        causal = _CAUSAL_TEMPLATES[4](mid_row["y"], mid_row["pi"])
    paragraphs.append(causal)

    # Pool-sector closing
    u_delta = last_row["u"] - first_row["u"]
    pool = _describe_pool(last_row["u"], u_delta)
    paragraphs.append(f"In the pool sector, {pool}.")

    return "\n\n".join(paragraphs)


# ═══════════════════════════════════════════════════════════════════════
# TOP-LEVEL PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_relabel_prompt(
    history_df: pd.DataFrame,
    world_id: str,
    forecast_variables: list[str],
    horizons: list[int],
) -> str:
    """Build the full RELABEL prompt for one (history, world) combination.

    Args:
        history_df: 60-quarter history DataFrame.
        world_id: unused directly; retained for interface parity.
        forecast_variables: list of original variable names to forecast
            (y, pi, ...). Un-mapped for the schema section.
        horizons: list of horizon integers (e.g. [1, 4, 8]).

    Returns:
        The full prompt text — system preamble + narrative + schema
        request. Every word is in RELABEL_WHITELIST.
    """
    data = history_df[history_df["period"] > 0]
    has_e = "e" in data.columns and data["e"].abs().max() > 0.01
    has_w = "w" in data.columns
    has_x7 = "u_natural" in data.columns

    y_min, y_max = data["y"].min(), data["y"].max()
    pi_min, pi_max = data["pi"].min(), data["pi"].max()
    r_min, r_max = data["r"].min(), data["r"].max()
    u_min, u_max = data["u"].min(), data["u"].max()

    # Overview (target Y2 value comes from median pi as in infer).
    tgt = f"{data['pi'].median():.0f}.0"
    overview = (_OVERVIEW_OPEN if has_e else _OVERVIEW_CLOSED).format(tgt=tgt)

    # Definitions block.
    qdefs: list[str] = []
    qdefs.append(_QDEF_X1.format(lo=f"{y_min:.1f}", hi=f"{y_max:+.1f}"))
    qdefs.append(_QDEF_X2.format(lo=f"{pi_min:.1f}", hi=f"{pi_max:.1f}", tgt=tgt))
    qdefs.append(_QDEF_X3.format(lo=f"{r_min:.1f}", hi=f"{r_max:.1f}"))
    qdefs.append(_QDEF_X5.format(
        lo=f"{u_min:.1f}", hi=f"{u_max:.1f}",
        baseline=f"{u_min + (u_max - u_min) * 0.4:.0f}.0",
    ))
    if has_e:
        e_min, e_max = data["e"].min(), data["e"].max()
        qdefs.append(_QDEF_X4.format(lo=f"{e_min:.1f}", hi=f"{e_max:+.1f}"))
    if has_w:
        w_min, w_max = data["w"].min(), data["w"].max()
        qdefs.append(_QDEF_X6.format(lo=f"{w_min:.1f}", hi=f"{w_max:.1f}"))
    if has_x7:
        un_min, un_max = data["u_natural"].min(), data["u_natural"].max()
        qdefs.append(_QDEF_X7.format(lo=f"{un_min:.1f}", hi=f"{un_max:.1f}"))

    intro = (
        "# System Trace of Reference Process P\n\n"
        "## Overview\n\n"
        + overview + "\n\n"
        "## Quantity Definitions and Units\n\n"
        + "\n".join(qdefs) + "\n\n"
        "## Detailed Periodic Review\n"
    )

    all_quarters = sorted(data["period"].astype(int).tolist())
    section_size = 12
    sections = [all_quarters[i:i + section_size]
                for i in range(0, len(all_quarters), section_size)]

    section_texts = []
    for i, quarter_group in enumerate(sections):
        section_texts.append(_write_section(
            i, quarter_group, history_df, has_e, has_w, has_x7,
        ))

    narrative = intro + "\n\n".join(section_texts)

    # Schema-request block.
    horizon_labels = ", ".join(f"Q+{h}" for h in horizons)
    relabeled_vars = [RELABEL_VARIABLE_MAP[v] for v in forecast_variables]
    variables_str = ", ".join(relabeled_vars)
    schema = (
        f"\n\nBased on the system trace above, provide your forecast for "
        f"the following quantities at horizons {horizon_labels}:\n\n"
        f"Quantities: {variables_str}\n\n"
        f"For each quantity and horizon, provide:\n"
        f"- point: your point estimate\n"
        f"- ci_low: lower bound of your 80% confidence interval\n"
        f"- ci_high: upper bound of your 80% confidence interval\n\n"
        f"Respond with a JSON object. Keys in the format "
        f"quantity_horizon, for example X1_1 for X1 at Q+1. "
        f"Each value should be an object with point, ci_low, ci_high.\n\n"
        f"Return only the JSON object, no other text."
    )

    return _SYSTEM_PREAMBLE + narrative + schema


# ═══════════════════════════════════════════════════════════════════════
# WHITELIST (all distinct alphabetic tokens from the templates above)
# ═══════════════════════════════════════════════════════════════════════
#
# This list was extracted from every _string constant and every f-string
# literal in this file, plus every distinct word returned by the qualitative
# helpers. It is committed as data — the verifier accepts a token only if
# it is (a) here, (b) an X-token (X1..X7), (c) a Y-index token (Y2), (d) a
# period token (P1..P60), (e) a numeric literal, or (f) pure punctuation.
#
# Hand-review this list when you touch any template constant. If you add
# a new word to a template, add it here — otherwise ``verify_whitelist``
# will fail loudly.
#
# Alphabetized for review. All entries are lowercased; the verifier
# case-folds tokens before checking.

RELABEL_WHITELIST: frozenset[str] = frozenset({
    # Extracted from every template constant and every branch of the
    # qualitative helpers by iterating build_relabel_prompt across all 4
    # worlds and 3 diverse seeds. Reviewed by hand.
    "a", "above", "abstract", "adjusts", "against", "aggregate", "an",
    "and", "are", "at",
    "based", "baseline", "basket", "be", "below", "block", "blocks",
    "bound", "broadly", "building", "by",
    "can", "change", "channel", "closed", "combination", "compress",
    "compressed", "confidence", "configuration", "consistent",
    "continued", "contributed", "control", "controller", "coupled",
    "coupling", "covers",
    "definitions", "describes", "detailed", "deviation", "downside",
    "downward", "drift", "drifted", "due", "during", "dynamical",
    "each", "eased", "elevated", "elevation", "emerging", "equals",
    "equilibrium", "estimate", "evident", "example",
    "falling", "fashion", "fell", "firmed", "following", "for",
    "forecast", "forecasting", "format", "fraction", "free-adjusting",
    "from", "further",
    "has", "held", "higher", "highest", "hold", "horizon", "horizons",
    "hovered",
    "in", "index", "interval", "is",
    "json",
    "keys",
    "labelled", "level", "line", "log", "lower", "lowered", "lowest",
    "measured", "mildly", "mixed", "mode", "modestly", "monitors",
    "move", "moving", "multiple",
    "near", "negative", "neutral", "no",
    "object", "observable", "observed", "of", "on", "only", "open",
    "other", "over", "overview",
    "p", "per", "period", "periodic", "periods", "persistence",
    "point", "pool", "positive", "presented", "pressure", "pressures",
    "prior", "process", "provide",
    "quantities", "quantity",
    "raised", "range", "rate", "reading", "real", "reference",
    "relative", "remained", "reports", "respond", "response",
    "restrictive", "return", "review", "rise", "rising", "risks",
    "rose", "running",
    "secondary", "sector", "sharply", "should", "showed", "signal",
    "signed", "slightly", "so", "softened", "some", "state", "steady",
    "stood", "strongly", "subsequent", "successive", "supportive",
    "system",
    "text", "that", "the", "this", "through", "time", "to", "trace",
    "increase", "decrease",  # signal deltas — neutral replacements
    # for the old "tightening"/"loosening" language.
    "units", "upper", "upward",
    "value",
    "was", "well", "were", "with",
    "you", "your",
})


# Regex helpers used by the verifier.
import re as _re

# X-token: X1, X2, ..., X7, and their horizon-suffixed forms X1_1, X2_4.
_X_TOKEN = _re.compile(r"^X\d+(_\d+)?$")
# Y-token: Y2 (index Y).
_Y_TOKEN = _re.compile(r"^Y\d+$")
# P-token: single period label (P1..P60) OR range (P1-P12).
_P_TOKEN = _re.compile(r"^P\d+(-P\d+)?$")
# Q-token: horizon label like Q+1 (with optional trailing punctuation
# already stripped by the tokenizer).
_Q_TOKEN = _re.compile(r"^Q\+\d+$")
# Numeric literals: optional sign, digits, optional decimal, optional
# trailing % or units letter.
_NUM_TOKEN = _re.compile(r"^[+-]?\d+(\.\d+)?%?$")
# Range like 1-3 or 10-12 used in block headers.
_RANGE_TOKEN = _re.compile(r"^\d+-\d+$")
# JSON field / structural tokens allowed literally.
_JSON_TOKENS = frozenset({
    "ci_low", "ci_high", "quantity_horizon",
})
# Punctuation that we strip from BOTH ends of each token before checking.
# ``.`` is included so "baseline." becomes "baseline". Decimal points
# within numbers (e.g. 0.4) are not at token boundaries so they survive.
# ``+`` and ``-`` are only stripped from the trailing end (not leading)
# so signed numbers like -0.4 keep their sign.
_STRIP_LEAD = _re.compile(r"^[^\w+\-]+")
_STRIP_TAIL = _re.compile(r"[^\w]+$")


def _split_tokens(text: str) -> list[str]:
    """Split into whitespace-separated tokens, stripping trailing
    punctuation from each. Empty results (pure punctuation) are dropped.
    """
    out = []
    for tok in text.split():
        stripped = _STRIP_LEAD.sub("", tok)
        stripped = _STRIP_TAIL.sub("", stripped)
        if not stripped:
            continue
        out.append(stripped)
    return out


def verify_whitelist(prompt: str, whitelist: frozenset[str] | None = None,
                     prompt_id: str | None = None) -> None:
    """Fail loudly if any token in ``prompt`` is not in the whitelist.

    A token is accepted if:
      (a) it is in ``RELABEL_WHITELIST`` (case-insensitive);
      (b) it matches an X-token (X1, X7, X1_4, ...);
      (c) it matches a Y-token (Y2);
      (d) it matches a P-token (P1 .. P60 or a range P1-P12);
      (e) it matches a Q-token (Q+1, Q+4, Q+8);
      (f) it is a numeric literal (optional sign, optional decimal,
          optional trailing %);
      (g) it is a numeric range like ``1-3``;
      (h) it is a bare JSON field name (``ci_low``, ``ci_high``,
          ``quantity_horizon``).
    """
    wl = whitelist if whitelist is not None else RELABEL_WHITELIST
    unknown = []
    for tok in _split_tokens(prompt):
        if tok.lower() in wl:
            continue
        if tok in _JSON_TOKENS:
            continue
        if _X_TOKEN.match(tok):
            continue
        if _Y_TOKEN.match(tok):
            continue
        if _P_TOKEN.match(tok):
            continue
        if _Q_TOKEN.match(tok):
            continue
        if _NUM_TOKEN.match(tok):
            continue
        if _RANGE_TOKEN.match(tok):
            continue
        unknown.append(tok)
    if unknown:
        seen = set()
        uniq = []
        for u in unknown:
            key = u.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(u)
        where = f" (prompt_id={prompt_id})" if prompt_id else ""
        raise ValueError(
            f"RELABEL prompt contains {len(unknown)} out-of-whitelist "
            f"tokens ({len(uniq)} distinct){where}. First 15 distinct: "
            + ", ".join(repr(u) for u in uniq[:15])
        )
