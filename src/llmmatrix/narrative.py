"""
Rule-based narrative generation for the LLMMatrix pilot.

Converts a 60-quarter simulated history into ~3-5k tokens of analyst-style
prose describing the economy of the fictional "Republic of Vantria."

The narrative is deterministic and template-based (NOT LLM-generated).
Every quarter's inflation, interest rate, and unemployment must be
recoverable from the text via regex.
"""

import logging
import re
from typing import Optional

import pandas as pd
import tiktoken

logger = logging.getLogger(__name__)

# Country and currency constants
COUNTRY = "Republic of Vantria"
CURRENCY = "vantra"
CURRENCY_CODE = "VTR"


# ---------------------------------------------------------------------------
# Qualitative descriptor helpers
# ---------------------------------------------------------------------------

def _describe_output_gap(y: float) -> str:
    """Convert output gap to qualitative language."""
    if y > 1.5:
        return "the economy was running well above potential"
    elif y > 0.5:
        return "output was above potential"
    elif y > 0.1:
        return "growth was modestly above trend"
    elif y > -0.1:
        return "the economy was operating near potential"
    elif y > -0.5:
        return "modest slack had emerged in the economy"
    elif y > -1.5:
        return "output was below potential"
    else:
        return "significant economic slack persisted"


def _describe_exchange_rate(e: float, delta_e: float) -> str:
    """Convert exchange rate level and change to qualitative language."""
    if delta_e > 1.0:
        return f"the {CURRENCY} appreciated sharply against the trade-weighted basket"
    elif delta_e > 0.3:
        return f"the {CURRENCY} strengthened against trading partners"
    elif delta_e > -0.3:
        return f"the {CURRENCY} held broadly steady on a trade-weighted basis"
    elif delta_e > -1.0:
        return f"the {CURRENCY} weakened against the trade-weighted basket"
    else:
        return f"the {CURRENCY} depreciated sharply against trading partners"


def _describe_inflation_trend(pi_current: float, pi_prev: float) -> str:
    """Describe inflation trend relative to target and prior period."""
    delta = pi_current - pi_prev
    if pi_current > 3.0:
        if delta > 0.3:
            return "inflation continued to accelerate, moving further above target"
        else:
            return "inflation remained elevated above the central bank's target"
    elif pi_current > 2.3:
        if delta > 0.2:
            return "inflationary pressures were building"
        else:
            return "inflation was running modestly above the 2% target"
    elif pi_current > 1.7:
        return "inflation hovered near the central bank's 2% target"
    elif pi_current > 1.0:
        if delta < -0.2:
            return "disinflationary pressures were evident"
        else:
            return "inflation was running below target"
    else:
        return "deflationary risks were emerging"


def _describe_unemployment_trend(u: float, delta_u: float) -> str:
    """Describe unemployment relative to natural rate and trend."""
    if u > 6.0:
        if delta_u > 0.2:
            return "the labor market deteriorated further"
        else:
            return "unemployment remained elevated"
    elif u > 5.3:
        if delta_u > 0.1:
            return "the labor market softened"
        else:
            return "the labor market showed some slack"
    elif u > 4.7:
        return "labor market conditions were broadly balanced"
    elif u > 4.0:
        if delta_u < -0.1:
            return "the labor market continued to tighten"
        else:
            return "the labor market was firm"
    else:
        return "the labor market was exceptionally tight"


def _describe_policy_stance(r: float, pi: float) -> str:
    """Describe monetary policy stance relative to inflation."""
    real_stance = r - pi
    if real_stance > 1.0:
        return "monetary policy was in restrictive territory"
    elif real_stance > 0.0:
        return "the policy stance was mildly restrictive"
    elif real_stance > -0.5:
        return "the policy stance was broadly neutral"
    elif real_stance > -1.5:
        return "monetary policy was accommodative"
    else:
        return "the central bank maintained a highly accommodative stance"


# ---------------------------------------------------------------------------
# Causal observation templates
# ---------------------------------------------------------------------------

# Each returns a sentence linking two variables. Rotated across sections.
_CAUSAL_TEMPLATES = [
    # monetary policy → output
    lambda r_delta, y_delta, lag: (
        f"Following the policy {'tightening' if r_delta > 0 else 'easing'} "
        f"in the prior period, growth {'softened' if y_delta < 0 else 'firmed'} "
        f"over the subsequent quarters."
    ),
    # exchange rate → inflation
    lambda e_delta, pi_delta, _: (
        f"The {CURRENCY}'s {'appreciation' if e_delta > 0 else 'depreciation'} "
        f"contributed to {'disinflationary' if pi_delta < 0 else 'inflationary'} pressure "
        f"through import price channels."
    ),
    # output → unemployment
    lambda y_val, u_delta, _: (
        f"With output {'above' if y_val > 0 else 'below'} potential, "
        f"unemployment {'edged lower' if u_delta < 0 else 'drifted higher'} "
        f"in line with historical patterns."
    ),
    # inflation → policy response
    lambda pi_delta, r_delta, _: (
        f"In response to {'rising' if pi_delta > 0 else 'falling'} inflation, "
        f"the central bank {'raised' if r_delta > 0 else 'lowered'} its policy rate "
        f"in a measured fashion."
    ),
    # combined
    lambda y_val, pi_val, _: (
        f"The combination of {'above-trend' if y_val > 0 else 'below-trend'} growth "
        f"and {'above-target' if pi_val > 2.0 else 'below-target'} inflation presented "
        f"a {'challenging' if (y_val > 0) != (pi_val > 2.0) else 'consistent'} "
        f"backdrop for policymakers."
    ),
]


# ---------------------------------------------------------------------------
# Section-level narrative templates
# ---------------------------------------------------------------------------

def _write_section(
    section_index: int,
    quarters: list[int],
    history_df: pd.DataFrame,
) -> str:
    """
    Write a 200-400 word section covering a group of quarters.

    Args:
        section_index: 0-based index of this section (for template rotation).
        quarters: list of quarter numbers in this section.
        history_df: full history DataFrame.

    Returns:
        Section text as a string.
    """
    first_q = quarters[0]
    last_q = quarters[-1]

    # Determine year range for the header
    year_start = 1 + (first_q - 1) // 4
    year_end = 1 + (last_q - 1) // 4
    if year_start == year_end:
        header = f"### Year {year_start} (Q{first_q}-Q{last_q})"
    else:
        header = f"### Years {year_start}-{year_end} (Q{first_q}-Q{last_q})"

    paragraphs = []

    # Opening paragraph: broad characterization
    mid_q = quarters[len(quarters) // 2]
    mid_row = history_df[history_df["period"] == mid_q].iloc[0]
    opening = _describe_output_gap(mid_row["y"])
    paragraphs.append(
        f"During this period, {opening}. "
        f"{_describe_inflation_trend(mid_row['pi'], history_df[history_df['period'] == max(first_q - 1, 0)].iloc[0]['pi']).capitalize()}. "
        f"{_describe_policy_stance(mid_row['r'], mid_row['pi']).capitalize()}."
    )

    # Quarter-by-quarter numerical anchors (the recoverable data).
    # Include all variables present in the history DataFrame so that
    # every target variable has numeric context for the model.
    has_e = "e" in history_df.columns and history_df["e"].abs().max() > 0.01
    has_w = "w" in history_df.columns
    has_u_natural = "u_natural" in history_df.columns

    anchor_lines = []
    for q in quarters:
        row = history_df[history_df["period"] == q].iloc[0]

        # Output gap
        y_val = row["y"]
        if y_val >= 0:
            y_str = f"output was {y_val:.1f}% above potential"
        else:
            y_str = f"output was {abs(y_val):.1f}% below potential"

        # Core variables (always present)
        pi_str = f"consumer prices rose {row['pi']:.1f}% year-on-year"
        r_str = f"the policy rate stood at {row['r']:.1f}%"
        u_str = f"unemployment was at {row['u']:.1f}%"

        parts = [y_str, pi_str, r_str, u_str]

        # Exchange rate (worlds 1, 3; absent in world 2 where e=0)
        if has_e:
            e_val = row["e"]
            if abs(e_val) < 0.3:
                e_str = f"the real exchange rate index was {e_val:+.1f} (near equilibrium)"
            elif e_val > 0:
                e_str = f"the real exchange rate index was {e_val:+.1f} (appreciated from equilibrium)"
            else:
                e_str = f"the real exchange rate index was {e_val:+.1f} (depreciated from equilibrium)"
            parts.append(e_str)

        # Wage inflation (world 4)
        if has_w:
            w_val = row["w"]
            parts.append(f"wage inflation was {w_val:.1f}%")

        # NAIRU (world 4)
        if has_u_natural:
            u_nat_val = row["u_natural"]
            parts.append(f"the structural unemployment rate (NAIRU) was {u_nat_val:.1f}%")

        # Join: "In Q{q}, part1, part2, ..., and last_part."
        if len(parts) > 1:
            anchor = f"In Q{q}, " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."
        else:
            anchor = f"In Q{q}, " + parts[0] + "."
        anchor_lines.append(anchor)

    paragraphs.append(" ".join(anchor_lines))

    # Exchange rate summary for the section
    first_row = history_df[history_df["period"] == first_q].iloc[0]
    last_row = history_df[history_df["period"] == last_q].iloc[0]
    e_delta = last_row["e"] - first_row["e"]
    e_desc = _describe_exchange_rate(last_row["e"], e_delta)
    paragraphs.append(
        f"On the external front, {e_desc}."
    )

    # Causal observation (rotated by section index)
    template_idx = section_index % len(_CAUSAL_TEMPLATES)
    template = _CAUSAL_TEMPLATES[template_idx]

    # Pick appropriate arguments based on which template
    if template_idx == 0:
        # monetary policy → output
        r_delta = last_row["r"] - first_row["r"]
        y_delta = last_row["y"] - first_row["y"]
        causal = template(r_delta, y_delta, None)
    elif template_idx == 1:
        # exchange rate → inflation
        pi_delta = last_row["pi"] - first_row["pi"]
        causal = template(e_delta, pi_delta, None)
    elif template_idx == 2:
        # output → unemployment
        u_delta = last_row["u"] - first_row["u"]
        causal = template(mid_row["y"], u_delta, None)
    elif template_idx == 3:
        # inflation → policy
        pi_delta = last_row["pi"] - first_row["pi"]
        r_delta = last_row["r"] - first_row["r"]
        causal = template(pi_delta, r_delta, None)
    else:
        # combined
        causal = template(mid_row["y"], mid_row["pi"], None)

    paragraphs.append(causal)

    # Closing: unemployment trend
    u_delta = last_row["u"] - first_row["u"]
    u_desc = _describe_unemployment_trend(last_row["u"], u_delta)
    paragraphs.append(
        f"In the labor market, {u_desc.lower()}."
    )

    section_text = header + "\n\n" + "\n\n".join(paragraphs)
    return section_text


# ---------------------------------------------------------------------------
# Main narrative generation
# ---------------------------------------------------------------------------

def generate_narrative(history_df: pd.DataFrame) -> str:
    """
    Generate the full narrative from a 60-quarter history.

    Groups quarters into ~12-quarter (3-year) sections, writes each
    section using rotated templates, and adds a framing introduction.

    Args:
        history_df: DataFrame with columns [period, y, pi, r, e, u, ...].

    Returns:
        Full narrative text (~3,000-5,000 tokens).
    """
    # Skip period 0 (initial conditions, not a real quarter)
    all_quarters = sorted(history_df[history_df["period"] > 0]["period"].astype(int).tolist())

    # Group into ~12-quarter sections
    section_size = 12
    sections = []
    for i in range(0, len(all_quarters), section_size):
        sections.append(all_quarters[i:i + section_size])

    # Detect which variables are present
    data = history_df[history_df["period"] > 0]
    has_e = "e" in data.columns and data["e"].abs().max() > 0.01
    has_w = "w" in data.columns
    has_u_natural = "u_natural" in data.columns

    # Compute historical ranges for the unit definitions block
    y_min, y_max = data["y"].min(), data["y"].max()
    pi_min, pi_max = data["pi"].min(), data["pi"].max()
    r_min, r_max = data["r"].min(), data["r"].max()
    u_min, u_max = data["u"].min(), data["u"].max()

    # Adapt overview text to economy type
    if has_e:
        economy_desc = (
            f"a small open economy with a floating exchange rate regime. "
            f"The domestic currency, the {CURRENCY} ({CURRENCY_CODE}), "
            f"trades freely on international markets. "
        )
    else:
        economy_desc = "a closed economy. "

    # Build unit definitions (only for variables present in this world)
    unit_defs = []

    unit_defs.append(
        f"- **Output gap (y)**: Deviation of actual output from potential, "
        f"in percentage points. Positive = above potential. "
        f"Historical range: {y_min:.1f} to {y_max:+.1f}."
    )
    unit_defs.append(
        f"- **Inflation rate (pi)**: Annual CPI inflation in percent. "
        f"Historical range: {pi_min:.1f}% to {pi_max:.1f}%. "
        f"The central bank's target is {data['pi'].median():.0f}.0%."
    )
    unit_defs.append(
        f"- **Policy rate (r)**: Real interest rate in percent per annum. "
        f"Historical range: {r_min:.1f}% to {r_max:.1f}%."
    )
    unit_defs.append(
        f"- **Unemployment rate (u)**: Percentage of labor force. "
        f"Historical range: {u_min:.1f}% to {u_max:.1f}%. "
        f"The estimated natural rate is {u_min + (u_max - u_min) * 0.4:.0f}.0%."
    )

    if has_e:
        e_min, e_max = data["e"].min(), data["e"].max()
        unit_defs.append(
            f"- **Real exchange rate (e)**: Index measuring the real effective exchange rate. "
            f"Positive values = real appreciation; negative values = real depreciation. "
            f"The long-run equilibrium value is 0.0. "
            f"Historical range: {e_min:.1f} to {e_max:+.1f}. "
            f"Example: a value of {e_min:.1f} represents the most depreciated "
            f"level observed; a value of {e_max:+.1f} represents the most appreciated."
        )

    if has_w:
        w_min, w_max = data["w"].min(), data["w"].max()
        unit_defs.append(
            f"- **Wage inflation (w)**: Annual wage growth in percent. "
            f"Historical range: {w_min:.1f}% to {w_max:.1f}%. "
            f"In steady state, wage inflation equals the inflation target."
        )

    if has_u_natural:
        un_min, un_max = data["u_natural"].min(), data["u_natural"].max()
        unit_defs.append(
            f"- **Structural unemployment rate / NAIRU (u_natural)**: "
            f"The unemployment rate consistent with stable inflation, in percent. "
            f"Historical range: {un_min:.1f}% to {un_max:.1f}%. "
            f"This rate can shift over time due to labor market structural changes."
        )

    # Introduction
    intro = (
        f"# Economic History of the {COUNTRY}\n\n"
        f"## Overview\n\n"
        f"The following is a quarterly economic review of the {COUNTRY}, "
        f"{economy_desc}"
        f"The central bank targets consumer price inflation at {data['pi'].median():.0f}.0% annually "
        f"and adjusts its policy rate accordingly. "
        f"This report covers {len(all_quarters)} quarters of economic history.\n\n"
        f"## Variable Definitions and Units\n\n"
        + "\n".join(unit_defs) + "\n\n"
        f"## Detailed Quarterly Review\n"
    )

    # Build each section
    section_texts = []
    for idx, quarter_group in enumerate(sections):
        section_text = _write_section(idx, quarter_group, history_df)
        section_texts.append(section_text)

    full_narrative = intro + "\n\n".join(section_texts)
    return full_narrative


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Count tokens using tiktoken.

    Args:
        text: The text to tokenize.
        model: Tokenizer name (cl100k_base works for GPT-4/Claude).

    Returns:
        Token count.
    """
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def sanity_check_narrative(
    narrative: str,
    history_df: pd.DataFrame,
) -> dict:
    """
    Verify that every quarter's output gap, inflation, interest rate,
    unemployment, and exchange rate can be recovered from the narrative via regex.

    Args:
        narrative: The generated narrative text.
        history_df: The source history DataFrame.

    Returns:
        Dict with keys: passed (bool), missing_quarters (list),
        total_quarters (int), recovery_rate (float).
    """
    missing_quarters = []
    quarters_to_check = history_df[history_df["period"] > 0]["period"].astype(int).tolist()

    for q in quarters_to_check:
        row = history_df[history_df["period"] == q].iloc[0]

        # Find the sentence for this quarter: "In Q{q}, ... ."
        # Each quarter's data is in a single sentence. Match from "In Q{q},"
        # up to the next "In Q" or end of paragraph (can't use [^.] because
        # decimal numbers like 1.6% contain periods).
        q_pattern = rf"In Q{q},.+?(?=In Q\d|\n\n|$)"
        q_match = re.search(q_pattern, narrative, re.DOTALL)
        if q_match is None:
            missing_quarters.append({"quarter": q, "missing": ["sentence"]})
            continue

        sentence = q_match.group(0)

        missing = []

        # Check output gap: "X.X% above/below potential"
        y_val = abs(row["y"])
        if f"{y_val:.1f}%" not in sentence:
            missing.append("y")

        # Check inflation: "X.X% year-on-year"
        if f"{row['pi']:.1f}% year-on-year" not in sentence:
            missing.append("pi")

        # Check policy rate: "rate stood at X.X%"
        if f"rate stood at {row['r']:.1f}%" not in sentence:
            missing.append("r")

        # Check unemployment: "unemployment was at X.X%"
        if f"unemployment was at {row['u']:.1f}%" not in sentence:
            missing.append("u")

        # Check exchange rate (only if present and non-trivial)
        has_e = "e" in history_df.columns and history_df["e"].abs().max() > 0.01
        if has_e:
            e_val = row["e"]
            e_str = f"{e_val:+.1f}"
            if e_str not in sentence:
                missing.append("e")

        # Check wage inflation (world 4)
        if "w" in history_df.columns:
            w_str = f"{row['w']:.1f}%"
            if "wage inflation" not in sentence or w_str not in sentence:
                missing.append("w")

        # Check NAIRU (world 4)
        if "u_natural" in history_df.columns:
            un_str = f"{row['u_natural']:.1f}%"
            if "NAIRU" not in sentence or un_str not in sentence:
                missing.append("u_natural")

        if missing:
            missing_quarters.append({"quarter": q, "missing": missing})

    total = len(quarters_to_check)
    recovered = total - len(missing_quarters)
    rate = recovered / total if total > 0 else 0.0

    passed = len(missing_quarters) == 0

    return {
        "passed": passed,
        "missing_quarters": missing_quarters,
        "total_quarters": total,
        "recovered_quarters": recovered,
        "recovery_rate": rate,
    }


def save_narrative(
    history_csv_path: str,
    output_path: str,
) -> dict:
    """
    Generate narrative from history CSV file and save to disk.

    Args:
        history_csv_path: Path to history.csv.
        output_path: Where to save the narrative text.

    Returns:
        Dict with token_count, sanity_check results.
    """
    history_df = pd.read_csv(history_csv_path)

    narrative = generate_narrative(history_df)

    token_count = count_tokens(narrative)
    logger.info(f"Narrative generated: {token_count} tokens, {len(narrative)} chars")

    sanity = sanity_check_narrative(narrative, history_df)
    if sanity["passed"]:
        logger.info(
            f"Sanity check PASSED: all {sanity['total_quarters']} quarters recoverable"
        )
    else:
        logger.warning(
            f"Sanity check FAILED: {len(sanity['missing_quarters'])} quarters "
            f"not fully recoverable. Missing: {sanity['missing_quarters'][:5]}"
        )

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(narrative)

    return {
        "token_count": token_count,
        "sanity_check": sanity,
    }
