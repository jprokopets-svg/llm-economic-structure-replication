# Pre-registration: Structural Tracking Test for LLM Macroeconomic Forecasts

**Registered:** 2026-06-20T19:30:00Z  
**Status:** LOCKED — do not edit after first model API call  
**Commit:** (see git log for the commit hash of this file)  
**Authors:** TODO

---

## 1. Research question

Do frontier LLMs' macroeconomic forecasts respond correctly to controlled
changes in a synthetic economy's structural parameters — and does this
structural tracking persist when the parameter change must be inferred from
data rather than stated explicitly?

**Sub-question (relabeling control):** Does structural tracking survive when
economic variable names are replaced with neutral tokens, indicating
data-driven rather than prior-driven reasoning?

## 2. Design

### 2.1 Synthetic worlds

| World | Model | Forecast variables |
|-------|-------|--------------------|
| W1 | Ball (1999) open economy | y, pi, r, e, u |
| W2 | Closed economy | y, pi, r, u |
| W3 | Emerging market (FX-augmented Taylor) | y, pi, r, e, u |
| W4 | Labor hysteresis (wage Phillips, NAIRU) | y, pi, r, u, w, u_natural |

Each world generates a 60-quarter deterministic history (seed=42). Ground truth
is a 1,000-path Monte Carlo forward simulation from the end of the history,
using a demand shock (magnitude +2.0pp for W1/W2/W4, +3.0pp for W3).

### 2.2 Structural parameters

Four parameters, selected by the criterion: unambiguous, monotonic
true-response direction on at least one variable/horizon combination across
all tested settings (see PARAM_AUDIT.md for full audit).

| # | Parameter | Config key | Worlds | Baseline | Settings |
|---|-----------|-----------|--------|----------|----------|
| P1 | Phillips-curve slope | `phillips_curve.output_slope` (W1–W3) / `price_phillips.output_slope` (W4) | W1, W2, W3, W4 | 0.4 (W1/W2), 0.5 (W3), 0.2 (W4) | 5 values per world spanning 0.1–0.9 |
| P2 | Taylor inflation coefficient | `taylor_rule.inflation_coefficient` | W1, W2, W3, W4 | 1.5 (W1/W2/W4), 2.0 (W3) | {1.1, 1.3, 1.5, 2.0, 2.5} (W1/W2/W4); {1.1, 1.5, 2.0, 2.5, 3.0} (W3) |
| P3 | IS interest sensitivity | `is_curve.interest_sensitivity` | W1, W2, W3, W4 | 0.6 | {0.2, 0.4, 0.6, 0.8, 1.0} |
| P4 | Wage-gap slope | `wage_phillips.unemployment_gap_slope` | W4 only | 0.5 | {0.1, 0.3, 0.5, 0.7, 1.0} |

**Excluded:** Okun coefficient (u is deterministic in y — not independent),
Taylor smoothing ρ (only 2 monotonic combos — fragile), inflation persistence
(small effect), productivity passthrough (zero effect under demand shocks).

### 2.3 Forecast horizons

h = 1, 4, 8 quarters. Requested in a single prompt (not separate calls).

### 2.4 Information conditions

| Condition | Description | Scope |
|-----------|-------------|-------|
| **TOLD** (interventional) | Narrative states the structural parameter and its value. Model performs forward simulation only. | All worlds × all parameters |
| **INFER** (abductive + interventional) | Narrative presents the 60-quarter history without mentioning the parameter change. Model must infer structure from data, then forecast. | All worlds × all parameters |
| **RELABEL** (prior-stripped robustness) | Same as INFER but variable names replaced with neutral tokens. | W1–W2 × P1 (Phillips slope) only |

**Interpretive constraint:** A TOLD result must NOT be cited as evidence of the
INFER capability. The TOLD − INFER gap is a primary reported quantity. (Kıcıman
et al. 2024, "Executable Counterfactuals.")

### 2.5 Seeds

10 seeds per (world, parameter, setting, condition, model) cell. Each seed
generates a different stochastic history (same structural parameters, different
shock draws). Baseline forecasts are shared across parameter settings within a
world/condition/seed.

### 2.6 Models

| Model | API route | Slug / ID | Pinned version | Settings |
|-------|-----------|-----------|----------------|----------|
| GPT-5.5 | OpenRouter | `openai/gpt-5.5` | `openai/gpt-5.5-20260423` | max_tokens=8192, temp=0 |
| Claude Sonnet 4.6 | OpenRouter | `anthropic/claude-sonnet-4.6` | `anthropic/claude-4.6-sonnet-20260217` | max_tokens=8192, temp=0 |
| DeepSeek V4 Pro | OpenRouter | `deepseek/deepseek-v4-pro` | `deepseek/deepseek-v4-pro-20260423` | max_tokens=4096, response_format=json_object, temp=0 |
| Gemini 3.5 Flash | Google direct | `gemini-3.5-flash` | `3.5-flash-05-2026` | max_output_tokens=16384, response_mime_type=application/json, temp=0 |

The prompt text is identical across all four models. Only API-level output
settings differ (response_format for DeepSeek, response_mime_type for Gemini).

### 2.7 Statistical baselines

Naive, AR(1), VAR — fit on the same 60-quarter history at each parameter
setting. VAR lag order reselected by AIC per setting. Baselines serve as
reference for the INFER condition only.

## 3. Predicted true-response directions

The following tables are the pre-stated predictions of what the TRUE model
(Monte Carlo simulation) does when each parameter increases from baseline.
These were computed from the parameter audit (PARAM_AUDIT.md, 500-path MC)
before any LLM was queried.

The structural-tracking test asks: does the LLM's forecast response match these
directions?

**Key:** ↑ = monotonically increasing, ↓ = monotonically decreasing, — = flat
(no effect at h=1 because the parameter change only affects future periods
through the model equations), × = non-monotonic (excluded from sign test for
that cell).

### P1: Phillips-curve slope (higher slope → output gap has stronger effect on inflation)

| Variable | h=1 | h=4 | h=8 | Mechanism |
|----------|-----|-----|-----|-----------|
| **y** | — | ↓ W1/W2/W3/W4 | ↓ W2/W4; × W1/W3 | Tighter policy (from higher pi → higher r) reduces output |
| **pi** | — | **↑ W1/W2/W4**; × W3 | **↓ W1/W2/W4**; × W3 | **SIGN FLIP h4→h8**: more inflation initially (direct slope effect), then policy tightening causes disinflation by h=8 |
| **r** | — | ↑ W1/W2/W4; ↓ W3 | ↓ W1/W2/W4; ↑ W3 | **SIGN FLIP h4→h8**: rates rise to fight inflation, then fall as inflation recedes |
| **e** (W1/W3) | — | ↑ W1; ↓ W3 | ↓ W1; ↑ W3 | Exchange rate follows interest differential |
| **u** | — | ↑ W1/W2/W4 | ↑ W2/W4; × W1/W3 | Higher u from lower output (Okun; derived, not independent) |
| **w** (W4) | — | ↓ W4 (small) | ↓ W4 | Wage inflation falls with tighter labor market |

### P2: Taylor inflation coefficient (higher φ_π → central bank responds more aggressively to inflation)

| Variable | h=1 | h=4 | h=8 | Mechanism |
|----------|-----|-----|-----|-----------|
| **y** | — | ↓ all worlds | ↑ W1; ↓ W2/W4; × W3 | Tighter policy reduces output; **partial sign flip at h=8** in W1 (recovery) |
| **pi** | — | ↓ all worlds | **↓ all worlds** | More aggressive response → inflation falls faster, stays lower |
| **r** | — | ↑ all worlds | **↓ all worlds** | **SIGN FLIP h4→h8**: rates spike (stronger response) then fall (inflation controlled faster) |
| **e** (W1/W3) | — | ↑ W1; ↓ W3 | ↓ W1; ↑ W3 | Follows interest differential |
| **u** | — | ↑ all worlds | ↓ W1; ↑ W2/W4; × W3 | Derived from y |
| **w** (W4) | — | ↓ W4 (small) | ↓ W4 | |

### P3: IS interest sensitivity (higher sensitivity → output responds more strongly to interest rates)

| Variable | h=1 | h=4 | h=8 | Mechanism |
|----------|-----|-----|-----|-----------|
| **y** | — | ↓ W1/W2/W4; × W3 | ↑ W1; × W2/W3/W4 | More transmission → bigger output drop from rate; **partial sign flip at h=8** |
| **pi** | — | ↓ all worlds | ↓ W2/W4; ↑ W3; × W1 | Lower output → less inflation |
| **r** | — | ↓ W1/W2/W3/W4 | ↓ W2/W4; × W1; ↑ W3 | Lower pi → lower rates (Taylor feedback) |
| **e** (W1/W3) | — | ↓ W1/W3 | × W1; ↑ W3 | |
| **u** | — | ↑ W1/W2/W4; × W3 | ↓ W1; × W2/W4; × W3 | Derived from y |
| **w** (W4) | — | ↓ W4 | ↓ W4 | |

### P4: Wage-gap slope (World 4 only; higher slope → wages respond more strongly to unemployment gap)

| Variable | h=1 | h=4 | h=8 | Mechanism |
|----------|-----|-----|-----|-----------|
| **y** | — | ↓ | ↓ | Stronger wage → pi → r channel contracts output |
| **pi** | — | ↑ | ↓ | **SIGN FLIP**: wages push inflation up at h=4; policy response drives it down by h=8 |
| **r** | — | ↑ | × (small) | Rates rise with inflation |
| **u** | — | ↑ | ↑ | From lower output |
| **w** | — | **↑** | **↓** | **SIGN FLIP**: the key channel — wages rise at h=4 (direct slope effect), then fall at h=8 (policy feedback contracts labor demand) |
| **u_natural** | — | — (tiny) | ↑ (small) | Hysteresis: sustained unemployment pushes NAIRU up |

### Summary of sign flips

The parameter audit found pervasive sign flips between h=4 and h=8 for pi, r,
e, and w across most parameters. These are economically correct — they reflect
the multi-period dynamics of the NK model (shock → inflation → policy response
→ disinflation). **The sign-and-slope regression must be run per-horizon, not
pooled**, because pooling across horizons with opposite signs would cancel out
real structural effects and produce a false null.

## 4. Success criteria

**These thresholds are fixed before observing any model results. They will not
be changed post-hoc. If results are mushy, we report them as-is and reframe the
paper as a methodology contribution with inconclusive empirical demonstration.
We do not adjust the bar.**

### 4a. Floor: directional accuracy (sign test)

For each (model, parameter, condition), the model's forecast response is
compared against the predicted true-response direction (from Section 3) at each
(variable, horizon) cell where the true direction is monotonic (marked ↑ or ↓
above; cells marked — or × are excluded).

**Floor criterion:** The model's forecast moves in the correct sign direction on
a **strict majority (>50%)** of the testable (variable, horizon, parameter-setting)
cells, aggregated across all worlds where the parameter is tested.

### 4b. Tracks: sign-and-slope regression (per-horizon)

For each (model, parameter, variable, horizon, condition) combination where the
true direction is monotonic:

```
model_response_i = alpha + beta * true_response_i + epsilon_i
```

where `i` indexes the non-baseline parameter settings (typically 4 settings per
parameter).

**Tracks criterion:** β > 0 **and** statistically significant at p < 0.05.

**Significance test:** Clustered bootstrap (10,000 resamples, seed=42). The
cluster unit is **(world, parameter-setting)** — all horizons, variables, and
seeds from the same parameter-setting within a world share a cluster. This
preserves within-cluster dependence from shared histories and model calls.
Significance is assessed by the percentile method: the 95% CI
[2.5th, 97.5th percentile of bootstrap β distribution] must exclude zero, and
the mean β must be positive.

**The regression is run per-horizon**, not pooled across horizons. The parameter
audit (Section 3) showed that many parameters have opposite true-response signs
at h=4 vs h=8. Pooling would cancel these effects and produce spurious nulls.

### 4c. TOLD − INFER gap (primary reported quantity)

For each model that passes the floor or tracks test in the TOLD condition:
- Compute the same metrics in the INFER condition.
- Report the difference: `metric_TOLD − metric_INFER`.

This is **descriptive**, not a pass/fail criterion. A small gap suggests the
model can infer structure from data; a large gap suggests it relies on explicit
instruction.

### 4d. INFER − RELABEL gap (robustness check)

For models evaluated on the RELABEL subset (W1–W2, P1):
- Report `metric_INFER − metric_RELABEL`.

Descriptive. A small gap suggests data-driven tracking; a large gap suggests
prior-driven tracking.

## 5. Analysis plan

1. Generate histories and MC ground truth for all (world, parameter, setting,
   seed) combinations.
2. Fit statistical baselines (Naive, AR(1), VAR) on each history.
3. Run each model under TOLD and INFER conditions for all parameters; run
   RELABEL on the subset.
4. Parse forecasts. Use two-stage extraction as fallback for DeepSeek if
   primary parse fails. Log all parse failures.
5. Compute directional accuracy (sign test) per (model, parameter, condition).
6. Run per-horizon sign-and-slope regressions with clustered bootstrap SEs.
7. Tabulate TOLD − INFER gaps and INFER − RELABEL gaps.
8. Compare LLM structural sensitivity against baselines.
9. Report **all results**, including parse failures, null results, and
   per-world breakdowns. Do not suppress unfavorable findings.

## 6. What we will NOT do

- We will not cherry-pick parameter settings after seeing results.
- We will not drop models or worlds that produce unfavorable results.
- We will not adjust the sign, slope, or significance thresholds after seeing
  any model output.
- We will not re-run histories with different seeds to improve results.
- We will not modify the predicted true-response directions in Section 3 after
  seeing model outputs.
- We will not report TOLD-condition results as evidence of INFER capability.
- We will not pool the sign-and-slope regression across horizons (the audit
  showed sign flips make pooling invalid).
- We will not switch to a different significance test or cluster unit post-hoc.
- **If results are mushy** (weak or inconsistent tracking, no horizon effect),
  we report them as-is. We reframe the paper as a methodology contribution
  with inconclusive empirical demonstration. We do not lower the bar.

## 7. Amendments

Any amendments after the lock date must be documented below with date and
rationale. The original criteria above remain the primary analysis.

| Date | Amendment | Rationale |
|------|-----------|-----------|
| — | — | — |
