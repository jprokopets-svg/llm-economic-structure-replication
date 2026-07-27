# Fork Experiment: Design Lock + Cost Estimate

**Date:** 2026-07-14
**Status:** PHASE 0 — design only, no API calls until Jake confirms balances.

---

## 1. Research question

When a model is given an identical economic history and asked to forecast under a structural-parameter change — with the governing equation explicitly supplied — does its forecast respond in the correct direction and at the correct magnitude? And does a null (placebo) announcement produce zero response?

---

## 2. Fork design

### 2.1 Setup

For each (world, parameter, non-baseline setting, seed ∈ {0,...,9}):

1. **Generate one baseline history** (60 quarters, seed-specific stochastic shocks, all structural parameters at their baseline values). This history is **identical across all three arms** — the model sees the same data verbatim.

2. Three prompt arms share that history:

| Arm | Prompt addition after history | Purpose |
|-----|-------------------------------|---------|
| **ARM 1 (placebo)** | "At quarter 60, the Central Bank of Vantria reviewed [parameter name in plain language] and left it unchanged at [baseline value]." Forecast h=1,4,8. | Null intervention. Movement here invalidates ARM 2 interpretation. |
| **ARM 2 (change + equation)** | "At quarter 60, the Central Bank of Vantria announced that [parameter name] will change from [baseline] to [modified value] effective quarter 61. The parameter enters the economy via: [structural equation in words + formula]. [One sentence on the parameter's economic role.]" Forecast h=1,4,8. | The interventional test. Model gets the mapping, not just a name and number. |
| **ARM 3 (baseline)** | No announcement clause at all. Plain forecast request from the history. | Reference forecast. |

### 2.2 ARM 2 equation text (per parameter)

Each ARM 2 prompt includes the structural equation and a one-sentence role description:

**Phillips-curve slope (b₂):**
> This parameter enters the Phillips curve: π_t = π* + ρ_π(π_{t-1} − π*) + **b₂** · y_{t-1} [+ FX passthrough terms if applicable] + ε. A higher slope means the output gap has a stronger direct effect on inflation.

**Taylor inflation coefficient (φ_π):**
> This parameter enters the Taylor rule: r_t = ρ · r_{t-1} + (1−ρ)[r* + **φ_π**(π_t − π*) + φ_y · y_t]. A higher coefficient means the central bank raises rates more aggressively in response to inflation above target.

**IS interest sensitivity (β_r):**
> This parameter enters the IS curve: y_t = φ_y · y_{t-1} − **β_r**(r_{t-1} − r*) [+ exchange rate terms if applicable] + ε. A higher sensitivity means output responds more strongly to interest rate deviations from neutral.

**Wage-gap slope (λ, World 4 only):**
> This parameter enters the wage Phillips curve: w_t = w* + ρ_w(w_{t-1} − w*) − **λ**(u_{t-1} − u^nat_{t-1}) + η · a_{t-1} + ε. A higher slope means wages respond more strongly to the unemployment gap.

### 2.3 Call-sharing structure

| Arm | Varies by | Unique calls per model |
|-----|-----------|----------------------|
| ARM 3 (baseline) | (world, seed) only | 4 worlds × 10 seeds = **40** |
| ARM 1 (placebo) | (world, param, seed) | 13 (world,param) × 10 seeds = **130** |
| ARM 2 (change) | (world, param, setting, seed) | 50 settings × 10 seeds = **500** |
| **Total per model** | | **670** |

ARM 3 is shared across all parameters (same history, no parameter mentioned). ARM 1 is shared across settings of the same parameter (same "left unchanged" text regardless of which modified value exists).

---

## 3. Ground truth (common random numbers)

For each (world, parameter, setting, seed):
1. Generate baseline history (60 quarters, seed-specific).
2. From the **same terminal state**, fork the simulator:
   - **Baseline branch:** roll forward 8 quarters under baseline parameter, using MC seeds 1000–1999 (1,000 paths).
   - **Modified branch:** roll forward 8 quarters under modified parameter, using the **same** MC seeds 1000–1999 (common random numbers).
3. `true_delta[var_h] = E[Y | do(θ_modified)] − E[Y | do(θ_baseline)]` per variable per horizon.
4. Store MC standard error: `MC_SE[var_h] = sqrt(Var(delta_path) / n_paths)` where `delta_path` = path-level difference (CRN makes this variance much smaller than independent simulation).

**Cell eligibility criterion:** `|true_delta| > 2 × MC_SE`. This replaces the old near-zero threshold (0.01) and the stability-tier framing. Cells where the true effect is not distinguishable from MC noise at the 2-SE level are excluded from the primary analysis.

---

## 4. Scoring

### 4.1 PRIMARY: directional accuracy

For each eligible cell: does `sign(forecast_ARM2 − forecast_ARM3)` match `sign(true_delta)`?

Report: accuracy rate with clustered bootstrap 95% CIs (10K resamples, seed 42, cluster = world\_\_param\_\_setting), per model and pooled.

### 4.2 SECONDARY: magnitude (response ratio)

`response_ratio = (forecast_ARM2 − forecast_ARM3) / true_delta`

Report: median, IQR, and fraction of cells where model response is <10% of true effect (microscopic) vs >200% (overshoot).

### 4.3 PLACEBO CHECK

Distribution of `(forecast_ARM1 − forecast_ARM3)` per model. Should be centered at zero. Report: mean, SD, and fraction of cells where |placebo_delta| > |true_delta|/2 (material placebo response).

### 4.4 Sign-reversal tracking

For (variable, parameter) pairs whose preregistered true response flips sign between h=4 and h=8 (documented in PREREGISTRATION.md Section 3): does the model's ARM2−ARM3 response also flip? Report as a paired within-cell metric (not independent horizon scoring). Expected sign-flip cells: pi under P1 (all worlds), r under P1/P2 (all worlds), w under P4 (W4), e under P1/P2 (W1/W3).

---

## 5. Models

| Model | API route | Pinned version | max_tokens | temp | Special |
|-------|-----------|----------------|-----------|------|---------|
| Claude Sonnet 4.6 | OpenRouter | `anthropic/claude-4.6-sonnet-20260217` | 8192 | 0 | — |
| GPT-5.5 | OpenRouter | `openai/gpt-5.5-20260423` | 8192 | 0 | `reasoning.effort=minimal` on all arms |
| Gemini 3.5 Flash | Google direct | `3.5-flash-05-2026` | 16384 | 0 | `response_mime_type=application/json` |
| Claude Opus 4.8 | OpenRouter | [same as structural run] | 8192 | 0 | — |

**DeepSeek V4 Pro:** excluded (45.5% parse rate in structural run; not recoverable).

**GPT-5.5 reasoning effort:** set to `minimal` from the start, uniform across all three arms. This matches the relabel v4 resume configuration and ensures consistent output volume.

### 5.1 Deterministic-decoding robustness (Gemini only)

For a random 10% subsample of ARM 2 cells (50 cells), rerun at temperature 0.7 × 3 independent samples. Report: within-cell agreement rate across the 3 samples, and whether conclusions change. Registered as robustness, not primary.

---

## 6. OOD extension

For World 1, Phillips slope only: add 2 out-of-range settings beyond the preregistered grid.

- Current grid: [0.1, 0.2, 0.4 (baseline), 0.6, 0.8]
- Below min: **0.05** (near-flat Phillips curve)
- Above max: **1.0** (strong inflation-output coupling)

**Stability check required:** before running, verify that both OOD settings produce finite GT values (no simulator explosion like World 3 at 0.7/0.9). Run GT computation for these 2 settings and confirm max|true_delta| < 100.

OOD calls: 2 settings × 10 seeds × ARM 2 only × 4 models = **80 calls** (ARM 1 and ARM 3 are already covered by the main grid).

---

## 7. Infrastructure

Reuse existing infrastructure verbatim:
- **SHA manifest** of all sent prompts (store prompt text + SHA256 hash)
- **Prompt storage:** full prompt text saved per call
- **Checkpoint/resume:** JSONL checkpoint with call_id deduplication
- **Credit guard:** per-model spend cap, halt if exceeded
- **Parse monitor:** 85% parse-rate threshold per model; halt and investigate if a model drops below
- **Post-run independent rescore:** verify_fork.py (independent logic, same standard as verify_relabel.py)

---

## 8. Cell count

### 8.1 Non-baseline settings after exclusions

| World | Parameter | Non-baseline settings | Count |
|-------|-----------|----------------------|------:|
| W1 | phillips_slope | 0.1, 0.2, 0.6, 0.8 | 4 |
| W1 | taylor_phi_pi | 1.1, 1.3, 2.0, 2.5 | 4 |
| W1 | is_sensitivity | 0.2, 0.4, 0.8, 1.0 | 4 |
| W2 | phillips_slope | 0.1, 0.2, 0.6, 0.8 | 4 |
| W2 | taylor_phi_pi | 1.1, 1.3, 2.0, 2.5 | 4 |
| W2 | is_sensitivity | 0.2, 0.4, 0.8, 1.0 | 4 |
| W3 | phillips_slope | 0.1, 0.3 *(0.7/0.9 excluded)* | 2 |
| W3 | taylor_phi_pi | 1.1, 1.5, 2.5, 3.0 | 4 |
| W3 | is_sensitivity | 0.2, 0.4, 0.8, 1.0 | 4 |
| W4 | phillips_slope | 0.05, 0.1, 0.3, 0.5 | 4 |
| W4 | taylor_phi_pi | 1.1, 1.3, 2.0, 2.5 | 4 |
| W4 | is_sensitivity | 0.2, 0.4, 0.8, 1.0 | 4 |
| W4 | wage_gap_slope | 0.1, 0.3, 0.7, 1.0 | 4 |
| **Total settings** | | | **50** |

Unique (world, param) combinations: 13

### 8.2 Call count

| Component | Calls per model | × 4 models | Total |
|-----------|---------------:|----------:|------:|
| ARM 3 (baseline) | 40 | 160 | 160 |
| ARM 1 (placebo) | 130 | 520 | 520 |
| ARM 2 (change) | 500 | 2,000 | 2,000 |
| OOD ARM 2 | 20 | 80 | 80 |
| **Subtotal** | **690** | | **2,760** |
| Gemini robustness (temp 0.7 × 3) | — | — | 150 |
| **Grand total** | | | **2,910** |

---

## 9. Cost estimate

### 9.1 Per-call costs (from verified actuals)

| Model | Source | Per-call cost |
|-------|--------|-------------:|
| Claude Sonnet 4.6 | v4 relabel: $19.53 / 650 calls | **$0.0300** |
| GPT-5.5 (effort=minimal) | v4 relabel: $31.69 / 650 calls | **$0.0487** |
| Gemini 3.5 Flash | v4 relabel: $15.68 / 650 calls | **$0.0241** |
| Claude Opus 4.8 | opus run: $65.02 / 1,400 calls | **$0.0464** |

Fork prompts have similar token counts to v4 relabel prompts (~5,000 input + ARM 2 equation text adds ~200 tokens; output is identical JSON format). Using v4 actuals is conservative.

### 9.2 Total cost by model

| Model | Calls | Cost/call | Model total | Platform |
|-------|------:|----------:|------------:|----------|
| Claude Sonnet 4.6 | 690 | $0.0300 | **$20.70** | OpenRouter |
| GPT-5.5 | 690 | $0.0487 | **$33.60** | OpenRouter |
| Gemini 3.5 Flash | 690 + 150 = 840 | $0.0241 | **$20.24** | Google |
| Claude Opus 4.8 | 690 | $0.0464 | **$32.02** | OpenRouter |
| **TOTAL** | **2,910** | | **$106.56** | |

### 9.3 By platform

| Platform | Models | Raw cost | +30% buffer | **LOAD TO** |
|----------|--------|----------:|------------:|------------:|
| **OpenRouter** | Sonnet + GPT-5.5 + Opus | $86.32 | $112.22 | **$113** |
| **Google** | Gemini | $20.24 | $26.31 | **$27** |
| **TOTAL** | | **$106.56** | **$138.53** | **$140** |

---

## LOAD OPENROUTER TO: $113. LOAD GOOGLE TO: $27.

Total: $140 (with 30% buffer over $107 estimated raw cost).

---

## 10. Multiple-testing plan

**Confirmatory (primary):** The fork ARM 2-vs-ARM 3 directional accuracy (per model, pooled across cells) is the single primary outcome. Report one test per model (4 tests total: does the model's 95% CI for accuracy exclude the permutation baseline?). BH correction within this family of 4.

**Exploratory:** All existing-grid analyses (told/infer, relabel, horizon breakdowns, world-level heterogeneity, sign-reversal tracking, response ratios, placebo check) are labeled exploratory. BH correction within the exploratory family. No cherry-picking between confirmatory and exploratory.

**Pre-registered before fork run executes.** Document in `docs/design/FORK_ANALYSIS_PLAN.md`.

---

## 11. Declined items (document for scope statement)

| Item | Decision | Rationale |
|------|----------|-----------|
| Code/calculator-enabled condition | Declined — future work | Out of scope for a letter |
| Full relabel redesign (affine rescaling/reordering) | Declined — future work | Relabel reframed as vocabulary-sensitivity test; decode probe is the manipulation check |
| Small neural baseline (LSTM/Transformer) | Declined | VAR + 5 simple baselines suffice for a letter |
| Multiple stochastic reps for all models | Declined | Temperature-0 justified + Gemini robustness slice |
| History-length robustness (30q VAR) | Declined — future work | VAR-side-only robustness; main results unchanged in spirit since VAR already dominates at all tiers; left to future work |

---

## 12. Phase 1 free analyses (no API calls, run while balances load)

1. **Expanded baselines:** balanced accuracy, sign-frequency, persistence, last-value, AR(1), pooled linear regression — alongside VAR on existing infer/told results.
2. **Equivalence tests (TOST):** told-vs-infer and relabel-vs-infer per model. Preregister margin ±3 pp before computing.
3. **MC_SE cell eligibility:** recompute cell eligibility via |true_delta| > 2×MC_SE on existing per-seed GT (using cross-seed variance as MC_SE proxy). Report headline table under this criterion alongside old tiers.
4. **Economic heterogeneity count:** cells where per-seed GT signs genuinely differ (not MC noise). Report separately.
5. **Full disaggregation tables:** world × parameter × horizon × intervention direction, with n per cell.
6. **Stationarity report:** classify each (world, parameter, setting) system from companion-matrix eigenvalues.
7. **History-length robustness:** rescore existing infer via 30-quarter truncated VAR (VAR-side only — no LLM reruns).

---

**HARD STOP.** This document is the complete Phase 0 deliverable. No API calls until Jake confirms balances are loaded.
