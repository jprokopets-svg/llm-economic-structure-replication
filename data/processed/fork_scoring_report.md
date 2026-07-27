# Fork Experiment Scoring Report

## 1. PRIMARY: Directional accuracy (ARM2−ARM3 vs GT)

Eligible cells: |true_delta| > 2×MC_SE. Clustered bootstrap 10K reps, seed 42, cluster = world__param__setting.

| Model | Accuracy | 95% CI | n eligible | n total |
|-------|--------:|-------:|----------:|--------:|
| Claude Opus 4.8 | 30.9% | [29.1%, 32.8%] | 5893 | 7620 |
| Claude Sonnet 4.6 | 35.1% | [32.2%, 38.2%] | 5754 | 7425 |
| GPT-5.5 | 49.9% | [46.8%, 53.3%] | 5893 | 7620 |
| Gemini 3.5 Flash | 42.5% | [40.0%, 45.1%] | 5480 | 7077 |

## 2. SECONDARY: Response ratio (model_delta / true_delta)

| Model | Median ratio | IQR | Fraction <10% | Fraction >200% | n |
|-------|------------:|----:|-------------:|---------------:|--:|
| Claude Opus 4.8 | 0.00 | [-0.20, 0.25] | 42.5% | 13.2% | 5893 |
| Claude Sonnet 4.6 | 0.00 | [-0.59, 0.55] | 30.9% | 23.5% | 5754 |
| GPT-5.5 | 0.00 | [-0.78, 1.38] | 14.9% | 33.5% | 5893 |
| Gemini 3.5 Flash | 0.00 | [-0.53, 1.06] | 25.9% | 29.1% | 5480 |

## 3. PLACEBO CHECK: ARM1−ARM3 distribution

| Model | Mean |placebo| | SD | Fraction |placebo| > |true_delta|/2 | n |
|-------|------------------:|---:|-------------------------------------------:|--:|
| Claude Opus 4.8 | 0.1319 | 0.2626 | 31.3% | 5893 |
| Claude Sonnet 4.6 | 0.3688 | 0.6438 | 54.9% | 5764 |
| GPT-5.5 | 0.4533 | 0.9872 | 60.3% | 5893 |
| Gemini 3.5 Flash | 0.4098 | 0.9641 | 51.7% | 5643 |

## 4. SIGN-REVERSAL TRACKING (paired h4→h8)

For (param, variable) pairs where GT flips sign between h=4 and h=8.

| Model | Param | Variable | GT flips | Model flips | Match rate | n |
|-------|-------|----------|--------:|:-----------:|----------:|--:|
| Claude Opus 4.8 | phillips_slope | pi | 84 | 10 | 12% | 140 |
| Claude Opus 4.8 | phillips_slope | r | 101 | 19 | 19% | 140 |
| Claude Opus 4.8 | taylor_phi_pi | r | 121 | 19 | 16% | 160 |
| Claude Opus 4.8 | wage_gap_slope | pi | 14 | 3 | 21% | 40 |
| Claude Opus 4.8 | wage_gap_slope | w | 21 | 2 | 10% | 40 |
| Claude Sonnet 4.6 | phillips_slope | pi | 82 | 27 | 33% | 136 |
| Claude Sonnet 4.6 | phillips_slope | r | 97 | 16 | 16% | 136 |
| Claude Sonnet 4.6 | taylor_phi_pi | r | 118 | 27 | 23% | 155 |
| Claude Sonnet 4.6 | wage_gap_slope | pi | 14 | 3 | 21% | 40 |
| Claude Sonnet 4.6 | wage_gap_slope | w | 21 | 5 | 24% | 40 |
| GPT-5.5 | phillips_slope | pi | 84 | 41 | 49% | 140 |
| GPT-5.5 | phillips_slope | r | 101 | 34 | 34% | 140 |
| GPT-5.5 | taylor_phi_pi | r | 121 | 49 | 40% | 160 |
| GPT-5.5 | wage_gap_slope | pi | 14 | 4 | 29% | 40 |
| GPT-5.5 | wage_gap_slope | w | 21 | 6 | 29% | 40 |
| Gemini 3.5 Flash | phillips_slope | pi | 79 | 29 | 37% | 133 |
| Gemini 3.5 Flash | phillips_slope | r | 95 | 36 | 38% | 133 |
| Gemini 3.5 Flash | taylor_phi_pi | r | 109 | 31 | 28% | 145 |
| Gemini 3.5 Flash | wage_gap_slope | pi | 14 | 1 | 7% | 40 |
| Gemini 3.5 Flash | wage_gap_slope | w | 21 | 9 | 43% | 40 |

## 5. CONFLICT CLASSIFICATION (exploratory)

Pre-fork trend direction vs sign(true_delta). NEUTRAL: |8q trend| < 0.5×rolling_sd. This analysis is **exploratory** — not part of the confirmatory primary outcome.

### 5a. Cell counts by classification

- ALIGNED: 10236
- CONFLICT: 9792
- NEUTRAL: 3544

### 5b. Directional accuracy by conflict class

| Model | Class | Accuracy | 95% CI | n |
|-------|-------|--------:|-------:|--:|
| Claude Opus 4.8 | ALIGNED | 35.3% | [32.4%, 38.2%] | 2559 |
| Claude Opus 4.8 | CONFLICT | 27.2% | [24.7%, 29.6%] | 2448 |
| Claude Opus 4.8 | NEUTRAL | 28.4% | [25.7%, 31.3%] | 886 |
| Claude Sonnet 4.6 | ALIGNED | 27.7% | [25.0%, 30.7%] | 2524 |
| Claude Sonnet 4.6 | CONFLICT | 43.5% | [39.4%, 47.8%] | 2417 |
| Claude Sonnet 4.6 | NEUTRAL | 33.3% | [29.8%, 36.8%] | 813 |
| GPT-5.5 | ALIGNED | 60.2% | [55.0%, 65.1%] | 2559 |
| GPT-5.5 | CONFLICT | 40.4% | [37.2%, 43.6%] | 2448 |
| GPT-5.5 | NEUTRAL | 46.7% | [42.3%, 51.1%] | 886 |
| Gemini 3.5 Flash | ALIGNED | 41.9% | [38.2%, 45.9%] | 2419 |
| Gemini 3.5 Flash | CONFLICT | 44.6% | [40.9%, 47.9%] | 2303 |
| Gemini 3.5 Flash | NEUTRAL | 37.9% | [32.9%, 43.0%] | 758 |

### 5c. Placebo movement by conflict class

| Model | Class | Mean |placebo| | n |
|-------|-------|------------------:|--:|
| Claude Opus 4.8 | ALIGNED | 0.1308 | 2559 |
| Claude Opus 4.8 | CONFLICT | 0.1314 | 2448 |
| Claude Opus 4.8 | NEUTRAL | 0.1367 | 886 |
| Claude Sonnet 4.6 | ALIGNED | 0.3544 | 2527 |
| Claude Sonnet 4.6 | CONFLICT | 0.3527 | 2420 |
| Claude Sonnet 4.6 | NEUTRAL | 0.4613 | 817 |
| GPT-5.5 | ALIGNED | 0.3898 | 2559 |
| GPT-5.5 | CONFLICT | 0.3892 | 2448 |
| GPT-5.5 | NEUTRAL | 0.8137 | 886 |
| Gemini 3.5 Flash | ALIGNED | 0.3805 | 2469 |
| Gemini 3.5 Flash | CONFLICT | 0.3682 | 2362 |
| Gemini 3.5 Flash | NEUTRAL | 0.6197 | 812 |

## 6. OOD EXTENSION (W1 phillips_slope)

| Model | Setting | Accuracy | n |
|-------|--------:|--------:|--:|
| Claude Opus 4.8 | 0.05 | 32.8% | 119 |
| Claude Opus 4.8 | 1.0 | 28.0% | 125 |
| Claude Sonnet 4.6 | 0.05 | 37.0% | 108 |
| Claude Sonnet 4.6 | 1.0 | 42.0% | 112 |
| GPT-5.5 | 0.05 | 59.7% | 119 |
| GPT-5.5 | 1.0 | 41.6% | 125 |
| Gemini 3.5 Flash | 0.05 | 50.4% | 119 |
| Gemini 3.5 Flash | 1.0 | 38.4% | 125 |
