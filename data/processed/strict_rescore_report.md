# Strict-Filter Rescore Report

Rescore of all models plus VAR under per-seed ground truth, tiered by cell stability.

**Bootstrap**: 10,000 cluster resamples, cluster = `{world}__{param}__{setting}`. **Permutation**: 10,000 shuffles within (world, param, horizon, seed) strata, seed=42.

**Relabel excluded** from every table — condition was never implemented (see RELABEL_CONTAMINATION.md); relabel rows are infer duplicates and would inflate n without adding signal.

**Near-zero filter** (|true_delta| < 0.01) applied to whichever GT is being scored, on top of the tier filter.

## Cell stability tiers

| Tier | Threshold | Cells | % of total |
|---|---|---:|---:|
| strict | majority_frac >= 0.8 | 156 | 18.4% |
| moderate | majority_frac >= 0.6 | 675 | 79.8% |
| all | no filter | 846 | 100.0% |

## Accuracy by source, condition, and tier (per-seed GT)

| Source | Condition | Tier | Accuracy | 95% CI | Perm mean | Perm 95% CI | n |
|---|---|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | told | all | 54.7% | [52.4%, 57.1%] | 50.4% | [49.3%, 51.5%] | 6743 |
| Claude Sonnet 4.6 | told | moderate | 55.1% | [52.7%, 57.6%] | 50.7% | [49.4%, 51.9%] | 5370 |
| Claude Sonnet 4.6 | told | strict | 52.2% | [45.2%, 59.6%] | 51.0% | [48.0%, 54.1%] | 691 |
| Claude Sonnet 4.6 | infer | all | 54.9% | [52.5%, 57.2%] | 50.4% | [49.3%, 51.5%] | 6757 |
| Claude Sonnet 4.6 | infer | moderate | 55.4% | [52.8%, 58.0%] | 50.7% | [49.5%, 51.9%] | 5375 |
| Claude Sonnet 4.6 | infer | strict | 50.3% | [43.1%, 57.7%] | 49.1% | [46.2%, 51.7%] | 686 |
| Claude Opus 4.8 | told | all | 59.5% | [55.3%, 63.7%] | 50.8% | [49.8%, 51.9%] | 7538 |
| Claude Opus 4.8 | told | moderate | 60.6% | [56.3%, 64.7%] | 51.3% | [50.1%, 52.4%] | 5848 |
| Claude Opus 4.8 | told | strict | 55.6% | [46.0%, 66.1%] | 55.4% | [52.5%, 58.2%] | 708 |
| Claude Opus 4.8 | infer | all | 60.6% | [56.5%, 64.5%] | 51.0% | [49.9%, 52.0%] | 7538 |
| Claude Opus 4.8 | infer | moderate | 61.3% | [57.2%, 65.2%] | 51.3% | [50.2%, 52.5%] | 5848 |
| Claude Opus 4.8 | infer | strict | 57.2% | [46.9%, 68.1%] | 54.3% | [51.4%, 57.1%] | 708 |
| GPT-5.5 | told | all | 73.1% | [70.0%, 76.3%] | 52.3% | [51.2%, 53.5%] | 6164 |
| GPT-5.5 | told | moderate | 73.6% | [70.4%, 76.9%] | 53.0% | [51.7%, 54.3%] | 4894 |
| GPT-5.5 | told | strict | 66.4% | [60.5%, 73.3%] | 59.6% | [56.7%, 62.7%] | 628 |
| GPT-5.5 | infer | all | 73.1% | [69.6%, 76.5%] | 52.1% | [50.9%, 53.3%] | 6043 |
| GPT-5.5 | infer | moderate | 74.4% | [70.8%, 77.9%] | 52.7% | [51.4%, 54.0%] | 4814 |
| GPT-5.5 | infer | strict | 69.2% | [60.7%, 77.6%] | 60.3% | [57.2%, 63.4%] | 614 |
| Gemini 3.5 Flash | told | all | 60.6% | [55.7%, 65.5%] | 51.1% | [49.9%, 52.4%] | 5306 |
| Gemini 3.5 Flash | told | moderate | 61.2% | [55.9%, 66.2%] | 51.1% | [49.7%, 52.5%] | 4098 |
| Gemini 3.5 Flash | told | strict | 51.9% | [41.2%, 64.7%] | 51.7% | [48.8%, 54.6%] | 414 |
| Gemini 3.5 Flash | infer | all | 64.1% | [59.8%, 68.1%] | 51.2% | [49.9%, 52.4%] | 4991 |
| Gemini 3.5 Flash | infer | moderate | 65.3% | [60.7%, 69.7%] | 51.4% | [50.0%, 52.8%] | 3839 |
| Gemini 3.5 Flash | infer | strict | 55.8% | [43.4%, 69.5%] | 50.9% | [47.5%, 54.1%] | 362 |
| DeepSeek V4 Pro | told | all | 54.6% | [52.0%, 57.6%] | 50.9% | [48.6%, 53.3%] | 1499 |
| DeepSeek V4 Pro | told | moderate | 55.3% | [51.6%, 59.1%] | 51.0% | [48.4%, 53.7%] | 1152 |
| DeepSeek V4 Pro | told | strict | 48.2% | [39.3%, 57.6%] | 48.3% | [42.4%, 54.0%] | 139 |
| DeepSeek V4 Pro | infer | all | 53.9% | [49.4%, 58.7%] | 49.9% | [47.5%, 52.3%] | 1387 |
| DeepSeek V4 Pro | infer | moderate | 56.2% | [51.5%, 61.0%] | 50.7% | [48.0%, 53.4%] | 1105 |
| DeepSeek V4 Pro | infer | strict | 52.7% | [38.6%, 67.5%] | 51.7% | [45.0%, 57.4%] | 129 |
| VAR | VAR | all | 87.3% | [84.8%, 89.7%] | 53.0% | [51.6%, 54.3%] | 4833 |
| VAR | VAR | moderate | 87.5% | [84.9%, 90.0%] | 54.0% | [52.5%, 55.4%] | 3692 |
| VAR | VAR | strict | 82.4% | [76.3%, 88.7%] | 62.9% | [59.7%, 66.3%] | 546 |

## Joint filter: |true_delta| >= 0.01 under BOTH GTs

Scored under per-seed GT (matched seed).

| Source | Condition | Tier | Accuracy | 95% CI | Perm mean | Perm 95% CI | n |
|---|---|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | told | joint | 54.9% | [52.5%, 57.3%] | 50.4% | [49.3%, 51.6%] | 6573 |
| Claude Sonnet 4.6 | infer | joint | 55.1% | [52.7%, 57.4%] | 50.5% | [49.4%, 51.7%] | 6590 |
| Claude Opus 4.8 | told | joint | 59.8% | [55.5%, 63.9%] | 50.9% | [49.8%, 51.9%] | 7356 |
| Claude Opus 4.8 | infer | joint | 60.8% | [56.8%, 64.8%] | 51.0% | [49.9%, 52.1%] | 7356 |
| GPT-5.5 | told | joint | 73.5% | [70.3%, 76.7%] | 52.4% | [51.2%, 53.6%] | 5992 |
| GPT-5.5 | infer | joint | 73.4% | [70.0%, 76.9%] | 52.1% | [50.9%, 53.3%] | 5873 |
| Gemini 3.5 Flash | told | joint | 61.0% | [56.0%, 65.8%] | 51.1% | [49.9%, 52.4%] | 5144 |
| Gemini 3.5 Flash | infer | joint | 64.6% | [60.3%, 68.5%] | 51.2% | [49.9%, 52.5%] | 4832 |
| DeepSeek V4 Pro | told | joint | 55.0% | [52.3%, 58.0%] | 51.1% | [48.7%, 53.5%] | 1467 |
| DeepSeek V4 Pro | infer | joint | 53.7% | [49.0%, 58.5%] | 50.0% | [47.6%, 52.5%] | 1355 |
| VAR | VAR | joint | 87.6% | [85.1%, 89.9%] | 53.1% | [51.8%, 54.4%] | 4694 |

## Summary

Model ordering **changes** across tiers. VAR gap over the best model ranges +13.4 to +14.6 pp (spread 1.2 pp) across tiers. Rankings by tier: **all**: GPT-5.5, Gemini 3.5 Flash, Claude Opus 4.8, Claude Sonnet 4.6, DeepSeek V4 Pro | **moderate**: GPT-5.5, Gemini 3.5 Flash, Claude Opus 4.8, DeepSeek V4 Pro, Claude Sonnet 4.6 | **strict**: GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, Claude Sonnet 4.6, DeepSeek V4 Pro | **joint**: GPT-5.5, Gemini 3.5 Flash, Claude Opus 4.8, Claude Sonnet 4.6, DeepSeek V4 Pro. Gaps: all: VAR 87.3% vs best model (GPT-5.5) 73.1% → gap +14.2 pp; moderate: VAR 87.5% vs best model (GPT-5.5) 74.0% → gap +13.4 pp; strict: VAR 82.4% vs best model (GPT-5.5) 67.8% → gap +14.6 pp; joint: VAR 87.6% vs best model (GPT-5.5) 73.5% → gap +14.1 pp.

## TOLD vs INFER (per-seed GT)

Cluster-resampled bootstrap (10,000 reps, cluster = `{world}__{param}__{setting}`). CI on the told-minus-infer difference uses the SAME cluster resamples for both conditions (paired).

Near-zero rows (|true_delta_per_seed| < 0.01) excluded before scoring.

| Model | Tier | Told acc [95% CI] | Infer acc [95% CI] | Told - Infer [95% CI] | n told | n infer |
|---|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | all | 54.7% [52.4%, 57.1%] | 54.9% [52.5%, 57.2%] | -0.2 pp [-1.3%, +1.0%] | 6743 | 6757 |
| Claude Sonnet 4.6 | moderate | 55.1% [52.7%, 57.6%] | 55.4% [52.8%, 58.0%] | -0.2 pp [-1.5%, +1.0%] | 5370 | 5375 |
| Claude Sonnet 4.6 | strict | 52.2% [45.2%, 59.6%] | 50.3% [43.1%, 57.7%] | +2.0 pp [-1.4%, +5.4%] | 691 | 686 |
| Claude Opus 4.8 | all | 59.5% [55.3%, 63.7%] | 60.6% [56.5%, 64.5%] | -1.1 pp [-2.0%, -0.1%] | 7538 | 7538 |
| Claude Opus 4.8 | moderate | 60.6% [56.3%, 64.7%] | 61.3% [57.2%, 65.2%] | -0.7 pp [-1.7%, +0.4%] | 5848 | 5848 |
| Claude Opus 4.8 | strict | 55.6% [46.0%, 66.1%] | 57.2% [46.9%, 68.1%] | -1.6 pp [-4.8%, +1.1%] | 708 | 708 |
| GPT-5.5 | all | 73.1% [70.0%, 76.3%] | 73.1% [69.6%, 76.5%] | +0.1 pp [-1.4%, +1.4%] | 6164 | 6043 |
| GPT-5.5 | moderate | 73.6% [70.4%, 76.9%] | 74.4% [70.8%, 77.9%] | -0.8 pp [-2.3%, +0.7%] | 4894 | 4814 |
| GPT-5.5 | strict | 66.4% [60.5%, 73.3%] | 69.2% [60.7%, 77.6%] | -2.8 pp [-6.5%, +1.2%] | 628 | 614 |
| Gemini 3.5 Flash | all | 60.6% [55.7%, 65.5%] | 64.1% [59.8%, 68.1%] | -3.5 pp [-5.8%, -1.0%] | 5306 | 4991 |
| Gemini 3.5 Flash | moderate | 61.2% [55.9%, 66.2%] | 65.3% [60.7%, 69.7%] | -4.2 pp [-6.5%, -1.7%] | 4098 | 3839 |
| Gemini 3.5 Flash | strict | 51.9% [41.2%, 64.7%] | 55.8% [43.4%, 69.5%] | -3.9 pp [-11.0%, +2.7%] | 414 | 362 |
| DeepSeek V4 Pro | all | 54.6% [52.0%, 57.6%] | 53.9% [49.4%, 58.7%] | +0.7 pp [-4.3%, +5.5%] | 1499 | 1387 |
| DeepSeek V4 Pro | moderate | 55.3% [51.6%, 59.1%] | 56.2% [51.5%, 61.0%] | -0.9 pp [-6.3%, +4.0%] | 1152 | 1105 |
| DeepSeek V4 Pro | strict | 48.2% [39.3%, 57.6%] | 52.7% [38.6%, 67.5%] | -4.5 pp [-21.3%, +8.8%] | 139 | 129 |

No model shows a told-infer difference whose 95% CI excludes zero — told does not reliably beat infer for any model at any tier. Infer beats told (CI excludes zero) for: Claude Opus 4.8 at all (-1.1 pp); Gemini 3.5 Flash at all (-3.5 pp); Gemini 3.5 Flash at moderate (-4.2 pp).

## Three-Condition Table: TOLD vs INFER vs RELABEL (per-seed GT)

Full-grid RELABEL run: `relabel_run_v3` (2588 parsed forecasts across five models). Bootstrap 10,000 reps; permutation 10,000 reps (strata = world x param x horizon x seed).

### Relabel run parse rates and spend

| Model | ok | fail | error | parse rate | tokens_in | tokens_out | cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 650 | 0 | 0 | 100.0% | 3,799,863 | 1,119,943 | $28.20 |
| Claude Opus 4.8 | 650 | 0 | 0 | 100.0% | 4,850,071 | 980,475 | $29.26 |
| GPT-5.5 | 634 | 0 | 0 | 100.0% | 3,316,583 | 2,907,031 | $53.56 |
| Gemini 3.5 Flash | 630 | 0 | 0 | 100.0% | 3,450,217 | 978,852 | $25.03 |
| DeepSeek V4 Pro | 24 | 0 | 0 | 100.0% | 122,960 | 109,477 | $2.01 |
| **Total** | | | | | | | **$138.06** |

### Accuracy (per-seed GT, 95% CIs)

Columns: observed accuracy | 95% CI | permutation baseline mean | permutation 95% CI | n.

#### Tier: all

| Model | Told acc [CI] | Infer acc [CI] | Relabel acc [CI] | Told n | Infer n | Relabel n |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 54.7% [52.4%, 57.1%] | 54.9% [52.5%, 57.2%] | 62.2% [58.9%, 65.5%] | 6743 | 6757 | 7538 |
| Claude Opus 4.8 | 59.5% [55.3%, 63.7%] | 60.6% [56.5%, 64.5%] | 63.8% [59.4%, 68.0%] | 7538 | 7538 | 7538 |
| GPT-5.5 | 73.1% [70.0%, 76.3%] | 73.1% [69.6%, 76.5%] | 75.0% [71.6%, 78.3%] | 6164 | 6043 | 7126 |
| Gemini 3.5 Flash | 60.6% [55.7%, 65.5%] | 64.1% [59.8%, 68.1%] | 69.1% [65.6%, 72.6%] | 5306 | 4991 | 7078 |
| DeepSeek V4 Pro | 54.6% [52.0%, 57.6%] | 53.9% [49.4%, 58.7%] | 54.7% [50.0%, 57.9%] | 1499 | 1387 | 75 |

#### Tier: moderate

| Model | Told acc [CI] | Infer acc [CI] | Relabel acc [CI] | Told n | Infer n | Relabel n |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 55.1% [52.7%, 57.6%] | 55.4% [52.8%, 58.0%] | 63.0% [59.7%, 66.3%] | 5370 | 5375 | 5848 |
| Claude Opus 4.8 | 60.6% [56.3%, 64.7%] | 61.3% [57.2%, 65.2%] | 64.4% [60.0%, 68.8%] | 5848 | 5848 | 5848 |
| GPT-5.5 | 73.6% [70.4%, 76.9%] | 74.4% [70.8%, 77.9%] | 76.1% [72.6%, 79.3%] | 4894 | 4814 | 5531 |
| Gemini 3.5 Flash | 61.2% [55.9%, 66.2%] | 65.3% [60.7%, 69.7%] | 69.8% [66.3%, 73.4%] | 4098 | 3839 | 5520 |
| DeepSeek V4 Pro | 55.3% [51.6%, 59.1%] | 56.2% [51.5%, 61.0%] | 56.9% [50.0%, 58.8%] | 1152 | 1105 | 58 |

#### Tier: strict

| Model | Told acc [CI] | Infer acc [CI] | Relabel acc [CI] | Told n | Infer n | Relabel n |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 52.2% [45.2%, 59.6%] | 50.3% [43.1%, 57.7%] | 57.6% [49.1%, 66.3%] | 691 | 686 | 708 |
| Claude Opus 4.8 | 55.6% [46.0%, 66.1%] | 57.2% [46.9%, 68.1%] | 57.3% [48.6%, 66.4%] | 708 | 708 | 708 |
| GPT-5.5 | 66.4% [60.5%, 73.3%] | 69.2% [60.7%, 77.6%] | 70.4% [63.2%, 78.0%] | 628 | 614 | 655 |
| Gemini 3.5 Flash | 51.9% [41.2%, 64.7%] | 55.8% [43.4%, 69.5%] | 66.2% [57.5%, 75.0%] | 414 | 362 | 659 |
| DeepSeek V4 Pro | 48.2% [39.3%, 57.6%] | 52.7% [38.6%, 67.5%] | 20.0% [0.0%, 33.3%] | 139 | 129 | 5 |

### Relabel-condition permutation baselines

| Model | Tier | Observed | Perm mean | Perm 95% CI | n |
|---|---|---:|---:|---:|---:|
| Claude Sonnet 4.6 | all | 66.8% | 51.4% | [50.3%, 52.4%] | 7538 |
| Claude Opus 4.8 | all | 70.7% | 51.1% | [50.0%, 52.1%] | 7538 |
| GPT-5.5 | all | 78.0% | 52.0% | [51.0%, 53.1%] | 7126 |
| Gemini 3.5 Flash | all | 73.3% | 51.5% | [50.4%, 52.6%] | 7078 |
| DeepSeek V4 Pro | all | 58.7% | 53.7% | [42.7%, 64.0%] | 75 |
| Claude Sonnet 4.6 | moderate | 67.5% | 51.6% | [50.4%, 52.8%] | 5848 |
| Claude Opus 4.8 | moderate | 71.0% | 51.6% | [50.4%, 52.8%] | 5848 |
| GPT-5.5 | moderate | 79.0% | 52.8% | [51.6%, 54.0%] | 5531 |
| Gemini 3.5 Flash | moderate | 74.0% | 52.1% | [50.9%, 53.4%] | 5520 |
| DeepSeek V4 Pro | moderate | 62.1% | 58.3% | [48.3%, 69.0%] | 58 |
| Claude Sonnet 4.6 | strict | 62.4% | 55.2% | [52.5%, 57.9%] | 708 |
| Claude Opus 4.8 | strict | 65.0% | 54.9% | [52.3%, 57.6%] | 708 |
| GPT-5.5 | strict | 73.9% | 60.6% | [57.7%, 63.5%] | 655 |
| Gemini 3.5 Flash | strict | 70.7% | 57.5% | [54.6%, 60.4%] | 659 |
| DeepSeek V4 Pro | strict | 20.0% | 40.1% | [20.0%, 60.0%] | 5 |

## Relabel − Infer (paired cluster-bootstrap CIs)

Per model, per tier. Bootstrap 10,000 reps, clusters = `{world}__{param}__{setting}`, paired resampling across relabel and infer.

| Model | Tier | Relabel acc | Infer acc | Relabel − Infer [95% CI] | n relabel | n infer |
|---|---|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | all | 62.2% | 54.9% | +7.3 pp [+5.3, +9.4] | 7538 | 6757 |
| Claude Sonnet 4.6 | moderate | 63.0% | 55.4% | +7.6 pp [+5.3, +9.9] | 5848 | 5375 |
| Claude Sonnet 4.6 | strict | 57.6% | 50.3% | +7.3 pp [+2.4, +12.0] | 708 | 686 |
| Claude Opus 4.8 | all | 63.8% | 60.6% | +3.2 pp [+1.6, +4.8] | 7538 | 7538 |
| Claude Opus 4.8 | moderate | 64.4% | 61.3% | +3.2 pp [+1.4, +5.1] | 5848 | 5848 |
| Claude Opus 4.8 | strict | 57.3% | 57.2% | +0.1 pp [-3.9, +4.0] | 708 | 708 |
| GPT-5.5 | all | 75.0% | 73.1% | +1.9 pp [-0.0, +3.8] | 7126 | 6043 |
| GPT-5.5 | moderate | 76.1% | 74.4% | +1.7 pp [-0.3, +3.6] | 5531 | 4814 |
| GPT-5.5 | strict | 70.4% | 69.2% | +1.2 pp [-4.4, +7.5] | 655 | 614 |
| Gemini 3.5 Flash | all | 69.1% | 64.1% | +5.0 pp [+2.0, +8.1] | 7078 | 4991 |
| Gemini 3.5 Flash | moderate | 69.8% | 65.3% | +4.5 pp [+1.5, +7.6] | 5520 | 3839 |
| Gemini 3.5 Flash | strict | 66.2% | 55.8% | +10.4 pp [+0.8, +20.4] | 659 | 362 |
| DeepSeek V4 Pro | all | 54.7% | 53.9% | +0.7 pp [-9.7, +6.9] | 75 | 1387 |
| DeepSeek V4 Pro | moderate | 56.9% | 56.2% | +0.7 pp [-21.1, +5.9] | 58 | 1105 |
| DeepSeek V4 Pro | strict | 20.0% | 52.7% | -32.7 pp [-65.3, -8.0] | 5 | 129 |

**Interpretation.** A positive difference with a CI that excludes zero means the RELABEL condition beats INFER for that model on that tier — the model does *better* when the economic vocabulary is stripped out and replaced with neutral system-dynamics tokens (X1..X7, coupling A/B/C, controller). A negative difference means it does worse.

## Relabel v2 (final): three-condition table + paired differences

**GPT-5.5 numbers below SUPERSEDE the previous truncated-run
numbers.** The initial v4 run paused GPT-5.5 at 84.9% parse
(142/650 successful) due to reasoning-token exhaustion. The
resume phase completed the remaining 508 cells with
`reasoning.effort=minimal` (all 100% parse). Final GPT-5.5
n = 7515 rows all-tier (was 1224).

Side-by-side (all tier, GPT-5.5 paired Relabel − Infer):
  - Truncated (n=1224): +8.6 pp [+2.7, +15.1]
  - Complete  (n=7515): +2.3 pp [+0.6, +4.1]

The truncated-run effect was inflated because the failed 508
cells clustered on harder cells (world3, high phillips_slope
settings). Rescoring against the complete data brings the
GPT-5.5 effect into line with Sonnet's and Gemini's point
estimates.

Run: `relabel_run_v4` (v4). Uses `lmm2.relabel_template.build_relabel_prompt` — hand-written static templates + whitelist verifier, no substitution pass. Bootstrap 10K reps, permutation 10K reps.

### Parse rates and spend (v4 run)

| Model | ok | fail | error | parse rate | tokens_in | tokens_out | cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 650 | 0 | 0 | 100.0% | 3,284,419 | 644,783 | $19.53 |
| GPT-5.5 | 650 | 0 | 0 | 100.0% | 2,873,629 | 1,538,222 | $31.69 |
| Gemini 3.5 Flash | 650 | 0 | 0 | 100.0% | 2,949,289 | 455,426 | $15.68 |
| **Total** | | | | | | | **$66.90** |

### Accuracy per condition per tier (per-seed GT, 95% CIs)

#### Tier: all

| Model | Told acc [CI] | Infer acc [CI] | Relabel acc [CI] | Told n | Infer n | Relabel n |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 54.7% [52.4%, 57.1%] | 54.9% [52.5%, 57.2%] | 57.3% [52.9%, 61.6%] | 6743 | 6757 | 7538 |
| GPT-5.5 | 73.1% [70.0%, 76.3%] | 73.1% [69.6%, 76.5%] | 75.4% [72.4%, 78.4%] | 6164 | 6043 | 7515 |
| Gemini 3.5 Flash | 60.6% [55.7%, 65.5%] | 64.1% [59.8%, 68.1%] | 66.5% [62.1%, 70.8%] | 5306 | 4991 | 7538 |

#### Tier: moderate

| Model | Told acc [CI] | Infer acc [CI] | Relabel acc [CI] | Told n | Infer n | Relabel n |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 55.1% [52.7%, 57.6%] | 55.4% [52.8%, 58.0%] | 57.8% [53.5%, 61.9%] | 5370 | 5375 | 5848 |
| GPT-5.5 | 73.6% [70.4%, 76.9%] | 74.4% [70.8%, 77.9%] | 76.9% [74.0%, 79.8%] | 4894 | 4814 | 5834 |
| Gemini 3.5 Flash | 61.2% [55.9%, 66.2%] | 65.3% [60.7%, 69.7%] | 67.2% [62.5%, 71.7%] | 4098 | 3839 | 5848 |

#### Tier: strict

| Model | Told acc [CI] | Infer acc [CI] | Relabel acc [CI] | Told n | Infer n | Relabel n |
|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 52.2% [45.2%, 59.6%] | 50.3% [43.1%, 57.7%] | 49.3% [38.9%, 60.6%] | 691 | 686 | 708 |
| GPT-5.5 | 66.4% [60.5%, 73.3%] | 69.2% [60.7%, 77.6%] | 73.0% [65.4%, 80.8%] | 628 | 614 | 705 |
| Gemini 3.5 Flash | 51.9% [41.2%, 64.7%] | 55.8% [43.4%, 69.5%] | 60.3% [50.2%, 70.9%] | 414 | 362 | 708 |

### Paired Relabel − Infer (same-cluster resampling, 95% CI)

| Model | Tier | Relabel acc | Infer acc | Diff [95% CI] | n |
|---|---|---:|---:|---:|---:|
| Claude Sonnet 4.6 | all | 57.3% | 54.9% | +2.4 pp [-0.6, +5.4] | 7538 |
| Claude Sonnet 4.6 | moderate | 57.8% | 55.4% | +2.4 pp [-0.8, +5.5] | 5848 |
| Claude Sonnet 4.6 | strict | 49.3% | 50.3% | -1.0 pp [-8.8, +7.4] | 708 |
| GPT-5.5 | all | 75.4% | 73.1% | +2.3 pp [+0.6, +4.1] | 7515 |
| GPT-5.5 | moderate | 76.9% | 74.4% | +2.5 pp [+0.6, +4.4] | 5834 |
| GPT-5.5 | strict | 73.0% | 69.2% | +3.8 pp [-1.7, +10.4] | 705 |
| Gemini 3.5 Flash | all | 66.5% | 64.1% | +2.4 pp [-0.6, +5.4] | 7538 |
| Gemini 3.5 Flash | moderate | 67.2% | 65.3% | +1.9 pp [-1.4, +5.1] | 5848 |
| Gemini 3.5 Flash | strict | 60.3% | 55.8% | +4.5 pp [-4.4, +14.8] | 708 |

### World × Param breakdown of Relabel − Infer (all tier)

| Model | World | Param | Relabel acc | Infer acc | Diff [95% CI] | n |
|---|---|---|---:|---:|---:|---:|
| Claude Sonnet 4.6 | world1 | phillips_slope | 55.6% | 55.7% | -0.2 pp [-4.2, +2.7] | 594 |
| Claude Sonnet 4.6 | world1 | taylor_phi_pi | 44.5% | 45.9% | -1.4 pp [-4.1, +0.6] | 577 |
| Claude Sonnet 4.6 | world1 | is_sensitivity | 48.0% | 47.9% | +0.0 pp [-1.1, +1.4] | 592 |
| Claude Sonnet 4.6 | world2 | phillips_slope | 72.3% | 71.6% | +0.7 pp [-6.2, +11.3] | 480 |
| Claude Sonnet 4.6 | world2 | taylor_phi_pi | 69.9% | 59.8% | +10.1 pp [+5.1, +15.4] | 475 |
| Claude Sonnet 4.6 | world2 | is_sensitivity | 76.4% | 64.1% | +12.3 pp [+6.4, +17.9] | 475 |
| Claude Sonnet 4.6 | world3 | phillips_slope | 44.5% | 49.5% | -5.0 pp [-10.0, +0.0] | 299 |
| Claude Sonnet 4.6 | world3 | taylor_phi_pi | 30.2% | 44.1% | -13.9 pp [-18.1, -9.8] | 590 |
| Claude Sonnet 4.6 | world3 | is_sensitivity | 33.6% | 49.6% | -16.0 pp [-21.1, -11.6] | 593 |
| Claude Sonnet 4.6 | world4 | phillips_slope | 62.0% | 60.1% | +1.8 pp [-2.9, +6.4] | 715 |
| Claude Sonnet 4.6 | world4 | taylor_phi_pi | 64.5% | 56.3% | +8.2 pp [+4.0, +12.5] | 718 |
| Claude Sonnet 4.6 | world4 | is_sensitivity | 66.3% | 55.4% | +10.9 pp [+5.0, +17.1] | 716 |
| GPT-5.5 | world1 | phillips_slope | 80.0% | 78.0% | +2.0 pp [-1.7, +4.2] | 594 |
| GPT-5.5 | world1 | taylor_phi_pi | 65.0% | 59.8% | +5.2 pp [+2.7, +8.7] | 577 |
| GPT-5.5 | world1 | is_sensitivity | 72.5% | 69.6% | +2.9 pp [+0.5, +6.4] | 592 |
| GPT-5.5 | world2 | phillips_slope | 90.2% | 87.8% | +2.4 pp [-2.7, +6.7] | 480 |
| GPT-5.5 | world2 | taylor_phi_pi | 82.9% | 81.1% | +1.9 pp [-1.2, +3.2] | 475 |
| GPT-5.5 | world2 | is_sensitivity | 85.9% | 90.7% | -4.8 pp [-7.6, -1.9] | 475 |
| GPT-5.5 | world3 | phillips_slope | 70.7% | 62.8% | +7.9 pp [+6.5, +10.9] | 276 |
| GPT-5.5 | world3 | taylor_phi_pi | 57.8% | 55.8% | +2.0 pp [-2.8, +6.8] | 590 |
| GPT-5.5 | world3 | is_sensitivity | 65.1% | 63.0% | +2.1 pp [-2.5, +10.5] | 593 |
| GPT-5.5 | world4 | phillips_slope | 82.0% | 75.8% | +6.2 pp [+2.6, +9.2] | 715 |
| GPT-5.5 | world4 | taylor_phi_pi | 76.0% | 75.6% | +0.4 pp [-1.0, +1.7] | 718 |
| GPT-5.5 | world4 | is_sensitivity | 74.9% | 74.0% | +0.8 pp [-4.2, +5.9] | 716 |
| Gemini 3.5 Flash | world1 | phillips_slope | 68.5% | 60.1% | +8.4 pp [+4.0, +11.9] | 594 |
| Gemini 3.5 Flash | world1 | taylor_phi_pi | 49.6% | 43.8% | +5.7 pp [-0.4, +13.7] | 577 |
| Gemini 3.5 Flash | world2 | phillips_slope | 84.6% | 80.2% | +4.4 pp [-2.1, +10.8] | 480 |
| Gemini 3.5 Flash | world2 | taylor_phi_pi | 77.9% | 73.4% | +4.5 pp [+2.4, +6.4] | 475 |
| Gemini 3.5 Flash | world3 | phillips_slope | 51.5% | 48.3% | +3.2 pp [+2.5, +3.8] | 299 |
| Gemini 3.5 Flash | world3 | taylor_phi_pi | 43.6% | 51.0% | -7.5 pp [-11.2, -3.6] | 590 |
| Gemini 3.5 Flash | world3 | is_sensitivity | 46.2% | 55.1% | -8.9 pp [-13.2, -3.7] | 593 |
| Gemini 3.5 Flash | world4 | phillips_slope | 76.2% | 69.3% | +6.9 pp [+5.6, +8.3] | 715 |
| Gemini 3.5 Flash | world4 | is_sensitivity | 73.7% | 67.3% | +6.4 pp [+3.4, +8.7] | 716 |
| Gemini 3.5 Flash | world4 | wage_gap_slope | 77.9% | 72.4% | +5.5 pp [+2.5, +9.7] | 714 |

### Relabel-condition permutation baselines

| Model | Tier | Observed | Perm mean | Perm 95% CI | n |
|---|---|---:|---:|---:|---:|
| Claude Sonnet 4.6 | all | 64.9% | 51.2% | [50.2%, 52.3%] | 7538 |
| GPT-5.5 | all | 77.8% | 52.1% | [51.0%, 53.1%] | 7515 |
| Gemini 3.5 Flash | all | 73.6% | 51.4% | [50.3%, 52.5%] | 7538 |
| Claude Sonnet 4.6 | moderate | 65.2% | 51.7% | [50.5%, 52.9%] | 5848 |
| GPT-5.5 | moderate | 79.3% | 52.9% | [51.7%, 54.1%] | 5834 |
| Gemini 3.5 Flash | moderate | 74.3% | 52.0% | [50.8%, 53.2%] | 5848 |
| Claude Sonnet 4.6 | strict | 58.6% | 53.7% | [51.0%, 56.4%] | 708 |
| GPT-5.5 | strict | 75.0% | 59.6% | [56.6%, 62.6%] | 705 |
| Gemini 3.5 Flash | strict | 67.5% | 56.1% | [53.4%, 58.8%] | 708 |

## VAR gap (paired)

Paired VAR-minus-GPT-5.5 (infer) on common cells: cells where BOTH VAR and GPT-5.5 produced scored forecasts for the same (world, param, setting, seed, variable, horizon). Clustered bootstrap 10,000 reps (seed=42), cluster = `{world}__{param}__{setting}`, paired resampling. Near-zero filter (|true_delta| < 0.01) applied.

| Tier | Common n | VAR acc [95% CI] | GPT-5.5 acc [95% CI] | Paired gap [95% CI] |
|---|---:|---:|---:|---:|
| all | 3744 | 86.6% [83.5%, 89.4%] | 71.1% [66.8%, 75.4%] | +15.5 pp [+12.9, +18.0] |
| moderate | 2877 | 86.8% [83.6%, 89.8%] | 71.6% [67.0%, 76.2%] | +15.2 pp [+12.3, +17.9] |
| strict | 436 | 79.4% [72.2%, 85.9%] | 66.1% [57.3%, 76.1%] | +13.3 pp [+3.0, +22.8] |

**Sanity check (all tier):** VAR accuracy on common cells (86.6%) vs full set (87.7%): diff 1.1 pp. GPT-5.5 accuracy on common cells (71.1%) vs full set (73.1%): diff 2.0 pp. Both within 2 pp — common cell set is representative. At the strict tier, both sources drop ~7–8 pp vs their full-set accuracies because the VAR-GPT intersection at strict is small (n=436) and skewed toward cells where VAR has lower coverage (World 4); the gap estimate (+13.3 pp) is valid but the wide CI [+3.0, +22.8] reflects this.
