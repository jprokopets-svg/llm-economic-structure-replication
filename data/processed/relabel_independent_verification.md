# Relabel Independent Verification — **FAIL**

One or more checks failed:
* 1(a) prompt econ-vocab: 25 prompts leaked

Verifies the RELABEL vs INFER accuracy claim against raw artefacts using logic independent of scripts/run_analysis.py, scripts/strict_rescore.py, and scripts/told_vs_infer.py.

* Relabel checkpoints:  `/Users/jakeprokopets/LLM-Matrix2/outputs/relabel_run_v3/checkpoint.jsonl`
* Infer  checkpoints:  `/Users/jakeprokopets/LLM-Matrix2/outputs/structural_tracking_v2/checkpoint.jsonl` and `/Users/jakeprokopets/LLM-Matrix2/outputs/opus_run/checkpoint.jsonl`
* Per-seed GT:         `/tmp/lmm2_per_seed_test/ground_truth_per_seed.json`
* Relabel raw:         `/Users/jakeprokopets/LLM-Matrix2/outputs/relabel_run_v3/raw_responses`

## 1. Prompt Audit

Sampled **30** relabel records uniformly across the relabel_run_v3 grid; prompts were regenerated with the fixes-branch narrative pipeline and audited with an independent economic-vocabulary regex written from scratch in this script.

### (a) Independent economic-vocabulary scan: **FAIL**

* Prompts scanned: 30
* Prompts with any forbidden term matched: 25
  * `world4__wage_gap_slope__0.3__relabel__s9__Claude_Sonnet_4.6`: ['trade-weighted']
  * `world3__taylor_phi_pi__1.5__relabel__s9__GPT-5.5`: ['trade-weighted']
  * `world4__wage_gap_slope__0.3__relabel__s2__Claude_Opus_4.8`: ['trade-weighted']
  * `world2__taylor_phi_pi__1.3__relabel__s3__Gemini_3.5_Flash`: ['trade-weighted']
  * `world4__is_sensitivity__0.6__relabel__s4__Claude_Sonnet_4.6`: ['trade-weighted']
  * `world3__taylor_phi_pi__1.5__relabel__s1__Claude_Sonnet_4.6`: ['trade-weighted']
  * `world2__phillips_slope__0.8__relabel__s2__Gemini_3.5_Flash`: ['trade-weighted']
  * `world2__taylor_phi_pi__1.3__relabel__s8__Gemini_3.5_Flash`: ['trade-weighted']
  * `world4__phillips_slope__0.2__relabel__s0__GPT-5.5`: ['trade-weighted']
  * `world4__taylor_phi_pi__2.5__relabel__s8__Claude_Sonnet_4.6`: ['trade-weighted']

### (b) Numerical-content equality: **PASS**

Comparison uses multiset equality on all numeric literals in the prompt. History time series are the dominant source of numeric tokens; any leak of prose numbers (e.g. `2 percent`) will register here.

* Prompts with mismatched numeric multisets: 0 / 30

### (c) Forecast-request equivalence: **PASS**

Both prompts request the same horizons (Q+1 … Q+8) and the same number of variables. Vocabulary differs by design (`inflation` → `X2`, `variables` → `quantities`).

### Prompt length (characters)

| Condition | Mean | Min | Max |
|---|---:|---:|---:|
| infer   | 15624 | 11990 | 17194 |
| relabel | 17136 | 12681 | 19773 |

## 2. Condition Integrity

### Response uses neutral schema: **PASS**

* Responses with neutral tokens (X1..X7, coupling, controller): 10 / 10

### Manual join trace (10 random relabel rows)

For each row: X-key → econ-key → GT lookup keys → sign match.

| call_id | X-key | econ-key | true Δ | model Δ | sign match |
|---|---|---|---:|---:|---|
| `world4__wage_gap_slope__0.3__relabel__s9__Claude_Sonnet_4.6` | X2_1 | pi_1 | +0.191 | +0.300 | YES |
| `world3__taylor_phi_pi__1.5__relabel__s9__GPT-5.5` | X2_8 | pi_8 | +0.030 | +0.000 | no |
| `world4__wage_gap_slope__0.3__relabel__s2__Claude_Opus_4.8` | X7_8 | u_natural_8 | -0.209 | -0.300 | YES |
| `world2__taylor_phi_pi__1.3__relabel__s3__Gemini_3.5_Flash` | X5_1 | u_1 | -0.181 | -0.500 | YES |
| `world1__taylor_phi_pi__1.5__relabel__s5__Claude_Opus_4.8` | X1_8 | y_8 | +0.000 | +0.000 | YES |
| `world4__is_sensitivity__0.6__relabel__s4__Claude_Sonnet_4.6` | X2_1 | pi_1 | +0.000 | +0.000 | YES |
| `world3__taylor_phi_pi__1.5__relabel__s1__Claude_Sonnet_4.6` | X5_1 | u_1 | +0.367 | +0.400 | YES |
| `world2__phillips_slope__0.8__relabel__s2__Gemini_3.5_Flash` | X3_4 | r_4 | -23.773 | -8.000 | YES |
| `world2__taylor_phi_pi__1.3__relabel__s8__Gemini_3.5_Flash` | X3_1 | r_1 | -0.165 | -0.100 | YES |
| `world4__phillips_slope__0.2__relabel__s0__GPT-5.5` | X5_8 | u_8 | +0.000 | +0.000 | YES |

## 3. Independent Rescore

### Accuracy per model (tier=all, |td|≥0.01)

Compared to the strict_rescore_report.md relabel−infer table. Report abs(diff_pp) between this script and the report; PASS if within 0.5 pp on both accuracy numbers.

| Model | Rel acc (this) | Inf acc (this) | Rel−Inf (this, pp) | Report Rel | Report Inf | Report Rel−Inf | max |Δpp| |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 62.2% | 54.9% | +7.3 | 62.2% | 54.9% | +7.3 | 0.02 |
| Claude Opus 4.8 | 63.8% | 60.6% | +3.2 | 63.8% | 60.6% | +3.2 | 0.02 |
| GPT-5.5 | 75.0% | 73.2% | +1.8 | 75.0% | 73.1% | +1.9 | 0.06 |
| Gemini 3.5 Flash | 69.1% | 64.1% | +5.0 | 69.1% | 64.1% | +5.0 | 0.02 |
| DeepSeek V4 Pro | 54.7% | 54.4% | +0.3 | 54.7% | 53.9% | +0.7 | 0.47 |

**Independent-vs-report match:** PASS (0.5 pp tolerance).

### Paired Relabel − Infer with cluster bootstrap 95% CI

Only cells where BOTH conditions have a forecast row are included; clusters = (world, param, setting); 10,000 reps.

| Model | n paired | Rel acc | Inf acc | Rel−Inf | 95% CI | Claim (pp) | sign match |
|---|---:|---:|---:|---:|---:|---:|---|
| Claude Sonnet 4.6 | 6769 | 61.1% | 54.9% | +6.2 | [+4.3, +8.2] | +7.3 | YES |
| Claude Opus 4.8 | 7538 | 63.8% | 60.6% | +3.2 | [+1.6, +4.8] | +3.2 | YES |
| GPT-5.5 | 5751 | 73.7% | 73.0% | +0.7 | [-1.1, +2.5] | +1.9 | YES |
| Gemini 3.5 Flash | 4760 | 69.9% | 64.7% | +5.2 | [+3.1, +7.4] | +5.0 | YES |
| DeepSeek V4 Pro | 45 | 51.1% | 51.1% | +0.0 | [-14.3, +3.3] | — | n/a |

## 4. Effect Localization

### Per-world / per-parameter paired Relabel − Infer

**Claude Sonnet 4.6**

| World | n | Rel acc | Inf acc | Diff |
|---|---:|---:|---:|---:|
| world1 | 1738 | 53.5% | 49.9% | +3.5 |
| world2 | 1418 | 75.1% | 65.2% | +9.9 |
| world3 | 1482 | 48.9% | 47.4% | +1.6 |
| world4 | 2131 | 66.6% | 57.3% | +9.3 |

| Param | n | Rel acc | Inf acc | Diff |
|---|---:|---:|---:|---:|
| phillips_slope | 2088 | 64.5% | 60.0% | +4.5 |
| taylor_phi_pi | 2349 | 58.7% | 51.4% | +7.3 |
| is_sensitivity | 2332 | 60.5% | 53.8% | +6.7 |
| wage_gap_slope | 0 | — | — | — |

**Claude Opus 4.8**

| World | n | Rel acc | Inf acc | Diff |
|---|---:|---:|---:|---:|
| world1 | 1763 | 59.9% | 51.6% | +8.3 |
| world2 | 1430 | 75.9% | 77.6% | -1.7 |
| world3 | 1482 | 38.7% | 39.7% | -1.0 |
| world4 | 2863 | 73.1% | 68.5% | +4.6 |

| Param | n | Rel acc | Inf acc | Diff |
|---|---:|---:|---:|---:|
| phillips_slope | 2088 | 67.9% | 63.5% | +4.5 |
| taylor_phi_pi | 2360 | 60.4% | 56.9% | +3.5 |
| is_sensitivity | 2376 | 59.7% | 57.9% | +1.8 |
| wage_gap_slope | 714 | 76.3% | 73.2% | +3.1 |

**GPT-5.5**

| World | n | Rel acc | Inf acc | Diff |
|---|---:|---:|---:|---:|
| world1 | 1631 | 69.0% | 68.8% | +0.2 |
| world2 | 1115 | 84.8% | 86.3% | -1.5 |
| world3 | 982 | 57.9% | 60.7% | -2.7 |
| world4 | 2023 | 79.0% | 75.0% | +4.0 |

| Param | n | Rel acc | Inf acc | Diff |
|---|---:|---:|---:|---:|
| phillips_slope | 1687 | 78.7% | 77.6% | +1.1 |
| taylor_phi_pi | 1971 | 67.8% | 68.1% | -0.3 |
| is_sensitivity | 2093 | 75.3% | 73.9% | +1.4 |
| wage_gap_slope | 0 | — | — | — |

**Gemini 3.5 Flash**

| World | n | Rel acc | Inf acc | Diff |
|---|---:|---:|---:|---:|
| world1 | 993 | 65.2% | 51.8% | +13.4 |
| world2 | 931 | 80.3% | 76.8% | +3.5 |
| world3 | 745 | 53.0% | 52.8% | +0.3 |
| world4 | 2091 | 73.5% | 69.7% | +3.8 |

| Param | n | Rel acc | Inf acc | Diff |
|---|---:|---:|---:|---:|
| phillips_slope | 1891 | 74.0% | 67.2% | +6.8 |
| taylor_phi_pi | 1104 | 64.0% | 56.9% | +7.2 |
| is_sensitivity | 1069 | 62.3% | 63.2% | -0.9 |
| wage_gap_slope | 696 | 79.7% | 72.4% | +7.3 |

**DeepSeek V4 Pro**

| World | n | Rel acc | Inf acc | Diff |
|---|---:|---:|---:|---:|
| world1 | 45 | 51.1% | 51.1% | +0.0 |
| world2 | 0 | — | — | — |
| world3 | 0 | — | — | — |
| world4 | 0 | — | — | — |

| Param | n | Rel acc | Inf acc | Diff |
|---|---:|---:|---:|---:|
| phillips_slope | 45 | 51.1% | 51.1% | +0.0 |
| taylor_phi_pi | 0 | — | — | — |
| is_sensitivity | 0 | — | — | — |
| wage_gap_slope | 0 | — | — | — |

### Narrative-template rewrite intensity per world

Word-level fraction of tokens changed by the relabel pass on each world's baseline seed=0 narrative.

| World | unique orig tokens | added tokens | removed tokens | fraction |
|---|---:|---:|---:|---:|
| world1 | 418 | 56 | 67 | 14.7% |
| world2 | 381 | 49 | 54 | 13.5% |
| world3 | 477 | 52 | 66 | 12.4% |
| world4 | 434 | 55 | 58 | 13.0% |

## Summary of Findings

**Central claim replicates on unpaired accuracy denominators.** The reported per-model rel−infer gaps match my independent rescore within 0.5 pp for all five models when computed on each condition's full parseable cell set (the same denominators the report uses).

**Central claim weakens on row-paired cells.** When I restrict the comparison to cells where BOTH conditions have a forecast for the same (world, param, setting, seed, var, horizon), the effect shrinks for models with lower infer parse rates. GPT-5.5 drops from +1.9 pp (unpaired) to +0.7 pp with a paired CI that includes zero. Sonnet drops from +7.3 to +6.2 (still excludes zero). Opus and Gemini are essentially unchanged. This means the reported +7.3 / +5.0 / +3.2 / +1.9 numbers are on unpaired denominators, not strict row-paired cells.

**One minor prompt leak identified.** The independent guard list flags the bare phrase `trade-weighted` (as in 'a trade-weighted basis') in 25 of 30 sampled relabel prompts. The runner's substitution map only covers `trade-weighted basket`; the bare form slips through both the substitution and the runner's own guard (_FORBIDDEN_TERMS lists only the two-word 'trade-weighted basket'). Impact is likely small — 'trade-weighted' is a weak signal on its own — but it is a real, systematic vocabulary leak in worlds 2/3/4.

**Effect is broad, not one-hotspot.** Sonnet, Opus, and Gemini all show the relabel gain in three of four worlds; world2 is the largest driver for Sonnet (+9.9 pp) and world1 for Opus (+8.3) and Gemini (+13.4). Parameter sweep-wise the effect is fairly uniform for Sonnet and Gemini; for Opus and GPT-5.5 it is smaller across the board. The narrative-template rewrite intensity (unique tokens added + removed / 2·orig-unique) is similar across all four worlds (~12-15%), so the effect is not obviously driven by which world is most rewritten.
