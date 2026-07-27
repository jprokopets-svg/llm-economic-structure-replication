# Data Description

## GitHub Files

The following data files are included in this repository under `data/processed/`:

| File | Description | Source |
|------|-------------|--------|
| `fork_ground_truth.json` | Fork experiment ground truth (100 MC paths, CRN) | output of run_fork.py |
| `fork_scoring_dual_report.md` | Primary fork results (ARM2-ARM1 and ARM2-ARM3) | output of score_fork_dual.py |
| `fork_scoring_report.md` | Single-contrast fork scoring report | output of score_fork.py |
| `fork_verification_report.md` | Independent verification of fork scoring | output of verify_fork.py |
| `prefork_trends.json` | Pre-fork trend data for conflict classification | output of run_fork.py |
| `strict_rescore_report.md` | Grid experiment: per-seed GT scoring with tiers, permutation baselines, RELABEL v4 | output of score_relabel_v2.py |
| `zero_response_report.md` | Zero-response reanalysis, tie-adjusted accuracy, conditional-on-movement | output of zero_response_reanalysis.py |
| `three_condition.json` | TOLD/INFER/RELABEL accuracy with CIs (machine-readable) | output of score_relabel_v2.py |
| `relabel_v2_final.json` | Final v4 RELABEL results (machine-readable) | output of score_relabel_v2.py |
| `relabel_independent_verification.md` | Independent verification of RELABEL v4 | output of verify_relabel_v2.py |
| `relabel_v2_verify_report.md` | RELABEL v2 verification report | output of verify_relabel_v2.py |
| `phase1_report.md` | Phase 1 analysis: TOST, told-infer equivalence | output of run_analysis.py |
| `run_stats.json` | Fork experiment run statistics (parse rates, spend) | from fork_run/ |
| `structural_run_stats.json` | Structural grid run statistics | from structural_tracking_v2/ |

## Fork Experiment: Observation Units

- **2,760 API calls** (690 per model × 4 models, minus 2 Gemini errors)
- **7,620 candidate variable–horizon pairs** before eligibility filtering
- **5,893 eligible contrasts** after applying |true_delta| > 2×MC_SE (77.3%)
- Each eligible cell pairs ARM2 (change + equation) with ARM1 (placebo)

## Structural Grid Experiment: Observation Units

- **~5,600 unique call IDs** after deduplication
- **~672 testable cells per model per condition** (after exclusions)
- **102 regression family** (5 models × 4 parameters × 2–3 conditions × 3 horizons,
  minus 15 invalid v1 RELABEL regressions and non-monotonic cells)

## RELABEL

**Only the final v4 static-template RELABEL results are valid.**

The valid RELABEL data cover three models:
- Claude Sonnet 4.6 (n = 7,538 eligible pairs)
- GPT-5.5 (n = 7,515 eligible pairs)
- Gemini 3.5 Flash (n = 7,538 eligible pairs)

Claude Opus 4.8 was not run in v4 RELABEL. DeepSeek V4 Pro was excluded due to
parse-rate failure. Earlier RELABEL implementations (v0: duplicated INFER prompts;
v3: regex substitution with terminology leaks) are not included in this package.

## Zenodo Deposit

Large raw response files that exceed GitHub size guidelines are archived at:
[Zenodo DOI placeholder]

These include:
- Fork experiment raw model responses (~3.7 MB checkpoint + individual JSON files)
- Structural grid raw model responses (individual JSON files)
- RELABEL v4 raw model responses (~2.5 MB checkpoint + individual JSON files)

## Licensing

The processed summary data in `data/processed/` and saved model responses are
provided under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**
license. You may copy, redistribute, and adapt the data for any purpose,
provided you give appropriate credit to the original author.

Model outputs are subject to the terms of service of the respective API
providers (OpenAI, Anthropic, Google, DeepSeek). Consult each provider's terms
before redistributing model outputs.

## Units

All economic variables are in percentage points:
- Output gap (y): percent deviation from potential
- Inflation (pi): annualized percent
- Policy rate (r): annualized percent
- Exchange rate (e): percent deviation from steady state
- Unemployment (u): percent
- Wage inflation (w): annualized percent
- Natural rate of unemployment (u_natural): percent
