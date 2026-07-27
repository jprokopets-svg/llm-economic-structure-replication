# Protocol Deviations

The preregistration (commit `a30b17b`, 2026-06-20) specified:
- 4 worlds, 4 parameters, 5 settings each, 3 horizons
- 2 primary conditions (TOLD, INFER) plus RELABEL
- 4 models (Claude Sonnet 4.6, GPT-5.5, Gemini 3.5 Flash, DeepSeek V4 Pro)
- Pass@1, clustered bootstrap (10,000; cluster = world × parameter setting)
- Success criteria: floor > 50% correct sign; tracks β > 0 with CI excluding zero

## Eleven Deviations

| # | Deviation | Date | Description |
|---|-----------|------|-------------|
| 1 | **Opus 4.8 added** | 2026-06-27 | Flagship robustness check; not in preregistration |
| 2 | **W3 Phillips 0.7/0.9 excluded** | 2026-06-26 | Dynamic instability at extreme settings |
| 3 | **RELABEL redesign** | 2026-07-10 | Complete redesign with hand-written static templates and whitelist enforcement |
| 4 | **RELABEL scope expansion** | 2026-07-10 | Extended from W1–W2/P1 to all worlds × all parameters |
| 5 | **Per-seed ground truth** | 2026-07-10 | Correct scoring: match history seed to model's forecast seed |
| 6 | **Stability tiers introduced** | 2026-07-10 | Added filter for ground-truth direction consistency across seeds |
| 7 | **GPT-5.5 RELABEL adjustment** | 2026-07-10 | Final 508 RELABEL cells used `reasoning.effort=minimal` |
| 8 | **DeepSeek exclusion from RELABEL** | 2026-07-10 | Halted by parse-rate guard (17/58 parseable) |
| 9 | **Fork experiment added** | 2026-07-14 | Three-arm design (placebo/change+equation/baseline) |
| 10 | **ARM2-ARM1 contrast** | 2026-07-15 | Placebo-controlled contrast motivated by placebo check failure |
| 11 | **Gemini temperature robustness** | 2026-07-19 | 10% subsample at temperature 0.7 |

## RELABEL Implementation History

- **v0 (original, 2026-06-20):** Used identical prompts to INFER (contaminated).
  All v0 results excluded from the paper.
- **v3 (intermediate, 2026-07-10):** Used regex substitution for vocabulary.
  Terminology leaks were found after the run completed. Not used in the paper.
- **v4 (final, 2026-07-10):** Hand-written static templates with whitelist enforcement.
  This is the sole valid RELABEL implementation. Covers Sonnet 4.6, GPT-5.5,
  and Gemini 3.5 Flash. Opus 4.8 was deferred; DeepSeek V4 Pro was excluded.

## Fork Experiment Status

The fork experiment and its placebo contrast were designed and executed after
the preregistration date. It is identified as a post-preregistration follow-up
in the paper.
