# Phase 1 Analyses (Free — Existing Data)

**Date:** 2026-07-14

## 1. Cell eligibility: |true_delta| > 2×MC_SE

MC_SE estimated as cross-seed standard error: sd(true_delta across 10 seeds) / sqrt(10). This captures both MC sampling noise and history-dependence.

| Criterion | Cells | % of total |
|-----------|------:|----------:|
| All cells | 846 | 100.0% |
| |true_delta| > 2×MC_SE (proposed) | 27 | 3.2% |
| Old strict (majority_frac ≥ 0.8) | 156 | 18.4% |
| Old moderate (majority_frac ≥ 0.6) | 675 | 79.8% |
| All seeds agree on sign | 84 | 9.9% |

## 2. Economic heterogeneity

**762** of 846 cells (90.1%) have at least one seed with opposite GT direction. These are **state-dependent cells** where the structural effect genuinely depends on the specific history realization, not just MC noise.

Of the 762 heterogeneous cells:
- Near-split (majority_frac < 0.6): 171
- Moderate agreement (0.6–0.8): 519
- Strong agreement (≥0.8): 72

## 3. Simple baselines on GT directions

GT direction balance: 50.2% positive (n=7538)

| Baseline | Accuracy | Note |
|----------|--------:|----|
| Always predict positive | 50.2% | Majority-class baseline |
| Sign-frequency (per param×horizon) | 51.7% | Best achievable by memorizing direction distributions |
| Random walk (Δ=0) | 0.0% | Always incorrect by scoring rule |

## 4. Model accuracy under 2×MC_SE criterion vs old tiers

| Model | Condition | All | 2×MC_SE | Old moderate | Old strict |
|-------|-----------|---:|---:|---:|---:|
| Claude Sonnet 4.6 | told | 54.7% (n=6743) | 51.7% (n=263) | 55.1% (n=5370) | 52.2% (n=691) |
| Claude Sonnet 4.6 | infer | 54.9% (n=6757) | 46.1% (n=267) | 55.4% (n=5375) | 50.3% (n=686) |
| Claude Opus 4.8 | told | nan% (n=0) | nan% (n=0) | nan% (n=0) | nan% (n=0) |
| Claude Opus 4.8 | infer | nan% (n=0) | nan% (n=0) | nan% (n=0) | nan% (n=0) |
| GPT-5.5 | told | 73.1% (n=6164) | 62.1% (n=264) | 73.6% (n=4894) | 66.4% (n=628) |
| GPT-5.5 | infer | 73.1% (n=6043) | 59.8% (n=246) | 74.4% (n=4814) | 69.2% (n=614) |
| Gemini 3.5 Flash | told | 60.6% (n=5306) | 48.3% (n=145) | 61.2% (n=4098) | 51.9% (n=414) |
| Gemini 3.5 Flash | infer | 64.1% (n=4991) | 45.4% (n=119) | 65.3% (n=3839) | 55.8% (n=362) |
| DeepSeek V4 Pro | told | 54.6% (n=1499) | 47.2% (n=53) | 55.3% (n=1152) | 48.2% (n=139) |
| DeepSeek V4 Pro | infer | 53.9% (n=1387) | 41.2% (n=51) | 56.2% (n=1105) | 52.7% (n=129) |

## 5. TOST equivalence tests: told vs infer

**Preregistered margin:** ±3 pp. Equivalence established if the 90% CI for (told − infer) accuracy lies entirely within [−0.03, +0.03].

| Model | n | Told−Infer | 90% CI | 95% CI | Equivalence? |
|-------|---:|----------:|-------:|-------:|:------------:|
| Claude Sonnet 4.6 | 6676 | -0.1 pp | [-1.1, +0.8] | [-1.2, +1.0] | **YES** |
| Claude Opus 4.8 | 7538 | -1.1 pp | [-1.9, -0.3] | [-2.0, -0.1] | **YES** |
| GPT-5.5 | 5635 | +0.2 pp | [-1.1, +1.4] | [-1.4, +1.6] | **YES** |
| Gemini 3.5 Flash | 4897 | -3.4 pp | [-5.4, -1.4] | [-5.7, -1.0] | No |
| DeepSeek V4 Pro | 338 | +1.2 pp | [-3.4, +6.6] | [-4.1, +7.9] | No |

*Note: Opus row added 2026-07-17. Computed from per-seed GT (regenerated, 1000 MC paths) + Opus checkpoint (outputs/opus\_run/checkpoint.jsonl). Point estimate and 95% CI match strict\_rescore\_report.md line 90 within 0.04 pp.*

## 6. Disaggregation: world × parameter × horizon (infer)

| Model | World | Param | h=1 | h=4 | h=8 |
|-------|-------|-------|---:|---:|---:|
| GPT-5.5 | world1 | is_sensitivity | 79% (195) | 70% (197) | 60% (200) |
| GPT-5.5 | world1 | phillips_slope | 90% (185) | 73% (183) | 71% (182) |
| GPT-5.5 | world1 | taylor_phi_pi | 67% (191) | 61% (193) | 51% (193) |
| GPT-5.5 | world2 | is_sensitivity | 95% (156) | 90% (154) | 88% (153) |
| GPT-5.5 | world2 | phillips_slope | 98% (120) | 79% (120) | 87% (120) |
| GPT-5.5 | world2 | taylor_phi_pi | 93% (131) | 76% (131) | 74% (129) |
| GPT-5.5 | world3 | is_sensitivity | 69% (138) | 72% (138) | 48% (140) |
| GPT-5.5 | world3 | phillips_slope | 79% (80) | 69% (80) | 41% (79) |
| GPT-5.5 | world3 | taylor_phi_pi | 61% (140) | 57% (134) | 49% (140) |
| GPT-5.5 | world4 | is_sensitivity | 90% (238) | 68% (239) | 64% (239) |
| GPT-5.5 | world4 | phillips_slope | 88% (219) | 74% (220) | 66% (222) |
| GPT-5.5 | world4 | taylor_phi_pi | 87% (221) | 73% (221) | 67% (222) |
| GPT-5.5 | world4 | wage_gap_slope | — | — | — |
| Claude Opus 4.8 | world1 | is_sensitivity | 77% (195) | 44% (197) | 26% (200) |
| Claude Opus 4.8 | world1 | phillips_slope | 84% (200) | 48% (198) | 44% (196) |
| Claude Opus 4.8 | world1 | taylor_phi_pi | 70% (191) | 43% (193) | 27% (193) |
| Claude Opus 4.8 | world2 | is_sensitivity | 91% (160) | 80% (158) | 62% (157) |
| Claude Opus 4.8 | world2 | phillips_slope | 94% (160) | 73% (160) | 69% (160) |
| Claude Opus 4.8 | world2 | taylor_phi_pi | 88% (159) | 67% (159) | 75% (157) |
| Claude Opus 4.8 | world3 | is_sensitivity | 59% (198) | 24% (196) | 38% (199) |
| Claude Opus 4.8 | world3 | phillips_slope | 63% (100) | 31% (100) | 34% (99) |
| Claude Opus 4.8 | world3 | taylor_phi_pi | 58% (200) | 21% (191) | 32% (199) |
| Claude Opus 4.8 | world4 | is_sensitivity | 88% (238) | 62% (239) | 50% (239) |
| Claude Opus 4.8 | world4 | phillips_slope | 91% (237) | 61% (238) | 45% (240) |
| Claude Opus 4.8 | world4 | taylor_phi_pi | 90% (239) | 64% (239) | 50% (240) |
| Claude Opus 4.8 | world4 | wage_gap_slope | 94% (236) | 66% (239) | 60% (239) |
| Claude Sonnet 4.6 | world1 | is_sensitivity | 77% (190) | 35% (193) | 32% (195) |
| Claude Sonnet 4.6 | world1 | phillips_slope | 83% (200) | 43% (198) | 41% (196) |
| Claude Sonnet 4.6 | world1 | taylor_phi_pi | 68% (186) | 35% (189) | 35% (191) |
| Claude Sonnet 4.6 | world2 | is_sensitivity | 88% (156) | 54% (154) | 50% (153) |
| Claude Sonnet 4.6 | world2 | phillips_slope | 93% (156) | 63% (156) | 59% (156) |
| Claude Sonnet 4.6 | world2 | taylor_phi_pi | 89% (159) | 51% (159) | 39% (157) |
| Claude Sonnet 4.6 | world3 | is_sensitivity | 75% (198) | 36% (196) | 37% (199) |
| Claude Sonnet 4.6 | world3 | phillips_slope | 78% (100) | 39% (100) | 31% (99) |
| Claude Sonnet 4.6 | world3 | taylor_phi_pi | 62% (200) | 32% (191) | 38% (199) |
| Claude Sonnet 4.6 | world4 | is_sensitivity | 87% (232) | 51% (233) | 29% (233) |
| Claude Sonnet 4.6 | world4 | phillips_slope | 91% (237) | 55% (238) | 35% (240) |
| Claude Sonnet 4.6 | world4 | taylor_phi_pi | 87% (239) | 54% (239) | 28% (240) |
| Claude Sonnet 4.6 | world4 | wage_gap_slope | — | — | — |
| Gemini 3.5 Flash | world1 | is_sensitivity | — | — | — |
| Gemini 3.5 Flash | world1 | phillips_slope | 86% (185) | 50% (183) | 44% (181) |
| Gemini 3.5 Flash | world1 | taylor_phi_pi | 69% (191) | 36% (193) | 27% (193) |
| Gemini 3.5 Flash | world2 | is_sensitivity | — | — | — |
| Gemini 3.5 Flash | world2 | phillips_slope | 96% (160) | 70% (160) | 74% (160) |
| Gemini 3.5 Flash | world2 | taylor_phi_pi | 88% (155) | 63% (155) | 69% (153) |
| Gemini 3.5 Flash | world3 | is_sensitivity | 73% (133) | 51% (132) | 42% (134) |
| Gemini 3.5 Flash | world3 | phillips_slope | 74% (80) | 44% (80) | 28% (80) |
| Gemini 3.5 Flash | world3 | taylor_phi_pi | 68% (65) | 45% (62) | 40% (65) |
| Gemini 3.5 Flash | world4 | is_sensitivity | 89% (232) | 63% (233) | 50% (233) |
| Gemini 3.5 Flash | world4 | phillips_slope | 93% (231) | 66% (232) | 50% (234) |
| Gemini 3.5 Flash | world4 | taylor_phi_pi | — | — | — |
| Gemini 3.5 Flash | world4 | wage_gap_slope | 93% (230) | 64% (233) | 60% (233) |
