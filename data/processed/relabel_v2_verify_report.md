# Independent rescore of v2 relabel run

Reference: `/Users/jakeprokopets/LLM-Matrix2/outputs/analysis/relabel_v2_final.json` (score_relabel_v2.py output).
Tolerance: **0.5 pp** absolute difference on accuracy or paired difference.

## Independent numbers (this script's computation)

| Tier | Model | Relabel acc | n (r) | Infer acc | n (i) | Paired diff [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| all | Claude Sonnet 4.6 | 57.3% | 7538 | 54.9% | 6757 | +2.4 pp [-0.6, +5.4] |
| all | GPT-5.5 | 75.4% | 7515 | 73.1% | 6043 | +2.3 pp [+0.6, +4.1] |
| all | Gemini 3.5 Flash | 66.5% | 7538 | 64.1% | 4991 | +2.4 pp [-0.6, +5.4] |
| moderate | Claude Sonnet 4.6 | 57.8% | 5848 | 55.4% | 5375 | +2.4 pp [-0.8, +5.5] |
| moderate | GPT-5.5 | 76.9% | 5834 | 74.4% | 4814 | +2.5 pp [+0.6, +4.4] |
| moderate | Gemini 3.5 Flash | 67.2% | 5848 | 65.3% | 3839 | +1.9 pp [-1.4, +5.1] |
| strict | Claude Sonnet 4.6 | 49.3% | 708 | 50.3% | 686 | -1.0 pp [-8.8, +7.4] |
| strict | GPT-5.5 | 73.0% | 705 | 69.2% | 614 | +3.8 pp [-1.7, +10.4] |
| strict | Gemini 3.5 Flash | 60.3% | 708 | 55.8% | 362 | +4.5 pp [-4.4, +14.8] |

## Agreement with score_relabel_v2

**No deviations above 0.5 pp.** Independent rescore agrees with score_relabel_v2 within tolerance for every model at every tier.
