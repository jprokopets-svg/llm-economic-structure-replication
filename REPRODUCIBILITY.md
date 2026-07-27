# Reproducibility

## Offline Reproduction (Default Path)

The offline path regenerates all reported tables, confidence intervals, and
statistics from saved processed data without calling any model API.

### Prerequisites
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run verification
```bash
python scripts/verify_replication.py
```

This script will:
1. Load all processed data from `data/processed/`
2. Validate schema and row counts against expected values
3. Regenerate headline statistics (fork strict accuracy, grid accuracy,
   permutation baselines, tie-adjusted accuracy, regression betas)
4. Compare against checked-in reference values
5. Exit 0 if all values match within tolerance, nonzero otherwise

### Regenerate figures
```bash
python scripts/generate_figures.py
```

Outputs: `paper/figures/fig_main_accuracy.pdf`, `fig2_fork_accuracy.pdf`,
`fig3_oldgrid.pdf`

### Regenerate bootstrap CIs
```bash
python scripts/score_fork_dual.py
python scripts/compute_stats.py data/processed/structural_results.json
```

## Full Experimental Reproduction

**Warning:** This path requires API credentials and may incur substantial costs.

### 1. Obtain API credentials
Copy `.env.example` to `.env` and populate with keys for:
- OpenRouter (for GPT-5.5, Claude models, DeepSeek)
- Google AI (for Gemini 3.5 Flash)

### 2. Structural Grid Experiment
```bash
# Generate histories and ground truth, then query models
python scripts/run_structural_tracking.py

# Run the preregistered analysis
python scripts/run_analysis.py
```

### 3. Fork Experiment
```bash
# Generate histories, ground truth, and query all three arms
python scripts/run_fork.py

# Score both contrasts
python scripts/score_fork_dual.py

# Independent verification
python scripts/verify_fork.py
```

### 4. RELABEL v4 Experiment
```bash
# Run static-template RELABEL condition
python scripts/run_relabel_v2.py

# Score and verify
python scripts/score_relabel_v2.py
python scripts/verify_relabel_v2.py
```

### 5. Supplementary Analyses
```bash
# Zero-response and tie-adjusted reanalysis
python scripts/zero_response_reanalysis.py

# Simple baselines
python scripts/compute_simple_baselines.py

# VAR benchmark
python scripts/add_var_and_bootstrap.py
```

### Models and Estimated Costs

| Model | Provider | Estimated cost (full run) |
|-------|----------|---------------------------|
| GPT-5.5 | OpenRouter | ~$85 |
| Claude Sonnet 4.6 | OpenRouter | ~$60 |
| Claude Opus 4.8 | OpenRouter | ~$165 |
| Gemini 3.5 Flash | Google AI | ~$60 |
| DeepSeek V4 Pro | OpenRouter | ~$20 |

Total estimated cost for all experiments: ~$390 (as of July 2026).
Current pricing may differ.

### Seeds

- History generation: 10 seeds (0–9) per cell
- Ground-truth MC: seed 42 for history, seeds 1000–1999 for forward paths (1,000 paths)
- Fork ground truth: seeds 1000–1099 (100 paths, CRN)
- Bootstrap: seed 42 for cluster resampling
- Permutation: seed 42 for shuffling

### Temperature

All model queries used temperature 0.
