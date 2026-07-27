# Do Large Language Models Track Economic Structure? Evidence from Synthetic Economies

**Author:** Jake Prokopets

Replication package for the Economics Letters manuscript.

## Project Description

This repository contains code, configurations, and processed data for testing whether
frontier large language models adjust macroeconomic forecasts when structural parameters
change in four synthetic New Keynesian economies. The paper reports two experiments:

1. **Fork experiment (primary):** Models forecast from identical simulated histories after
   either a placebo announcement or a parameter change stated with its governing equation.
   Ground truth is computed by forking the simulator with both parameter values and
   comparing average forward paths under common random numbers.

2. **Grid experiment (exploratory):** Models forecast from histories generated under different
   parameter values, testing whether they detect and use structure present in the data.

## Repository Scope

This package includes everything needed to:
- Understand the experimental design
- Reproduce all reported tables and figures from saved model responses (offline)
- Rerun simulations and scoring without calling model APIs
- Rerun model queries with your own credentials (online, may incur costs)

## Directory Map

```
├── README.md                        This file
├── LICENSE                          License information
├── CITATION.cff                     Citation metadata
├── pyproject.toml                   Python build/installation metadata
├── requirements.txt                 Python dependencies
├── DATA.md                          Data description and provenance
├── REPRODUCIBILITY.md               Reproduction instructions
├── PROVENANCE.md                    Source file provenance
├── config/
│   ├── ball_baseline.yaml           World 1 configuration (Ball 1999 open economy)
│   ├── world2_closed_economy.yaml   World 2 configuration
│   ├── world3_emerging_market.yaml  World 3 configuration
│   ├── world4_labor_hysteresis.yaml World 4 configuration
│   ├── models.yaml                  Model definitions
│   ├── worlds.yaml                  World definitions
│   └── parameter_grids.yaml         Authoritative parameter value grids
├── src/
│   └── lmm2/
│       ├── scoring.py               Directional accuracy, regression, bootstrap
│       ├── baselines.py             Naive, AR(1), VAR baselines
│       ├── narrative.py             Prompt construction (TOLD, INFER)
│       ├── relabel_template.py      RELABEL v4 static templates with whitelist
│       ├── config_loader.py         YAML config loader
│       ├── simulator.py             World 1 simulator (Ball 1999 open economy)
│       ├── world2_simulator.py      World 2 simulator (closed economy)
│       ├── world3_simulator.py      World 3 simulator (emerging market)
│       ├── world4_simulator.py      World 4 simulator (labor hysteresis)
│       ├── shocks.py                Shock-generation utilities
│       ├── monte_carlo.py           Monte Carlo forward simulation
│       └── model_caller.py          Model API caller wrapper
├── scripts/
│   ├── run_structural_tracking.py   Structural grid experiment runner
│   ├── run_fork.py                  Fork experiment runner
│   ├── run_relabel_v2.py            RELABEL v4 experiment runner
│   ├── run_analysis.py              Preregistered analysis pipeline
│   ├── score_fork.py                Fork directional scoring
│   ├── score_fork_dual.py           Dual-contrast fork scoring
│   ├── score_relabel_v2.py          RELABEL v4 scoring
│   ├── verify_fork.py               Independent fork verification
│   ├── verify_relabel_v2.py         Independent RELABEL verification
│   ├── compute_stats.py             Sign-and-slope statistics
│   ├── compute_simple_baselines.py  Simple baseline computation
│   ├── zero_response_reanalysis.py  Zero-response and tie-adjusted analysis
│   ├── add_var_and_bootstrap.py     VAR baseline + clustered bootstrap
│   ├── relabel_minus_infer.py       Paired RELABEL-minus-INFER CIs
│   ├── gpt55_fix_spot.py           GPT-5.5 RELABEL spot-check
│   └── generate_figures.py          Manuscript figure generation
├── tests/
│   ├── test_scoring.py              Scoring unit tests
│   ├── test_baselines.py            Baseline unit tests
│   └── test_config_loader.py        Config loader unit tests
├── data/
│   ├── processed/                   Analysis-ready datasets and reports
│   └── README.md                    Data description
├── outputs/
│   └── final/                       Final output reports
├── docs/
│   ├── preregistration/
│   │   └── PREREGISTRATION.md       Original preregistration
│   ├── prompts/
│   │   ├── FORK_PROMPT_TEMPLATES.md Fork experiment prompt templates
│   │   └── FORK_EXPERIMENT_DESIGN.md Fork design document
│   └── deviations.md                Protocol deviations
├── release/
│   ├── manifest.csv                 Complete file manifest
│   └── checksums.sha256             SHA-256 checksums
└── .gitignore                       Git ignore rules
```

## Installation

### Prerequisites
- Python 3.11 or 3.12
- pip

### Setup
```bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
# venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Python Version

Tested on Python 3.11 and 3.12.

## Offline Reproduction (Default)

To reproduce the main reported tables from saved processed data without any API calls:

```bash
python scripts/verify_replication.py
```

This will:
1. Load saved processed data
2. Validate row counts and schemas
3. Regenerate headline statistics
4. Compare against checked-in reference values
5. Exit nonzero if any value differs beyond documented tolerance

To regenerate figures:
```bash
python scripts/generate_figures.py
```

## Full Experimental Reproduction (Requires API Credentials)

**WARNING:** Rerunning model queries requires your own API keys and may incur
substantial costs (the original experiments cost approximately $390 in API fees).

1. Copy `.env.example` to `.env` and fill in your API keys
2. Run the structural grid experiment:
   ```
   python scripts/run_structural_tracking.py
   ```
3. Run the fork experiment:
   ```
   python scripts/run_fork.py
   ```
4. Run the RELABEL v4 experiment:
   ```
   python scripts/run_relabel_v2.py
   ```
5. Run the analysis pipeline:
   ```
   python scripts/run_analysis.py
   python scripts/score_fork_dual.py
   python scripts/score_relabel_v2.py
   python scripts/zero_response_reanalysis.py
   ```

## Credentials

**No API credentials are included in this repository.** You must provide your own
OpenRouter, Google AI, or direct provider API keys to rerun model queries. See
`.env.example` for the required environment variable names.

## Data DOI

Processed data are archived at: https://doi.org/10.5281/zenodo.21616867

## Citation

See CITATION.cff for bibliographic metadata.
