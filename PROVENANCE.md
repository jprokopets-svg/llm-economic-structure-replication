# File Provenance

Every file in this repository was copied from either the `LLM-Matrix2` or
`LLM-Matrix` repositories. The table below lists the original source for each file.

## Source Repositories

- **LLM-Matrix2:** `../LLM-Matrix2/` (commit 19164c4)
- **LLM-Matrix:** `../LLM-Matrix/` (v1 simulator repository)

## File Map

| Destination path | Source repo | Source path | Modified? |
|---|---|---|---|
| `config/ball_baseline.yaml` | LLM-Matrix | `config/ball_baseline.yaml` | No |
| `config/world2_closed_economy.yaml` | LLM-Matrix | `config/world2_closed_economy.yaml` | No |
| `config/world3_emerging_market.yaml` | LLM-Matrix | `config/world3_emerging_market.yaml` | No |
| `config/world4_labor_hysteresis.yaml` | LLM-Matrix | `config/world4_labor_hysteresis.yaml` | No |
| `config/models.yaml` | LLM-Matrix2 | `config/models.yaml` | No |
| `config/worlds.yaml` | LLM-Matrix2 | `config/worlds.yaml` | No |
| `config/parameter_grids.yaml` | LLM-Matrix2 | *(new file)* | New |
| `src/lmm2/scoring.py` | LLM-Matrix2 | `src/lmm2/scoring.py` | No |
| `src/lmm2/baselines.py` | LLM-Matrix2 | `src/lmm2/baselines.py` | No |
| `src/lmm2/narrative.py` | LLM-Matrix2 | `src/lmm2/narrative.py` | No |
| `src/lmm2/config_loader.py` | LLM-Matrix2 | `src/lmm2/config_loader.py` | No |
| `src/lmm2/relabel_template.py` | LLM-Matrix2 (worktree) | `.claude/worktrees/upbeat-nash-3999e8/src/lmm2/relabel_template.py` | No |
| `src/lmm2/simulator.py` | LLM-Matrix | `src/llmmatrix/simulator.py` | Import paths adjusted |
| `src/lmm2/world2_simulator.py` | LLM-Matrix | `src/llmmatrix/world2_simulator.py` | Import paths adjusted |
| `src/lmm2/world3_simulator.py` | LLM-Matrix | `src/llmmatrix/world3_simulator.py` | Import paths adjusted |
| `src/lmm2/world4_simulator.py` | LLM-Matrix | `src/llmmatrix/world4_simulator.py` | Import paths adjusted |
| `src/lmm2/shocks.py` | LLM-Matrix | `src/llmmatrix/shocks.py` | Import paths adjusted |
| `src/lmm2/monte_carlo.py` | LLM-Matrix | `src/llmmatrix/monte_carlo.py` | Import paths adjusted |
| `src/lmm2/narrative.py` (v1) | LLM-Matrix | `src/llmmatrix/narrative.py` | Import paths adjusted |
| `scripts/run_structural_tracking.py` | LLM-Matrix2 | `scripts/run_structural_tracking.py` | Import paths adjusted |
| `scripts/run_fork.py` | LLM-Matrix2 | `scripts/run_fork.py` | Import paths adjusted |
| `scripts/run_analysis.py` | LLM-Matrix2 | `scripts/run_analysis.py` | No |
| `scripts/run_relabel_v2.py` | LLM-Matrix2 (branch) | `fixes/gt-seeds-baserate-parse:scripts/run_relabel_v2.py` | No |
| `scripts/score_relabel_v2.py` | LLM-Matrix2 (branch) | `fixes/gt-seeds-baserate-parse:scripts/score_relabel_v2.py` | No |
| `scripts/verify_relabel_v2.py` | LLM-Matrix2 (branch) | `fixes/gt-seeds-baserate-parse:scripts/verify_relabel_v2.py` | No |
| `scripts/gpt55_fix_spot.py` | LLM-Matrix2 (branch) | `fixes/gt-seeds-baserate-parse:scripts/gpt55_fix_spot.py` | No |
| `scripts/relabel_minus_infer.py` | LLM-Matrix2 (branch) | `fixes/gt-seeds-baserate-parse:scripts/relabel_minus_infer.py` | No |
| `scripts/score_fork.py` | LLM-Matrix2 | `scripts/score_fork.py` | No |
| `scripts/score_fork_dual.py` | LLM-Matrix2 | `scripts/score_fork_dual.py` | No |
| `scripts/verify_fork.py` | LLM-Matrix2 | `scripts/verify_fork.py` | No |
| `scripts/compute_stats.py` | LLM-Matrix2 | `scripts/compute_stats.py` | No |
| `scripts/compute_simple_baselines.py` | LLM-Matrix2 | `scripts/compute_simple_baselines.py` | No |
| `scripts/zero_response_reanalysis.py` | LLM-Matrix2 | `scripts/zero_response_reanalysis.py` | No |
| `scripts/add_var_and_bootstrap.py` | LLM-Matrix2 | `scripts/add_var_and_bootstrap.py` | No |
| `scripts/generate_figures.py` | LLM-Matrix2 | `scripts/generate_figures.py` | No |
| `tests/test_scoring.py` | LLM-Matrix2 | `tests/test_scoring.py` | No |
| `tests/test_baselines.py` | LLM-Matrix2 | `tests/test_baselines.py` | No |
| `tests/test_config_loader.py` | LLM-Matrix2 | `tests/test_config_loader.py` | No |
| `docs/preregistration/PREREGISTRATION.md` | LLM-Matrix2 | `docs/design/PREREGISTRATION.md` | No |
| `docs/prompts/FORK_PROMPT_TEMPLATES.md` | LLM-Matrix2 | `docs/FORK_PROMPT_TEMPLATES.md` | No |
| `docs/prompts/FORK_EXPERIMENT_DESIGN.md` | LLM-Matrix2 | `docs/FORK_EXPERIMENT_DESIGN.md` | No |

**Import path changes applied to:** All simulator files (`simulator.py`,
`world2_simulator.py`, `world3_simulator.py`, `world4_simulator.py`,
`shocks.py`, `monte_carlo.py`, `narrative.py`) had their package imports
changed from `llmmatrix.*` to `lmm2.*` to match the new repository structure.
No numerical logic was altered.
