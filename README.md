# Testing-Bias-Simulator
Shows how differential care-seeking and misclassification bias VE under TND vs TTE.

## Quickstart (GitHub-only, no local setup)

- Go to **Actions → Build figures (bias simulator) → Run workflow**.
- After it completes, open `figures/` to see:
  - `bias_heatmap_TND.png`
  - `bias_heatmap_TTE.png`

## Scripts

- `bias_simulator/simulate_bias.py` — core logic (population, care-seeking, test simulation, estimators).
- `bias_simulator/make_plots.py` — generates heatmaps and saves them in `figures/`.

## Outputs

The workflow saves:
- `figures/bias_heatmap_TND.png`
- `figures/bias_heatmap_TTE.png`
- `figures/bias_sweep_results.csv` (full table for reproducibility)
