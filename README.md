# Testing-Bias-Simulator
Shows how differential care-seeking and misclassification bias VE under TND vs TTE.

## Quickstart (GitHub-only, no local setup)

- Go to **Actions \u2192 Build figures** and click **Run workflow**.
  After it completes, open the `figures/` directory to see the bias heatmaps, misclassification sweeps, and results table:
  • `bias_heatmap_TND.png` and `bias_heatmap_TTE.png`
  • `misclass_sweep_sens.png` and `misclass_sweep_spec.png`
  • `bias_sweep_results.csv`
- Go to **Actions \u2192 Build brief PDF** and click **Run workflow** to generate `reports/brief.pdf`.
## Scripts

- `bias_simulator/simulate_bias.py` — core logic (population, care-seeking, test simulation, estimators).
- `bias_simulator/make_plots.py` — generates heatmaps and saves them in `figures/`.
- - bias_simulator/misclass_plots.py' — generates misclassification sweeps (sensitivity and specificity) and saves heatmaps in 'figures/'.

## Outputs

The workflow saves:
- `- `figures/bias_heatmap_TND.png` and `figures/bias_heatmap_TTE.png`
- `figures/misclass_sweep_sens.png` and `figures/misclass_sweep_spec.png`
- `figures/bias_sweep_results.csv`
- `artifacts/validation_summary.json` (sanity and stability metrics)

Note: random seeds are fixed so results are deterministic and reproducible across runs.
Note: random seeds are fixed so results are deterministic and reproducible across runs.
