"""
Misclassification sweeps for Testing-Bias Simulator.

What this file does:
- Sweeps over test sensitivity and specificity separately.
- For each combination, computes observed VE (TND) at different care-seeking differentials.
- Saves two heatmaps in figures/: misclass_sweep_sens.png and misclass_sweep_spec.png.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from bias_simulator.simulate_bias import (
    generate_population,
    simulate_observation_process,
    estimate_TND,
)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _heatmap(xs, ys, M, title: str, outfile: str, xlabel: str):
    plt.figure()
    plt.imshow(
        M,
        aspect="auto",
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    )
    plt.colorbar(label="Observed VE (TND)")
    plt.xlabel(xlabel)
    plt.ylabel("Δ(seeking) (unvaccinated − vaccinated)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

def sweep_over_sensitivity(
    sens_vals=np.linspace(0.75, 0.99, 11),
    delta_seek_vals=np.linspace(-0.4, 0.4, 17),
    base_prev=0.10,
    true_VE=0.5,
    spec_fixed=0.98,
    N=80_000,
    seed=7,
):
    grid = np.zeros((len(delta_seek_vals), len(sens_vals)))
    for i, ds in enumerate(delta_seek_vals):
        for j, s in enumerate(sens_vals):
            pop = generate_population(N=N, base_prev=base_prev, true_VE=true_VE, seed=seed+i+j)
            obs = simulate_observation_process(pop, delta_seek=ds, sens=s, spec=spec_fixed, seed=seed+i+j)
            grid[i, j] = estimate_TND(obs)
    return sens_vals, delta_seek_vals, grid

def sweep_over_specificity(
    spec_vals=np.linspace(0.95, 0.995, 11),
    delta_seek_vals=np.linspace(-0.4, 0.4, 17),
    base_prev=0.10,
    true_VE=0.5,
    sens_fixed=0.90,
    N=80_000,
    seed=13,
):
    grid = np.zeros((len(delta_seek_vals), len(spec_vals)))
    for i, ds in enumerate(delta_seek_vals):
        for j, sp in enumerate(spec_vals):
            pop = generate_population(N=N, base_prev=base_prev, true_VE=true_VE, seed=seed+i+j)
            obs = simulate_observation_process(pop, delta_seek=ds, sens=sens_fixed, spec=sp, seed=seed+i+j)
            grid[i, j] = estimate_TND(obs)
    return spec_vals, delta_seek_vals, grid

if __name__ == "__main__":
    _ensure_dir("figures")
    xs, ys, M = sweep_over_sensitivity()
    _heatmap(xs, ys, M, "Observed VE (TND) vs Sensitivity", "figures/misclass_sweep_sens.png", "Sensitivity")

    xs, ys, M = sweep_over_specificity()
    _heatmap(xs, ys, M, "Observed VE (TND) vs Specificity", "figures/misclass_sweep_spec.png", "Specificity")

    print("Saved: figures/misclass_sweep_sens.png, figures/misclass_sweep_spec.png")
