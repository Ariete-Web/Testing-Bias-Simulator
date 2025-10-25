"""
Misclassification sweeps for Testing-Bias Simulator.

This script sweeps over test sensitivity and specificity separately.
For each combination, it computes observed vaccine effectiveness (VE) using
 the test‑negative design (TND) at different care‑seeking differentials.
It saves two heatmaps in the ``figures/`` directory: ``misclass_sweep_sens.png``
and ``misclass_sweep_spec.png``.
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


def _ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _heatmap(xs: np.ndarray, ys: np.ndarray, M: np.ndarray, *, title: str, outfile: str, xlabel: str) -> None:
    """Save a heatmap given x‑axis values, y‑axis values and a matrix of results."""
    plt.figure()
    plt.imshow(
        M,
        aspect="auto",
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    )
    plt.colorbar(label="Observed VE (TND)")
    plt.xlabel(xlabel)
    plt.ylabel("Δ(seeking) (unvaccinated – vaccinated)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def sweep_over_sensitivity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep over test sensitivity values and return the grid of VE estimates."""
    sens_vals = np.linspace(0.75, 0.99, 11)
    delta_seek_vals = np.linspace(-0.4, 0.4, 17)
    base_prev = 0.10
    true_VE = 0.5
    spec_fixed = 0.98
    N = 80_000
    seed_base = 0

    grid = np.zeros((len(delta_seek_vals), len(sens_vals)))
    for i, s in enumerate(sens_vals):
        for j, ds in enumerate(delta_seek_vals):
            seed = seed_base + i * len(delta_seek_vals) + j
            pop = generate_population(N=N, base_prev=base_prev, true_VE=true_VE, seed=seed)
            obs = simulate_observation_process(pop, delta_seek=ds, sens=s, spec=spec_fixed, seed=seed)
            grid[j, i] = estimate_TND(obs)
    return sens_vals, delta_seek_vals, grid


def sweep_over_specificity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep over test specificity values and return the grid of VE estimates."""
    spec_vals = np.linspace(0.65, 0.995, 11)
    delta_seek_vals = np.linspace(-0.4, 0.4, 17)
    base_prev = 0.10
    true_VE = 0.5
    sens_fixed = 0.90
    N = 80_000
    seed_base = 1000

    grid = np.zeros((len(delta_seek_vals), len(spec_vals)))
    for i, sp in enumerate(spec_vals):
        for j, ds in enumerate(delta_seek_vals):
            seed = seed_base + i * len(delta_seek_vals) + j
            pop = generate_population(N=N, base_prev=base_prev, true_VE=true_VE, seed=seed)
            obs = simulate_observation_process(pop, delta_seek=ds, sens=sens_fixed, spec=sp, seed=seed)
            grid[j, i] = estimate_TND(obs)
    return spec_vals, delta_seek_vals, grid


def main() -> None:
    """Generate misclassification sweep heatmaps and save them to disk."""
    sens_vals, delta_vals_s, grid_sens = sweep_over_sensitivity()
    spec_vals, delta_vals_p, grid_spec = sweep_over_specificity()

    _ensure_dir("figures")

    # Plot sensitivity sweep heatmap (transpose grid for correct orientation)
    _heatmap(
        xs=sens_vals,
        ys=delta_vals_s,
        M=grid_sens.T,
        title="TND VE vs Test Sensitivity and Care-seeking Differential",
        outfile="figures/misclass_sweep_sens.png",
        xlabel="Test Sensitivity",
    )

    # Plot specificity sweep heatmap
    _heatmap(
        xs=spec_vals,
        ys=delta_vals_p,
        M=grid_spec.T,
        title="TND VE vs Test Specificity and Care-seeking Differential",
        outfile="figures/misclass_sweep_spec.png",
        xlabel="Test Specificity",
    )


if __name__ == "__main__":
    main()
