"""
Bias sweeps for Testing-Bias Simulator.

This script sweeps over care‑seeking differential (delta_seek) and baseline infection prevalence.
For each combination, it computes observed vaccine effectiveness using both the
Test‑Negative Design (TND) estimator and the Target Trial Emulation (TTE) estimator.
It saves two heatmaps (``bias_heatmap_TND.png`` and ``bias_heatmap_TTE.png``) and a tidy results
CSV file (``bias_sweep_results.csv``) in the ``figures/`` directory.  All random number
seeds are fixed so that results are deterministic across runs.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bias_simulator.simulate_bias import (
    generate_population,
    simulate_observation_process,
    estimate_TND,
    estimate_TTE,
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
    plt.colorbar(label="Observed VE")
    plt.xlabel(xlabel)
    plt.ylabel("Δ(seeking) (unvaccinated – vaccinated)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def bias_sweep(
    delta_seek_vals: np.ndarray,
    base_prev_vals: np.ndarray,
    *,
    true_VE: float = 0.5,
    sens: float = 0.98,
    spec: float = 0.98,
    N: int = 80_000,
    seed_base: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute VE estimates across a grid of delta_seek and base prevalence values."""
    grid_TND = np.zeros((len(delta_seek_vals), len(base_prev_vals)))
    grid_TTE = np.zeros((len(delta_seek_vals), len(base_prev_vals)))
    for j, ds in enumerate(delta_seek_vals):
        for i, bp in enumerate(base_prev_vals):
            seed = seed_base + j * len(base_prev_vals) + i
            pop = generate_population(N=N, base_prev=bp, true_VE=true_VE, seed=seed)
            obs = simulate_observation_process(pop, delta_seek=ds, sens=sens, spec=spec, seed=seed)
            grid_TND[j, i] = estimate_TND(obs)
            grid_TTE[j, i] = estimate_TTE(obs)
    return delta_seek_vals, base_prev_vals, grid_TND, grid_TTE


def main() -> None:
    """Run the bias sweep and save heatmaps and results."""
    # Define parameter grids
    delta_seek_vals = np.linspace(-0.4, 0.4, 17)  # unvaccinated – vaccinated
    base_prev_vals = np.linspace(0.01, 0.30, 15)  # baseline infection prevalence

    # Perform sweep
    ds_vals, bp_vals, grid_TND, grid_TTE = bias_sweep(delta_seek_vals, base_prev_vals)

    # Ensure output directories exist
    _ensure_dir("figures")

    # Save heatmaps (transpose grids for correct orientation: columns correspond to base_prev)
    _heatmap(
        xs=bp_vals,
        ys=ds_vals,
        M=grid_TND.T,
        title="Observed VE (TND) vs Baseline Prevalence and Care-seeking Differential",
        outfile="figures/bias_heatmap_TND.png",
        xlabel="Baseline infection prevalence",
    )

    _heatmap(
        xs=bp_vals,
        ys=ds_vals,
        M=grid_TTE.T,
        title="Observed VE (TTE) vs Baseline Prevalence and Care-seeking Differential",
        outfile="figures/bias_heatmap_TTE.png",
        xlabel="Baseline infection prevalence",
    )

    # Save tidy results as CSV
    # Each row: delta_seek, base_prev, VE_TND, VE_TTE
    records = []
    for j, ds in enumerate(ds_vals):
        for i, bp in enumerate(bp_vals):
            records.append({
                "delta_seek": float(ds),
                "base_prev": float(bp),
                "VE_TND": float(grid_TND[j, i]),
                "VE_TTE": float(grid_TTE[j, i]),
            })
    df = pd.DataFrame.from_records(records)
    df.to_csv("figures/bias_sweep_results.csv", index=False)


if __name__ == "__main__":
    main()
