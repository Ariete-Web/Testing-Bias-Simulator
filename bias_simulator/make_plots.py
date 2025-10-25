"""
Plots for Testing-Bias Simulator — generates heatmaps from a parameter sweep.

What this file does:
- Calls the core simulator to compute observed VE (TND and TTE) over a grid of settings.
- Converts tidy results to 2D grids.
- Saves two heatmap PNGs into the repo's figures/ folder.
- Also saves the full sweep results to CSV for reproducibility.

Why separate file?
- Keeps plotting isolated so the core logic stays clean and testable.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from bias_simulator.simulate_bias import sweep_observed_ve


def _to_grid(df, x_col: str = "delta_seek", y_col: str = "prevalence", z_col: str = "VE_TND"):
    """
    Converts a tidy DataFrame (x, y, z columns) into a matrix suitable for heatmaps.
    - Ensures columns/rows are sorted by x (columns) and y (rows).
    - Returns (xs, ys, M) where M[i, j] corresponds to y[i], x[j].
    """
    xs = np.sort(df[x_col].unique())
    ys = np.sort(df[y_col].unique())

    # Build matrix M with shape (len(ys), len(xs))
    M = np.empty((len(ys), len(xs)), dtype=float)
    for i, y in enumerate(ys):
        row = df[df[y_col] == y].sort_values(x_col)[z_col].to_numpy()
        M[i, :] = row
    return xs, ys, M


def _ensure_dir(path: str):
    """
    Creates a folder if it doesn't exist (safe to call multiple times).
    """
    os.makedirs(path, exist_ok=True)


def _heatmap(xs, ys, M, title: str, outfile: str):
    """
    Saves a simple heatmap figure:
    - x-axis: care-seeking differential (unvaccinated minus vaccinated).
    - y-axis: prevalence.
    - color: observed VE under the chosen estimator.
    """
    plt.figure()
    plt.imshow(
        M,
        aspect="auto",
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    )
    plt.colorbar(label="Observed VE")
    plt.xlabel("Δ(seeking) (unvaccinated − vaccinated)")
    plt.ylabel("Prevalence")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def main():
    """
    Runs the sweep, saves CSV + two heatmaps.
    """
    results = sweep_observed_ve()  # uses deterministic seeds inside

    # Ensure output directory exists
    _ensure_dir("figures")

    # Save tidy results for full reproducibility
    results.to_csv("figures/bias_sweep_results.csv", index=False)

    # Build grids for both estimators
    xs, ys, grid_tnd = _to_grid(results, z_col="VE_TND")
    _,  _, grid_tte = _to_grid(results, z_col="VE_TTE")

    # Write out the two required figures
    _heatmap(xs, ys, grid_tnd, "TND Observed VE", "figures/bias_heatmap_TND.png")
    _heatmap(xs, ys, grid_tte, "TTE Observed VE", "figures/bias_heatmap_TTE.png")

    print("Saved: figures/bias_heatmap_TND.png, figures/bias_heatmap_TTE.png")
    print("Saved: figures/bias_sweep_results.csv")


if __name__ == "__main__":
    main()
