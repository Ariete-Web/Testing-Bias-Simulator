"""
Plots for Testing-Bias Simulator — generates heatmaps from the sweep.
This file only handles plotting so the core logic stays clean.
"""
import numpy as np
import matplotlib.pyplot as plt
from bias_simulator.simulate_bias import sweep_observed_ve

def _to_grid(df, x_col="delta_seek", y_col="prevalence", z_col="VE_TND"):
    xs = np.sort(df[x_col].unique())
    ys = np.sort(df[y_col].unique())
    grid = np.empty((len(ys), len(xs)))
    for i, y in enumerate(ys):
        row = df[df[y_col]==y].sort_values(x_col)[z_col].values
        grid[i, :] = row
    return xs, ys, grid

def heatmap(xs, ys, M, title, outfile):
    plt.figure()
    plt.imshow(M, aspect="auto", origin="lower",
               extent=[xs.min(), xs.max(), ys.min(), ys.max()])
    plt.colorbar(label="Observed VE")
    plt.xlabel("Δ(seeking) (unvacc − vacc)")
    plt.ylabel("Prevalence")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)

if __name__ == "__main__":
    df = sweep_observed_ve()
    xs, ys, tnd = _to_grid(df, z_col="VE_TND")
    _,  _, tte = _to_grid(df, z_col="VE_TTE")
    heatmap(xs, ys, tnd, "TND Observed VE", "figures/bias_heatmap_TND.png")
    heatmap(xs, ys, tte, "TTE Observed VE", "figures/bias_heatmap_TTE.png")
    print("Saved: figures/bias_heatmap_TND.png, figures/bias_heatmap_TTE.png")
