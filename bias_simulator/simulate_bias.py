"""
Testing-Bias Simulator (Vaccine Effectiveness) — single-file version

What this script shows:
- How differences in care-seeking behavior and test accuracy can bias vaccine effectiveness (VE)
- Comparison of two estimators:
    1) Test-Negative Design (TND) — uses only people who sought a test
    2) Target-Trial Emulation (TTE) — compares infection risk in the whole cohort

How to use:
- Run this file (e.g., python simulate_bias.py) to print example VE numbers
- It will also generate two heatmaps in ./figures/:
    - bias_heatmap_TND.png
    - bias_heatmap_TTE.png
"""

# ----------------------------
# Imports
# ----------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Population & behavior models
# ----------------------------
def generate_population(
    N: int = 100_000,
    p_vacc: float = 0.6,
    true_VE: float = 0.5,
    base_prev: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic population with vaccination status, true infection, and symptom severity.

    Parameters
    ----------
    N : total number of people
    p_vacc : fraction vaccinated
    true_VE : true vaccine effectiveness (reduces infection probability for vaccinated)
    base_prev : baseline infection prevalence without vaccination
    seed : random seed for reproducibility

    Returns
    -------
    DataFrame with columns:
      - vacc (0/1)
      - infected_true (0/1)
      - severity (0..1, higher ~ more symptoms)
    """
    rng = np.random.default_rng(seed)

    # Vaccination assignment
    vacc = (rng.random(N) < p_vacc).astype(int)

    # True infection probability is reduced for vaccinated by true_VE
    p_inf = base_prev * (1 - true_VE * vacc)
    infected_true = (rng.random(N) < p_inf).astype(int)

    # Symptom severity: base noise + bump if truly infected
    severity = np.clip(rng.beta(2, 5, size=N) + 0.4 * infected_true, 0, 1)

    return pd.DataFrame(
        {"vacc": vacc, "infected_true": infected_true, "severity": severity}
    )


def care_seek_prob(
    severity: np.ndarray,
    vacc: np.ndarray,
    delta_seek: float = 0.2,
) -> np.ndarray:
    """
    Convert severity and vaccination into a probability of seeking a diagnostic test.

    Idea:
      - People with higher severity are more likely to seek a test
      - delta_seek shifts care-seeking for unvaccinated (positive means they seek more)

    Returns a vector of probabilities in [0, 1].
    """
    severity = np.asarray(severity)
    vacc = np.asarray(vacc)

    # Base relationship: low severity ~ ~10% seek; high severity ~ ~80% seek
    base = 0.1 + 0.7 * severity

    # Shift unvaccinated seeking up or down by delta_seek
    # (1 - vacc) is 1 for unvaccinated, 0 for vaccinated
    adjusted = base + delta_seek * (1 - vacc)

    return np.clip(adjusted, 0.0, 1.0)


def apply_test(
    infected_true: np.ndarray,
    sens: float = 0.90,
    spec: float = 0.98,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate a diagnostic test with given sensitivity and specificity.

    Returns a 0/1 array 'test_pos':
      1 = test returns positive
      0 = test returns negative
    """
    rng = rng or np.random.default_rng(0)
    infected_true = np.asarray(infected_true).astype(int)
    N = len(infected_true)

    # True/false outcomes for each person
    tp = (rng.random(N) < sens) & (infected_true == 1)  # true positives
    fn = (rng.random(N) >= sens) & (infected_true == 1)  # false negatives
    tn = (rng.random(N) < spec) & (infected_true == 0)  # true negatives
    fp = (rng.random(N) >= spec) & (infected_true == 0)  # false positives

    test_pos = tp | fp
    # (tn and fn are implicitly 'not positive')
    return test_pos.astype(int)


# ----------------------------
# Effectiveness estimators
# ----------------------------
def estimate_TND(df: pd.DataFrame) -> float:
    """
    Test-Negative Design VE estimate.

    Uses only people who sought testing.
    Constructs the odds ratio of vaccination among test-positives vs test-negatives,
    then converts to VE = 1 − OR.
    """
    sub = df[df["seek"] == 1]
    if len(sub) == 0:
        return 0.0

    a = ((sub["vacc"] == 1) & (sub["test_pos"] == 1)).sum()
    b = ((sub["vacc"] == 1) & (sub["test_pos"] == 0)).sum()
    c = ((sub["vacc"] == 0) & (sub["test_pos"] == 1)).sum()
    d = ((sub["vacc"] == 0) & (sub["test_pos"] == 0)).sum()

    # Continuity correction to avoid division by zero when any cell is 0
    or_est = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    VE_obs = 1 - or_est
    return float(VE_obs)


def estimate_TTE(df: pd.DataFrame) -> float:
    """
    Target-Trial Emulation VE estimate (idealized).

    Compares infection risk in vaccinated vs unvaccinated in the whole cohort.
    VE = 1 − risk_ratio.
    """
    risk_v = df.loc[df["vacc"] == 1, "infected_true"].mean()
    risk_u = df.loc[df["vacc"] == 0, "infected_true"].mean()
    rr = (risk_v + 1e-6) / (risk_u + 1e-6)  # small epsilon to avoid 0/0
    VE_obs = 1 - rr
    return float(VE_obs)


# ----------------------------
# Parameter sweep + plotting
# ----------------------------
def sweep_and_plot(
    delta_seek_vals: np.ndarray | None = None,
    prev_vals: np.ndarray | None = None,
    true_VE: float = 0.5,
    sens: float = 0.90,
    spec: float = 0.98,
    outdir: str = "figures",
) -> None:
    """
    Explore how observed VE changes across:
      - delta_seek: how much more (or less) unvaccinated seek tests vs vaccinated
      - prevalence: baseline infection level in the population

    Saves two heatmaps:
      - bias_heatmap_TND.png  (observed VE under TND)
      - bias_heatmap_TTE.png  (observed VE under TTE)
    """
    os.makedirs(outdir, exist_ok=True)

    # Ranges to explore (x-axis = care-seeking difference, y-axis = prevalence)
    delta_seek_vals = (
        delta_seek_vals if delta_seek_vals is not None else np.linspace(-0.4, 0.4, 17)
    )
    prev_vals = prev_vals if prev_vals is not None else np.linspace(0.02, 0.25, 10)

    grid_TND = np.zeros((len(prev_vals), len(delta_seek_vals)))
    grid_TTE = np.zeros_like(grid_TND)

    for i, pv in enumerate(prev_vals):
        for j, ds in enumerate(delta_seek_vals):
            # Build population
            df = generate_population(base_prev=pv, true_VE=true_VE)

            # Who seeks testing?
            rng = np.random.default_rng(123)
            pr = care_seek_prob(df["severity"].values, df["vacc"].values, delta_seek=ds)
            df["seek"] = (rng.random(len(df)) < pr).astype(int)

            # Apply tests for seekers only
            df["test_pos"] = 0
            is_seeker = df["seek"] == 1
            df.loc[is_seeker, "test_pos"] = apply_test(
                df.loc[is_seeker, "infected_true"].values, sens=sens, spec=spec, rng=rng
            )

            # Estimates
            grid_TND[i, j] = estimate_TND(df)
            grid_TTE[i, j] = estimate_TTE(df)

    # Helper to draw a heatmap
    def plot_heatmap(M, xvals, yvals, title, outfile):
        plt.figure()
        plt.imshow(
            M,
            aspect="auto",
            origin="lower",
            extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]],
        )
        plt.colorbar(label="Observed VE")
        plt.xlabel("Δ(seeking)  (unvacc − vacc)")
        plt.ylabel("Prevalence")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, outfile), dpi=160)
        plt.close()

    plot_heatmap(grid_TND, delta_seek_vals, prev_vals, "TND Observed VE", "bias_heatmap_TND.png")
    plot_heatmap(grid_TTE, delta_seek_vals, prev_vals, "TTE Observed VE", "bias_heatmap_TTE.png")


# ----------------------------
# Example run (safe defaults)
# ----------------------------
if __name__ == "__main__":
    # Build a population and simulate a single scenario to print some numbers
    df = generate_population(N=50_000, base_prev=0.10, true_VE=0.5)

    # Who seeks testing (unvaccinated seek a bit more: delta_seek=+0.2)
    rng = np.random.default_rng(999)
    seek_p = care_seek_prob(df["severity"].values, df["vacc"].values, delta_seek=0.2)
    df["seek"] = (rng.random(len(df)) < seek_p).astype(int)

    # Apply tests for seekers only (imperfect sensitivity/specificity)
    df["test_pos"] = 0
    seekers = df["seek"] == 1
    df.loc[seekers, "test_pos"] = apply_test(
        df.loc[seekers, "infected_true"].values, sens=0.90, spec=0.98, rng=rng
    )

    # Print observed VE from both methods
    print("Observed VE (TND):", round(estimate_TND(df), 3))
    print("Observed VE (TTE):", round(estimate_TTE(df), 3))

    # Create heatmaps to visualize how observed VE changes across settings
    sweep_and_plot()
    print("Saved heatmaps to ./figures/")
