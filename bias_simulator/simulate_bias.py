"""
Testing-Bias Simulator (VE) — clean version with explanatory notes.

What this script shows:
- How differential care-seeking and imperfect tests can bias vaccine effectiveness (VE)
  under two approaches: Test-Negative Design (TND) and Target-Trial Emulation (TTE).

Only standard libraries used: numpy, pandas (for convenience).
No plotting here; this file focuses on the core logic and reproducible calculations.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# -----------------------------
# Utility: reproducible RNG
# -----------------------------
def _rng(seed: int | None = None) -> np.random.Generator:
    """
    Creates a NumPy random number generator. Using a seed makes runs reproducible.
    """
    return np.random.default_rng(seed)


# ----------------------------------------------------
# 1) Population generator (exposure, outcome, severity)
# ----------------------------------------------------
def generate_population(
    N: int = 100_000,
    p_vacc: float = 0.6,
    true_VE: float = 0.5,
    base_prev: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Builds a synthetic cohort:
      - vacc: 1 if vaccinated, else 0
      - infected_true: 1 if truly infected, else 0
      - severity: continuous symptom severity in [0,1] (higher ~ worse symptoms)

    Assumptions:
      - Vaccination reduces infection probability by 'true_VE' multiplicatively.
      - Severity skews higher for truly infected individuals.
    """
    rng = _rng(seed)

    # Exposure: who is vaccinated
    vacc = (rng.random(N) < p_vacc).astype(int)

    # True infection probability is reduced for vaccinated
    p_inf = base_prev * (1.0 - true_VE * vacc)
    infected_true = (rng.random(N) < p_inf).astype(int)

    # Severity: start from a mild distribution, bump it up if infected
    # (beta(2,5) ~ mostly low values; +0.4 for infected shifts severity upward)
    severity = np.clip(rng.beta(2, 5, size=N) + 0.4 * infected_true, 0.0, 1.0)

    return pd.DataFrame(
        {"vacc": vacc, "infected_true": infected_true, "severity": severity}
    )


# -------------------------------------------------------
# 2) Care-seeking behavior (who actually goes to get a test)
# -------------------------------------------------------
def care_seek_prob(
    severity: np.ndarray,
    vacc: np.ndarray,
    delta_seek: float = 0.2,
) -> np.ndarray:
    """
    Converts severity and vaccination into a probability of seeking a test.

    Interpretation:
      - Base seeking increases with severity (people with worse symptoms tend to seek care).
      - 'delta_seek' shifts the seeking probability for unvaccinated vs vaccinated:
          * Positive delta_seek => unvaccinated seek testing more often on average.
          * Negative delta_seek => unvaccinated seek testing less often on average.

    Returns:
      - Array of probabilities in [0,1], one per person.
    """
    severity = np.asarray(severity)
    vacc = np.asarray(vacc)

    base = 0.10 + 0.70 * severity  # low at 0, high near 1
    # Adjust unvaccinated: (1 - vacc) is 1 for unvaccinated, 0 for vaccinated
    adj = base + delta_seek * (1 - vacc)
    return np.clip(adj, 0.0, 1.0)


# -------------------------------------------------------
# 3) Imperfect diagnostic testing (sensitivity / specificity)
# -------------------------------------------------------
def apply_test(
    infected_true: np.ndarray,
    sens: float = 0.90,
    spec: float = 0.98,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Simulates a diagnostic test result with given sensitivity/specificity.

    Inputs:
      - infected_true: 1 if truly infected, else 0
      - sens: probability test is positive if infected (true positive rate)
      - spec: probability test is negative if not infected (true negative rate)

    Output:
      - test_pos: 1 if test is positive, else 0
    """
    rng = _rng(seed)
    infected_true = np.asarray(infected_true).astype(int)
    N = len(infected_true)

    # True positives & false negatives for infected people
    is_infected = infected_true == 1
    tp = (rng.random(N) < sens) & is_infected
    fn = (~tp) & is_infected  # not used directly, here for clarity

    # True negatives & false positives for non-infected people
    not_infected = ~is_infected
    tn = (rng.random(N) < spec) & not_infected
    fp = (~tn) & not_infected

    test_pos = tp | fp
    return test_pos.astype(int)


# -------------------------------------------------------
# 4) Estimators: TND and TTE observed VE
# -------------------------------------------------------
def estimate_TND(df: pd.DataFrame) -> float:
    """
    Test-Negative Design VE estimate using seekers only.

    2x2 table among those who sought testing:
        test_pos   test_neg
    v=1     a         b
    v=0     c         d

    OR = (a*d)/(b*c); VE = 1 - OR
    Continuity correction (+0.5) avoids division by zero when any cell is 0.
    """
    sub = df[df["seek"] == 1]
    if len(sub) == 0:
        return float("nan")

    a = ((sub["vacc"] == 1) & (sub["test_pos"] == 1)).sum()
    b = ((sub["vacc"] == 1) & (sub["test_pos"] == 0)).sum()
    c = ((sub["vacc"] == 0) & (sub["test_pos"] == 1)).sum()
    d = ((sub["vacc"] == 0) & (sub["test_pos"] == 0)).sum()

    or_est = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    ve = 1.0 - or_est
    return float(ve)


def estimate_TTE(df: pd.DataFrame) -> float:
    """
    Target-Trial Emulation VE estimate using true infection risk in the full cohort.

    VE = 1 - RR, where RR = risk_vaccinated / risk_unvaccinated.
    A tiny epsilon avoids divide-by-zero if a group has zero cases.
    """
    eps = 1e-6
    risk_v = df.loc[df["vacc"] == 1, "infected_true"].mean()
    risk_u = df.loc[df["vacc"] == 0, "infected_true"].mean()
    rr = (risk_v + eps) / (risk_u + eps)
    ve = 1.0 - rr
    return float(ve)


# -------------------------------------------------------
# 5) Tiny helper: construct a seeking+testing dataset from a cohort
# -------------------------------------------------------
def simulate_observation_process(
    df: pd.DataFrame,
    delta_seek: float = 0.2,
    sens: float = 0.90,
    spec: float = 0.98,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Adds 'seek' and 'test_pos' columns to the cohort, representing:
      - who actually seeks testing (based on severity & vaccination)
      - who tests positive (based on true infection and test performance)
    """
    rng = _rng(seed)

    # Who seeks testing?
    pr_seek = care_seek_prob(df["severity"].values, df["vacc"].values, delta_seek=delta_seek)
    seek = (rng.random(len(df)) < pr_seek).astype(int)

    # Apply test only to seekers
    test_pos = np.zeros(len(df), dtype=int)
    idx_seek = seek == 1
    test_pos[idx_seek] = apply_test(
        df.loc[idx_seek, "infected_true"].values,
        sens=sens, spec=spec, seed=seed
    )

    out = df.copy()
    out["seek"] = seek
    out["test_pos"] = test_pos
    return out


# -------------------------------------------------------
# 6) Optional: quick parameter sweep (no plotting, returns a tidy table)
# -------------------------------------------------------
def sweep_observed_ve(
    delta_seek_vals: np.ndarray = np.linspace(-0.4, 0.4, 9),
    prev_vals: np.ndarray = np.linspace(0.05, 0.25, 5),
    true_VE: float = 0.5,
    p_vacc: float = 0.6,
    sens: float = 0.90,
    spec: float = 0.98,
    N: int = 100_000,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Explores how observed VE changes across care-seeking differentials and prevalence.

    Returns a tidy DataFrame with columns:
      - prevalence, delta_seek, VE_TND, VE_TTE
    """
    rows = []
    for pv in prev_vals:
        for ds in delta_seek_vals:
            pop = generate_population(N=N, p_vacc=p_vacc, true_VE=true_VE, base_prev=pv, seed=seed)
            obs = simulate_observation_process(pop, delta_seek=ds, sens=sens, spec=spec, seed=seed)
            ve_tnd = estimate_TND(obs)
            ve_tte = estimate_TTE(obs)
            rows.append({"prevalence": pv, "delta_seek": ds, "VE_TND": ve_tnd, "VE_TTE": ve_tte})
    return pd.DataFrame(rows)


# -------------------------------------------------------
# 7) Minimal demo: prints a couple of numbers so users see it's working
# -------------------------------------------------------
if __name__ == "__main__":
    # Build a cohort with default settings
    cohort = generate_population(N=50_000, p_vacc=0.6, true_VE=0.5, base_prev=0.10, seed=42)

    # Run the observation process: who seeks testing, who tests positive
    observed = simulate_observation_process(
        cohort,
        delta_seek=0.2,     # unvaccinated seek a bit more
        sens=0.90,          # test sensitivity
        spec=0.98,          # test specificity
        seed=123
    )

    # Compute observed VE under TND (seekers only) and TTE (full cohort)
    ve_tnd = estimate_TND(observed)
    ve_tte = estimate_TTE(observed)
    print({"VE_TND": round(ve_tnd, 3), "VE_TTE": round(ve_tte, 3)})

    # Optional: small sweep to show trends (comment out if not needed)
    # results = sweep_observed_ve()
    # print(results.head())
