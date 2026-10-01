"""
Testing-Bias Simulator (VE) — clean core with explanatory notes.

What this script shows:
- How care-seeking differences and imperfect tests can bias vaccine effectiveness (VE)
  under two approaches: Test-Negative Design (TND) and Target-Trial Emulation (TTE).
- Reproducible helpers and a small validation routine.

Only uses: numpy, pandas (lightweight).
"""

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd


# -----------------------------
# Utility: reproducible RNG
# -----------------------------
def _rng(seed: int | None = None) -> np.random.Generator:
    """Creates a NumPy random generator; seeding makes runs reproducible."""
    return np.random.default_rng(seed)


# ----------------------------------------------------
# 1) Cohort generator (exposure, outcome, severity)
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
      - severity: continuous symptom severity in [0,1]
    Assumptions:
      - Vaccination multiplies down infection probability by 'true_VE'.
      - Severity skews higher for truly infected.
    """
    rng = _rng(seed)
    vacc = (rng.random(N) < p_vacc).astype(int)
    p_inf = base_prev * (1.0 - true_VE * vacc)
    infected_true = (rng.random(N) < p_inf).astype(int)
    severity = np.clip(rng.beta(2, 5, size=N) + 0.4 * infected_true, 0.0, 1.0)
    return pd.DataFrame({"vacc": vacc, "infected_true": infected_true, "severity": severity})


# -------------------------------------------------------
# 2) Care-seeking behavior (who goes to get a test)
# -------------------------------------------------------
def care_seek_prob(
    severity: np.ndarray,
    vacc: np.ndarray,
    delta_seek: float = 0.2,
) -> np.ndarray:
    """
    Maps severity & vaccination to a probability of seeking testing.
    - Base seeking increases with severity.
    - delta_seek shifts unvaccinated relative to vaccinated:
        + positive => unvaccinated seek more on average
        + negative => unvaccinated seek less on average
    Returns probabilities in [0,1].
    """
    severity = np.asarray(severity)
    vacc = np.asarray(vacc)
    base = 0.10 + 0.70 * severity
    adj = base + delta_seek * (1 - vacc)     # (1 - vacc) = 1 for unvaccinated
    return np.clip(adj, 0.0, 1.0)


# -------------------------------------------------------
# 3) Imperfect diagnostic testing (sens/spec)
# -------------------------------------------------------
def apply_test(
    infected_true: np.ndarray,
    sens: float = 0.90,
    spec: float = 0.98,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Simulates test results with given sensitivity & specificity.
    Output: 1 if test is positive, else 0.
    """
    rng = _rng(seed)
    infected_true = np.asarray(infected_true).astype(int)
    N = len(infected_true)

    is_infected = infected_true == 1
    tp = (rng.random(N) < sens) & is_infected
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
    TND (seekers only): VE = 1 - OR, OR = (a*d)/(b*c)
      a = vacc=1 & test_pos=1
      b = vacc=1 & test_pos=0
      c = vacc=0 & test_pos=1
      d = vacc=0 & test_pos=0
    Continuity correction (+0.5) avoids zero-cell problems.
    """
    sub = df[df["seek"] == 1]
    if len(sub) == 0:
        return float("nan")
    a = ((sub["vacc"] == 1) & (sub["test_pos"] == 1)).sum()
    b = ((sub["vacc"] == 1) & (sub["test_pos"] == 0)).sum()
    c = ((sub["vacc"] == 0) & (sub["test_pos"] == 1)).sum()
    d = ((sub["vacc"] == 0) & (sub["test_pos"] == 0)).sum()
    or_est = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    return float(1.0 - or_est)


def estimate_TTE(df: pd.DataFrame) -> float:
    """
    TTE (full cohort): VE = 1 - RR, with RR = risk_vaccinated / risk_unvaccinated.
    Risk uses the observed outcome (test_pos among the full cohort, with
    non-seekers counted as undetected) rather than the unobservable true
    infection status. This keeps TTE a full-cohort comparator (no selection
    on care-seeking, unlike TND) while still exposing it to the same
    under-ascertainment bias a real target-trial emulation would face.
    """
    eps = 1e-6
    risk_v = df.loc[df["vacc"] == 1, "test_pos"].mean()
    risk_u = df.loc[df["vacc"] == 0, "test_pos"].mean()
    rr = (risk_v + eps) / (risk_u + eps)
    return float(1.0 - rr)


# -------------------------------------------------------
# 5) Observation process (seeking + testing)
# -------------------------------------------------------
def simulate_observation_process(
    df: pd.DataFrame,
    delta_seek: float = 0.2,
    sens: float = 0.90,
    spec: float = 0.98,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Adds 'seek' and 'test_pos' columns:
      - 'seek' indicates who actually goes for testing
      - 'test_pos' is the observed test result for seekers
    """
    # Offset from `seed` so this RNG stream never collides with the one used
    # by generate_population(seed=...): callers (repeat_eval, the bias and
    # misclassification sweeps) pass the SAME seed to both functions, and
    # both functions' first draw is an N-length rng.random(N) array (for
    # 'vacc' there, for 'seek' here). With the same seed those two arrays
    # are bit-for-bit identical, which spuriously ties who is vaccinated to
    # who seeks care, swamping the intended delta_seek effect.
    rng = _rng(seed + 1_000_003)
    pr_seek = care_seek_prob(df["severity"].values, df["vacc"].values, delta_seek=delta_seek)
    seek = (rng.random(len(df)) < pr_seek).astype(int)

    test_pos = np.zeros(len(df), dtype=int)
    idx_seek = seek == 1
    # Also offset from the seek draw above, for the same reason: reusing the
    # same seed for testing would make the test-outcome stream an exact
    # prefix of the seek-outcome stream.
    test_pos[idx_seek] = apply_test(
        df.loc[idx_seek, "infected_true"].values,
        sens=sens, spec=spec, seed=seed + 2_000_003
    )

    out = df.copy()
    out["seek"] = seek
    out["test_pos"] = test_pos
    return out


# -------------------------------------------------------
# 6) Validation helpers (stability & sanity checks)
# -------------------------------------------------------
def repeat_eval(
    repeats: int = 10,
    N: int = 50_000,
    p_vacc: float = 0.6,
    true_VE: float = 0.5,
    base_prev: float = 0.10,
    delta_seek: float = 0.2,
    sens: float = 0.90,
    spec: float = 0.98,
    seed0: int = 100,
) -> dict:
    """
    Repeats the full pipeline multiple times with different seeds and summarizes
    the mean±sd for VE_TND and VE_TTE.
    """
    ve_tnd, ve_tte = [], []
    for k in range(repeats):
        pop = generate_population(N=N, p_vacc=p_vacc, true_VE=true_VE, base_prev=base_prev, seed=seed0 + k)
        obs = simulate_observation_process(pop, delta_seek=delta_seek, sens=sens, spec=spec, seed=seed0 + k)
        ve_tnd.append(estimate_TND(obs))
        ve_tte.append(estimate_TTE(obs))
    arr_tnd = np.array(ve_tnd, dtype=float)
    arr_tte = np.array(ve_tte, dtype=float)
    return {
        "VE_TND": {"mean": float(np.nanmean(arr_tnd)), "sd": float(np.nanstd(arr_tnd))},
        "VE_TTE": {"mean": float(np.nanmean(arr_tte)), "sd": float(np.nanstd(arr_tte))},
        "repeats": repeats
    }


def validate_sanity(
    base_prev: float = 0.10,
    true_VE: float = 0.5,
    delta_seek: float = 0.0,
) -> dict:
    """
    Simple check: with no care-seeking differential and perfect tests,
    TND and TTE should be close to the true VE.
    """
    pop = generate_population(base_prev=base_prev, true_VE=true_VE)
    obs = simulate_observation_process(pop, delta_seek=delta_seek, sens=1.0, spec=1.0, seed=999)
    return {
        "VE_true": true_VE,
        "VE_TND": estimate_TND(obs),
        "VE_TTE": estimate_TTE(obs)
    }


# -------------------------------------------------------
# 7) Minimal console demo & JSON summary
# -------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)

    # Sanity check near perfect conditions
    sanity = validate_sanity()
    print({"sanity": sanity})

    # Stability over repeats (mean ± sd)
    summary = repeat_eval()
    print({"stability": summary})

    # Save a small JSON so results are preserved
    with open("artifacts/validation_summary.json", "w") as f:
        json.dump({"sanity": sanity, "stability": summary}, f, indent=2)
    print("Saved artifacts/validation_summary.json")
