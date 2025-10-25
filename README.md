# Testing‑Bias Simulator

A small, fully reproducible simulation that illustrates how differential care‑seeking and test misclassification can bias estimates of vaccine effectiveness under two study designs: **test‑negative design (TND)** and **target‑trial emulation (TTE)**.  The repository is designed to run entirely via GitHub Actions without any local setup, and it includes scripts, figures, a brief report and validation summary.

## Scientific objectives

* Demonstrate how differences in care‑seeking between vaccinated and unvaccinated individuals (delta seeking) create bias in TND and TTE vaccine effectiveness estimates.
* Quantify the impact of imperfect diagnostic tests (varying sensitivity and specificity) on observed vaccine effectiveness.
* Provide deterministic, replicable simulations that can be used for teaching or portfolio demonstration.

## Methods in plain language

1. **Generate a population** of synthetic individuals with a true infection status and vaccination indicator.  A helper function draws from a specified distribution for underlying risk and assigns a baseline infection probability.
2. **Care‑seeking probability**: assign a probability of seeking care (and therefore being tested) based on severity and a delta parameter that controls whether vaccinated people seek care more or less often than unvaccinated people.
3. **Imperfect testing**: apply a diagnostic test with configurable sensitivity and specificity; misclassification is simulated on those who seek care.
4. **Estimators**:
   - **Test‑Negative Design (TND)**: estimate vaccine effectiveness using the odds ratio of vaccination among test‑positive vs test‑negative care seekers.
   - **Target‑Trial Emulation (TTE)**: estimate vaccine effectiveness by comparing infection risk in vaccinated vs unvaccinated groups.
5. **Observation and repeat evaluation**: simulate the full pipeline many times (using a fixed random seed) to compute mean and standard deviation of both estimators.  Sanity and stability checks write a summary to `artifacts/validation_summary.json`.

## Repository layout

```
bias_simulator/
├── simulate_bias.py     # core logic: population generation, care seeking, testing, estimators, validation
├── make_plots.py        # sweeps over care‑seeking differential and prevalence; produces bias heatmaps and results CSV
├── misclass_plots.py    # sweeps test sensitivity and specificity; produces misclassification heatmaps
figures/                 # output images (heatmaps) and results table
artifacts/
├── validation_summary.json  # sanity and stability summary from simulate_bias.py
reports/
├── brief.md             # two‑page narrative summary (Markdown)
├── brief.pdf            # PDF report (built via workflow)
README.md                # this file
CHANGELOG.md             # change log with v1.0 and v1.1 entries
LICENSE                  # MIT license
requirements.txt         # minimal dependencies (numpy, pandas, matplotlib)
.github/workflows/
├── build-figures.yml    # builds and commits figures and results
└── build-brief.yml      # builds and commits the brief PDF
```

## Running locally (optional)

To run the simulation or generate figures on your own machine:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# generate a validation summary
python bias_simulator/simulate_bias.py

# generate bias heatmaps and results table
python bias_simulator/make_plots.py

# generate misclassification heatmaps
python bias_simulator/misclass_plots.py
```

## Running on GitHub (recommended)

You do not need to install anything locally.  Use GitHub’s Actions tab:

1. **Generate figures and CSV** – in the Actions tab, select **Build figures** and click **Run workflow**.  This runs both plotting scripts and commits the following to the repository:
   - `figures/bias_heatmap_TND.png`  
   - `figures/bias_heatmap_TTE.png`  
   - `figures/misclass_sweep_sens.png`  
   - `figures/misclass_sweep_spec.png`  
   - `figures/bias_sweep_results.csv`
2. **Build the report** – from Actions choose **Build brief PDF** and click **Run workflow**.  This converts the Markdown brief into `reports/brief.pdf` and commits it.

Outputs are deterministic because random seeds are fixed; rerunning the workflows yields identical files (apart from metadata).

## Outputs

* **Figures**: `bias_heatmap_TND.png` and `bias_heatmap_TTE.png` show the bias in observed vaccine effectiveness (TND and TTE) across care‑seeking differential and true prevalence.  `misclass_sweep_sens.png` and `misclass_sweep_spec.png` show how varying sensitivity or specificity affects the TND estimator.  
* **Table**: `bias_sweep_results.csv` contains the underlying numeric values for the heatmaps.  
* **Validation**: `artifacts/validation_summary.json` summarises sanity and stability checks.  
* **Brief**: `reports/brief.pdf` is a two‑page narrative explaining the problem and results.

---

Random seeds are controlled throughout, so results are reproducible.  Use this simulator as a compact teaching tool for understanding biases in vaccine effectiveness studies or for demonstration in your portfolio.
