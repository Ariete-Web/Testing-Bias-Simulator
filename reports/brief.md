# Testing-Bias Simulator Brief

This brief summarizes the simulation that examines how differential care-seeking behavior and imperfect diagnostic tests can bias vaccine effectiveness estimates in two epidemiologic study designs: the test-negative design (TND) and a target-trial emulation (TTE).

## Key components

- **Population generation**: The simulator creates synthetic cohorts with a vaccination indicator, true infection status, and severity.
- **Care-seeking behavior**: Care-seeking probability differs between vaccinated and unvaccinated individuals by a delta parameter.
- **Test misclassification**: Imperfect test sensitivity and specificity yield false positives and negatives.
- **Estimators**: Vaccine effectiveness is estimated using both a TND (via odds ratio) and a TTE (via risk ratio).

## Bias exploration

Heatmaps illustrate how the observed VE under TND and TTE varies across levels of care-seeking differential and baseline infection prevalence. Separate sweeps explore how VE estimates change under varying test sensitivity and specificity.

## Validation

The simulation includes sanity checks ensuring that when care-seeking differential is zero and tests are perfect, the observed VE matches the true VE. Repeat evaluations assess the stability of estimates across multiple runs.
