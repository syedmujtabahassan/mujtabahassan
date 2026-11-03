# Ground-truth test of explanation methods on environmental-style time series

Author: Syed Muhammad Mujtaba Hassan. Prepared as supporting work for an application to the UvA
"Two PhD Positions in AI for Science" (Position 2: mechanistic interpretability).
This is a small, fully reproducible pilot, not a finished study.

## Question
When a model predicts a pollutant-like target well, do standard explanation methods point to the
variables that truly drive it, and what happens when a non-causal "proxy" feature is correlated with
a true driver?

## Setup (datagen.py)
- 6 features: wind, traffic, temperature (true drivers at lags 3, 1, 2), humidity and pressure
  (pure distractors), and `proxy` (non-causal, correlation rho with traffic).
- Target: y_t = -0.8*wind[t-3] + 0.9*traffic[t-1] + 0.5*tanh(temp[t-2]) + 0.3*wind[t-3]*traffic[t-1] + noise.
- A model sees the previous 8 steps (8 x 6 = 48 input cells). The planted ground truth is a binary
  mask over those cells: exactly 3 cells are true drivers.
- Experiment 1: all features observed cleanly. Experiment 2: the observed traffic sensor has added
  noise (sd 1.0), while the target and proxy come from the clean traffic, so the proxy can carry
  information the sensor lost.
- Model: LightGBM (300 trees). Explanations: SHAP (TreeExplainer) and permutation importance.
- Metrics: average precision of attribution vs. the planted mask (AUPRC), share of total attribution
  on the proxy, share on the true cells. rho in {0, 0.5, 0.9, 0.99}; 5 seeds each; mean reported.

## Results (means over 5 seeds; see results_*.csv and the figures)
Experiment 1, clean sensors (fig_gbdt_attribution.png): accuracy is constant (R2 = 0.94) for all rho,
while SHAP's attribution on the proxy grows from 0.019 (rho 0) to 0.114 (rho 0.99); permutation importance
leaks less (0.000 to 0.039). The effect is modest: true cells keep 0.74 (SHAP) and 0.95 (permutation)
of attribution at rho 0.99.

Experiment 2, noisy traffic sensor (fig_gbdt_noisy_sensor.png): accuracy RISES with rho (R2 0.70 to 0.93)
because the proxy is a better predictor than the noisy sensor, and attribution moves to the proxy:
proxy share 0.04 to 0.41 (SHAP) and 0.00 to 0.52 (permutation); AUPRC against the planted truth falls
from about 0.95 to 0.61.

## How to read this (important)
In Experiment 2 the explainers are arguably doing their job: they report what the model uses. What fails
is the model's mechanism relative to the world, and accuracy hides it. Attribution scores alone cannot
separate "faithful to the model" from "correct about the world". Interventional tests (for example
activation patching in a Transformer) are the natural next step and are not done here.

## Limitations / not done
- Synthetic data, one model family (GBDT). No LSTM/Transformer, no Integrated Gradients, no LIME, no patching.
- Novelty not established: a literature check against existing synthetic interpretability benchmarks
  for time series has not been done. Treat this as a rigorous pilot, not a new benchmark.
- 5 seeds; spread across seeds is in the CSV files.

## Reproduce
    pip install -r requirements.txt
    python datagen.py            # sanity check of the generator
    python exp_gbdt.py           # -> results_gbdt.csv        (about 1 min on one CPU)
    python exp_gbdt_noisy.py     # -> results_gbdt_noisy.csv
    python plot_gbdt.py && python plot_noisy.py
