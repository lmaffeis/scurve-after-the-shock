# The S-Curve After the Shock

Machine learning evidence on mortgage prepayment in the lock-in era. Loan-level
hazard models (logistic, LightGBM, and a neural network) estimated on 18.5 million
Fannie Mae mortgages, 2018–2025, evaluated walk-forward against full-population
cohort speeds.

The full writeup is in [docs/paper/latex/main.pdf](docs/paper/latex/main.pdf).

## Summary of results

A model estimated through 2021 over-predicted aggregate speeds by roughly 9 CPR at
the onset of the 2022 rate shock, and its cohort errors never recovered (3.1–3.8
CPR points in every year through 2025). Annual re-estimation halved the error
within a year and needed about two more years of data to get back below one point.

![Aggregate CPR forecasts around the 2022 regime break](artifacts/figures/e1_frozen_vs_retrained.png)

Estimating the model separately by regime and tracing partial-dependence S-curves
on a common reference population shows the lock-in era rotated the S-curve rather
than flattening it: the refinancing response above the money is more than twice as
steep as before the pandemic (30.5 vs 13.2 CPR at +150bp of incentive), while slow
discount speeds are carried by the level of prevailing rates.

![Model-implied S-curves by regime](artifacts/figures/e2_scurves.png)

Applied to the December 2025 cross-section, the model places 2023–24 vintage 7.0%
coupons near 25 CPR at unchanged rates and above 50 CPR in a 150bp rally; the 6.5s
roughly triple over the same interval.

![Projected CPR under parallel rate shifts](artifacts/figures/e3_scenarios.png)

A note for anyone modeling on this dataset: several monthly fields (balance,
modification flag, delinquency status, loan age) are blanked on a loan's removal
record, so used contemporaneously their missingness reveals the prepayment event.
Untreated, this produced out-of-sample AUCs of 0.9999 here. Details and the
corrections are in [docs/leakage-audit.md](docs/leakage-audit.md); each fix has a
regression test.

## Reproducing the results

Everything runs on an ordinary laptop (8GB RAM) plus the free Kaggle GPU tier for
the neural network. Total disk footprint is about 25GB.

```powershell
# 0. Register (free) at https://datadynamics.fanniemae.com and download the
#    quarterly zips (2018Q1 onward) into data/raw/fannie/
py -3.11 -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python scripts/run_external.py      # PMMS + FHFA HPI (public APIs)
.venv\Scripts\python scripts/make_layout.py "data/raw/fannie/<layout-file>.xlsx"
.venv\Scripts\python scripts/run_ingest.py        # raw -> parquet, 3GB RAM cap
.venv\Scripts\python scripts/run_cohorts.py       # full-population CPR ground truth
.venv\Scripts\python scripts/run_panel.py         # sampled hazard panel
.venv\Scripts\python scripts/run_models.py        # walk-forward logistic + GBM
.venv\Scripts\python scripts/run_experiments.py   # all experiments + robustness
# Neural net: scripts/export_kaggle.py, then notebooks/kaggle_nn_training.py on Kaggle
.venv\Scripts\pytest                              # 42 tests
```

## Layout

- `src/scurve/` — ingestion, cohort actuals, sampling, features, models, experiments
- `docs/paper/latex/` — the paper (LaTeX source and PDF)
- `docs/research/literature-survey.md` — literature review with verification notes
- `artifacts/figures/`, `artifacts/tables/` — every figure and number in the paper

## Data

The Fannie Mae Single-Family Loan Performance dataset is used under its own terms
and no loan-level data is redistributed here; rates and house-price series come
from FRED and FHFA.

Personal research project on public data. Not investment advice; views are my own
and not those of any employer.
