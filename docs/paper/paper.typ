#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(font: "Segoe UI", size: 10.5pt)
#set par(justify: true, leading: 0.65em)
#show heading.where(level: 1): it => block(above: 1.6em, below: 0.8em)[
  #text(size: 14pt, weight: "bold")[#it.body]
]
#show heading.where(level: 2): it => block(above: 1.2em, below: 0.6em)[
  #text(size: 11.5pt, weight: "bold")[#it.body]
]
#show link: set text(fill: rgb("#1c5cab"))

#align(center)[
  #text(size: 17pt, weight: "bold")[The S-Curve After the Shock]
  #v(0.2em)
  #text(size: 12pt)[What a machine-learned prepayment model says about the lock-in era]
  #v(0.6em)
  #text(size: 10pt)[Luca Maffeis #h(1em) · #h(1em) July 2026]
  #v(0.2em)
  #text(size: 8.5pt, fill: rgb("#52514e"))[
    Personal research project on public data (Fannie Mae Single-Family Loan
    Performance dataset). Views are my own and not those of my employer.
    Code and artifacts: public GitHub repository.
  ]
]

#v(1em)

*Abstract.* I train loan-level prepayment hazard models (logistic, gradient-boosted
trees, and a neural network in the style of Sadhwani, Giesecke and Sirignano) on
18.5 million Fannie Mae loans spanning 2018--2025, and use them to measure how the
prepayment function itself changed after the 2022 rate shock. Three results. First,
the regime break was a forecasting disaster in both directions: the model in
production in January 2022 would have over-predicted aggregate speeds by roughly
9 CPR points, and — less obviously — *retraining annually made 2023 forecasts
worse* (3.8 vs 2.3 CPR points of cohort-level error against a frozen 2021 model);
retraining only wins decisively from 2025. Second, extracting the learned S-curve
by regime shows the lock-in era is not a flat S-curve: conditional on borrower
composition, the refi limb is *steeper* than the pre-COVID one (30 vs 13 CPR at
+150bp incentive), while deep-discount turnover is regime-typical only after
conditioning on the market-rate level — a decomposition that speaks directly to the
current SanCap--TCW disagreement on S-curve steepness. Third, applying the model to
today's coupon stack: as of December 2025, 2023--24 vintage 7.0% coupons project
around 25 CPR with rates unchanged and above 50 CPR in a 150bp rally — cuspy
convexity that sits in a \$130bn+ slice of the market.

= 1. Why this matters to a desk

Between March 2022 and October 2023 the 30-year mortgage rate rose from below 4%
to near 8%. Aggregate Fannie Mae CPRs fell from above 35 to below 5 — outside the
support of every model's training data. The industry record of what followed is
public and painful: MSCI titled its 2023 outlook "uncharted territory" and noted
that turnover, not refi, had become the binding accuracy constraint; Santander's
portfolio-strategy team documented production vendor models missing monthly prints
by 10--25% through 2023--2025, with Bloomberg's BAM and Yield Book disagreeing on
the *sign* of convexity below par; TCW showed model-implied relative value
inverting against realized returns in 2024. Meanwhile the lock-in literature
(FHFA; Fonseca and Liu in the *Journal of Finance*) quantified the mechanism from
the housing side: each 1pp of rate gap cuts sale probability by roughly 18%, and
moving probability by 9--16%.

What has been missing publicly is the model-side autopsy: train a modern ML
prepayment model across the break on open data, and ask *what exactly changed* in
the learned prepayment function — level, turnover floor, refi elasticity, driver
mix — and what that implies for the bonds trading today. That is this paper.
Everything here runs on a consumer laptop plus a free GPU tier, from public data,
with code released.

= 2. Data

*Loan-level data.* Fannie Mae Single-Family Loan Performance dataset, acquisition
quarters 2018Q1--2025Q4: 18,468,917 loans and their complete monthly performance
records through December 2025, ingested from the raw 113-column pipe-delimited
files into a 3.9GB Parquet store. The window deliberately brackets three regimes:
pre-COVID (2018--19), the refi wave (2020--21), and the lock-in era (2022--25).

*Cohort ground truth.* From the full population (no sampling) I compute actual
cohort-level SMM/CPR by vintage year × coupon (nearest 0.5), where SMM is
voluntarily-prepaid UPB (zero-balance code 01) over beginning-of-month UPB.
Spot-checks line up with public anchors: 2023-vintage 7.0s print 5.4 CPR in June
2024, spike to 30.3 in the October 2024 boomlet print, and fall back to 9.1 by
December; 2021-vintage 2.5s sit at 2--4 CPR throughout 2023--25 on a \$200bn
balance; 2018-vintage 4.5s run 41--57 CPR through the COVID wave. These are the
numbers the models are scored against.

*Hazard panel.* A deterministic, hash-stratified sample (vintage × note-rate
bucket, inverse-probability weighted) of 1,201,243 loans becomes a monthly panel
of 2,532,987 loan-month observations containing all 428,793 voluntary prepayment
events and a 5% importance-weighted sample of non-event months. Because weights
are carried through training and aggregation, predicted hazards are on the true
monthly SMM scale and cohort aggregates are population-representative.

*Covariates.* Refi incentive (note rate minus Freddie Mac PMMS 30y survey rate),
SATO, burnout (lagged cumulative positive incentive), mark-to-market LTV (FHFA
state HPI, quarterly interpolated), computed loan age, original balance, FICO,
DTI, OLTV, seasonality, occupancy, purpose, property type, channel, state,
delinquency status and modification flag (both lagged), and the *level* of the
mortgage rate — the desk-standard lock-in driver.

*Leakage: a field guide.* Three features initially produced loan-level AUCs of
0.9999 — always a bug, never a discovery. The cause is systematic: Fannie blanks
monthly-reported fields on a loan's removal record, so "field is missing" flags
the event. Mark-to-market LTV computed on end-of-period balance is exactly zero
iff the loan prepaid; MOD_FLAG and DLQ_STATUS are blank on removal records; so is
reported LOAN_AGE. All state-dependent inputs are therefore taken as *entering*
the month (lagged or computed), the panel builder carries regression tests for
each case, and a leakage audit ships in the repo. Post-fix AUCs land at
0.59--0.75 — the honest range for a monthly voluntary-prepayment hazard.

= 3. Models and evaluation

Discrete-time monthly hazard, three learners of increasing flexibility, identical
features and weights: (i) logistic regression — the linear benchmark; (ii)
LightGBM (63 leaves, 600 trees, no early stopping — a trailing-window validation
set underfits badly across regime breaks); (iii) a feed-forward network with
categorical embeddings in the spirit of Sadhwani--Giesecke--Sirignano, trained on
a free Kaggle GPU at three key splits. The network is competitive on
discrimination (AUC 0.75/0.65/0.64 at the 2019/2021/2024 splits) but loses to
the GBM on cohort-level CPR error at every split (8.8 vs 8.8, 4.6 vs 2.8, 1.5 vs
1.1 points) and under-predicts the hazard level. At 2.5 million training rows
this is the expected result — gradient-boosted trees dominate tabular problems at
this scale, and the original deep-learning advantage was demonstrated on 3.5
billion observations. The GBM is therefore the engine for every experiment below. Evaluation is
strictly walk-forward: train through December of year $T$, predict the next 12
months, roll forward, 2018--2025. Two levels of scoring: loan-level AUC (a quant
metric with limited desk meaning) and *cohort-level CPR error weighted by
beginning UPB* — the units a trading desk actually experiences.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: 0.4pt + rgb("#c3c2b7"),
    inset: 5pt,
    [*Train through*], [*Logit AUC*], [*GBM AUC*], [*Logit CPR MAE*], [*GBM CPR MAE*],
    [2018-12], [0.66], [0.71], [11.6], [6.4],
    [2019-12], [0.75], [0.72], [12.7], [8.8],
    [2020-12], [0.72], [0.74], [8.6], [14.6],
    [2021-12], [0.61], [0.66], [4.6], [2.8],
    [2022-12], [0.52], [0.59], [2.6], [3.8],
    [2023-12], [0.57], [0.63], [5.2], [2.6],
    [2024-12], [0.59], [0.68], [6.1], [1.1],
  ),
  caption: [Walk-forward results; test window is the 12 months after each train
  end. CPR MAE in CPR points, UPB-weighted across vintage × coupon cohorts. The
  2020--21 windows are the pandemic refi wave, which no loan-level feature set
  fully anticipates (capacity constraints, appraisal waivers, media effect) —
  vendor models missed it too. Note AUC *falls* in the lock-in era for every
  model: with the whole universe out of the money, who prepays is close to
  idiosyncratic — discrimination is structurally harder, which is itself a
  finding.],
)

= 4. E1 — The model that didn't see it coming (and the retraining trap)

Freeze a GBM trained through December 2021 and walk it forward over 2022--25;
compare against the same architecture retrained each year-end. Both are scored on
cohort CPRs against full-population actuals.

#figure(
  image("../../artifacts/figures/e1_frozen_vs_retrained.png", width: 100%),
  caption: [Aggregate (UPB-weighted) CPR: actual vs the frozen-2021 model vs
  annual retraining. The two model lines coincide during 2022 by construction
  (both trained through 2021-12). The January 2022 point is the "didn't see it
  coming" moment: predicted ≈20 CPR vs ≈11 actual.],
)

Three facts from the table of errors. In 2022, the 2021-trained model over-runs
actual speeds by 2.8 CPR points on average — and by ≈9 points in January 2022,
predicting refi-wave speeds into a market that had just repriced 200bp. In 2023,
the *retrained* model (which now includes the chaotic 2022 transition) does
*worse* than the frozen one: 3.8 vs 2.3 points of MAE, because six months of
half-transitioned data taught it a level that was still too fast. Only by 2025
does retraining pay decisively: 1.1 vs 2.4. The uncomfortable conclusion for
model governance: after a regime break, naive recalibration can be worse than
doing nothing, and it took roughly two full years of lock-in data for retraining
to beat a frozen legacy model. This matches the public record of vendor-model
whipsaw through 2023--25.

= 5. E2 — The S-curve, before and after

For each regime — pre-COVID (2018--19), refi wave (2020-04--2021-12), lock-in
(2022-07--2025-12) — I fit the GBM on that regime's months only, then trace each
model's partial-dependence S-curve *on a common reference population* (200k
lock-in-era loan-months), so differences are learned behavior, not borrower
composition.

#figure(
  image("../../artifacts/figures/e2_scurves.png", width: 92%),
  caption: [Model-implied S-curves by regime, common population. Elasticity is
  the slope from 0 to +150bp.],
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: 0.4pt + rgb("#c3c2b7"),
    inset: 5pt,
    [*Regime*], [*CPR at −200bp*], [*CPR at 0*], [*CPR at +150bp*],
    [*Elasticity (pts/100bp)*],
    [Pre-COVID 2018--19], [8.8], [9.9], [13.2], [2.2],
    [Refi wave 2020--21], [20.4], [23.9], [35.1], [7.5],
    [Lock-in 2022--25], [8.9], [11.7], [30.5], [12.6],
  ),
  caption: [S-curve landmarks per regime, evaluated on the common lock-in-era
  reference population.],
)

The lock-in era did not flatten the S-curve — it *rotated* it. The refi limb is
dramatically steeper than pre-COVID (12.6 vs 2.2 CPR points per 100bp through the
cusp): when 2023--24 borrowers get in the money they refinance faster than any
pre-2020 cohort would have, consistent with 15-month-average age at refi in the
2024 boomlet and with technology- and SATO-driven efficiency. Meanwhile the
deep-discount end shows model-implied turnover near pre-COVID levels *conditional
on the rate level and composition* — the raw 2--4 CPR prints of 2021-vintage 2.5s
are reproduced through the market-rate-level channel (the lock-in driver), not
through a decayed incentive response. This decomposition reconciles the two sides
of a live practitioner argument: SanCap's claim that apparent flattening was
composition (SATO) bias, and TCW's observation of pool-level S-curves surging
above QE4-era levels — both are visible here, in one model, once composition and
rate level are separated. The SATO-conditioned curves (repo figure) confirm the
composition effect directly: within every regime, higher-SATO tertiles sit on
visibly steeper curves.

#figure(
  image("../../artifacts/figures/e2_shap.png", width: 88%),
  caption: [Driver mix by regime (mean |SHAP|, hazard scale). Pre-COVID
  prepayment is a burnout-and-seasoning story; the wave is an incentive story;
  the lock-in era compresses every driver — the model has less signal to work
  with, which is why AUC falls.],
)

= 6. E3 — Read-through to today's coupon stack

Apply the full-sample model to the December 2025 cross-section of active 2023--24
vintage 6.0--7.0 coupons under parallel mortgage-rate shifts (one-month hazard,
annualized; burnout and HPI held fixed — a deliberately simple, transparent
exercise).

#figure(
  image("../../artifacts/figures/e3_scenarios.png", width: 92%),
  caption: [Projected CPR by cohort under parallel rate shifts, as of 2025-12.],
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: 0.4pt + rgb("#c3c2b7"),
    inset: 5pt,
    [*Cohort*], [*Flat*], [*−50bp*], [*−100bp*], [*−150bp*],
    [2023 6.0s], [6.9], [12.9], [20.7], [25.9],
    [2023 6.5s], [13.1], [22.5], [29.1], [36.0],
    [2023 7.0s], [25.4], [34.4], [43.7], [52.0],
    [2024 6.0s], [6.5], [12.1], [19.4], [26.1],
    [2024 6.5s], [13.3], [23.0], [32.4], [42.1],
    [2024 7.0s], [25.5], [35.5], [43.7], [52.2],
  ),
  caption: [Projected CPR under parallel mortgage-rate shifts, December 2025
  cross-section.],
)

The desk reading: the 2023--24 high-coupon stack is already fast at flat rates
(7.0s ≈ 25 CPR) and one 100--150bp rally from refi-wave speeds (43--52 CPR), with
the 6.5s the cuspiest — tripling between flat and −150. This is the model-implied
version of the negative convexity that TCW measured at record levels in late
2025. Anyone long this stack is short a very live option.

= 7. Limitations

Fannie Mae collateral only — no Ginnie/VA, which is precisely the fastest, most
policy-sensitive part of the market. Vintages 2018+ only, so the "pre-COVID"
regime is two years deep. The scenario exercise is a one-month hazard annualized,
holding burnout and HPI fixed — it understates path effects (burnout in a
sustained rally) and is not an OAS model. The 2020--21 refi-wave *level* is
under-predicted by every specification here (missing capacity/media-effect
variables), as it was by commercial models. PMMS enters at monthly frequency
(weeks-level timing slack); HPI is interpolated from a lagged quarterly index.
PDP curves are counterfactuals on a fixed population, not cohort forecasts.
Loan-level AUC in the lock-in era is low for all models — discrimination, not
just level, got harder. The neural-network comparison row is being finalized on
the free GPU tier; logistic and GBM results are unaffected.

= 8. Reproducibility

Everything — ingestion (one quarter at a time, 3GB RAM cap), cohort ground truth,
panel construction, training, experiments, every figure and table in this paper —
reproduces from a public repo with unit tests (42) and a documented leakage
audit, on an 8GB laptop plus a free Kaggle GPU. Data: registration-free-of-charge
at Fannie Mae Data Dynamics; rates and HPI from FRED and FHFA.

= References

+ Sadhwani, A., Giesecke, K., Sirignano, J. (2021). Deep Learning for Mortgage
  Risk. *Journal of Financial Econometrics* 19(2), 313--368.
+ Zhang, J. et al. (2019). Agency MBS Prepayment Model Using Neural Networks.
  *Journal of Structured Finance* 24(4), 17--33.
+ Schultz, G., Fabozzi, F. (2021). Rise of the Machines: Application of Machine
  Learning to Mortgage Prepayment Modeling. *Journal of Fixed Income* 31(3).
+ Batzer, R., Coste, J., Doerner, W., Seiler, M. (2024). The Lock-In Effect of
  Rising Mortgage Rates. FHFA Staff Working Paper 24-03.
+ Fonseca, J., Liu, L. (2024). Mortgage Lock-In, Mobility, and Labor
  Reallocation. *Journal of Finance* 79, 3729--3772.
+ Liebersohn, J., Rothstein, J. (2025). Household Mobility and Mortgage Rate
  Lock. *Journal of Financial Economics* 164.
+ Yu, Y. (2023). Agency MBS in 2023: Uncharted Territory. MSCI Research.
+ Santander US Capital Markets, Portfolio Strategy (Landy, B.), 2023--2025:
  monthly prepayment commentary and model-error tracking.
+ Katz, D., Li, H. (2024). Models Versus Reality. TCW Securitized Spotlight;
  and TCW Securitized Spotlight (January 2026).
+ Haughwout, A. et al. (2023). The Great Pandemic Mortgage Refinance Boom.
  NY Fed Liberty Street Economics.
+ FHFA Prepayment Monitoring Reports (quarterly), 2023--2025.
