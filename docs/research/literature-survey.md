# Literature Survey: Machine Learning for Mortgage Prepayment Modeling (2020–2026)

*Compiled July 2026. Verification convention: [VERIFIED] = source page/PDF fetched and numbers confirmed; [SEARCH-VERIFIED] = confirmed from multiple independent citation records but primary page paywalled/blocked; [UNVERIFIED] = could not confirm, flagged explicitly.*

---

## 1. Deep Learning / ML for Mortgage Prepayment and Credit Risk

### 1.1 Anchor paper — Sadhwani, Giesecke & Sirignano, "Deep Learning for Mortgage Risk" [VERIFIED]

**Citation correction:** the published version is ***Journal of Financial Econometrics*, Vol. 19, Issue 2 (Spring 2021), pp. 313–368**, DOI 10.1093/jjfinec/nbaa025 (advance access July 2020). Journal author order is Sadhwani, Giesecke, Sirignano; the arXiv version (arXiv:1607.02470, v1 July 2016, v2 March 2018) lists Sirignano first.

- **Data:** loan-level origination + monthly performance for **>120 million US mortgages** (~93M prime + ~25M subprime), originated 1995–2014, **~3.5 billion loan-month observations**, ~70% of all US originations in the period, 30,000+ zip codes, **272 explanatory variables** including zip-level time-varying macro covariates. Vendor is CoreLogic per the working paper [SEARCH-VERIFIED].
- **Method:** discrete-time multi-state transition model over **7 states** (current, 30/60/90+ dpd, foreclosure, REO, paid off); transition probabilities from a **feedforward net with 5 hidden layers of 140–200 ReLU units**, softmax output, GPU-trained, CV-tuned.
- **Headline findings:** large out-of-sample gains over logistic regression in likelihood and classification; **prepayment is the transition with the strongest nonlinear effects**; state/local **unemployment has the greatest explanatory power** of all variables; DNN-selected portfolios realize lower prepayment/delinquency than logistic-selected ones. Exact AUC table values not extracted [UNVERIFIED at number level].
- **URLs:** https://academic.oup.com/jfec/article/19/2/313/6329869 · https://arxiv.org/abs/1607.02470 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2799443

### 1.2 Key follow-ups and related deep-learning work

| Paper | Venue | Data | Method | Headline result | URL |
|---|---|---|---|---|---|
| Zhang, Zhao, Zhang, Teng, Lin & Li (2019), "Agency MBS Prepayment Model Using Neural Networks" | *J. Structured Finance* 24(4), 17–33 [VERIFIED via MSCI PDF] | Agency 30yr FRM pool-level prepay data ~2003–2018; trained on 10% of pools | Deep feedforward NN vs MSCI "human model" | Accurate out-of-time CPR tracking 2016–18 trained only pre-2016; follow-up MSCI blog (2020): **avg abs error 0.1 SMM vs 0.3 SMM** for the traditional model | https://jsf.pm-research.com/content/24/4/17 |
| Kvamme, Sellereite, Aas & Sjursen (2018), CNN mortgage default | *Expert Systems with Applications* 102, 207–217 [SEARCH-VERIFIED] | DNB (Norway) transaction time series | CNN on account-balance series (+RF ensemble) | **AUC 0.918 (CNN), 0.926 (CNN+RF)** | https://www.sciencedirect.com/science/article/abs/pii/S0957417418301179 |
| Babaev et al. (2019), "E.T.-RNN" | KDD '19 applied track [VERIFIED] | European bank credit-card transaction histories | GRU encoder on raw transaction sequences | Beats classical credit scoring on AUC; deployed | https://arxiv.org/pdf/1911.02496 |
| Albanesi & Vamossy (2019), "Predicting Consumer Default" | NBER WP 26165 [VERIFIED] | Experian ~1% US credit-file sample | DNN + gradient boosting hybrid | Consistently beats conventional credit scores | https://www.nber.org/papers/w26165 |
| Blumenstock, Lessmann & Seow (2022), deep survival/competing risks | *J. Operational Research Society* 73(1), 26–38 [SEARCH-VERIFIED] | Large US mortgage dataset, competing risks default+prepay | DeepHit, random survival forests vs Cox PH / Fine–Gray | ML models beat statistical benchmarks for **both default and prepayment**, robust in calm and stressed periods | https://www.tandfonline.com/doi/abs/10.1080/01605682.2020.1838960 |
| Kündig & Sigrist (2024) | arXiv:2410.02846 [VERIFIED] | Large US mortgage credit dataset | Tree boosting + latent spatio-temporal Gaussian process | More accurate loan-level PDs and portfolio loss distributions than linear hazards | https://arxiv.org/abs/2410.02846 |

### 1.3 Gradient boosting vs neural nets for prepayment

- **Schultz & Fabozzi (2021/22), "Rise of the Machines: Application of Machine Learning to Mortgage Prepayment Modeling," *Journal of Fixed Income* 31(3), 6–19** [SEARCH-VERIFIED; paywalled]. Agency RMBS loan/pool-level data; **gradient boosted classifier** replacing the traditional modular S-curve model. https://www.pm-research.com/content/iijfixinc/31/3/6
- **Liu & Liang (2024/25), *Journal of Risk* 27(2), 14–43** [VERIFIED]: ensemble ML on origination-time-only variables predicts prepayment; note rate, LTV, credit score dominate. https://www.risk.net/journal-of-risk/7960705/the-prediction-of-mortgage-prepayment-risks-in-the-early-stages-of-loan-origination-a-machine-learning-approach
- Assessment: **no rigorous open head-to-head GBM-vs-NN benchmark specifically for agency prepayment exists**; the practitioner split is MSCI (NN, pool-level) vs Schultz–Fabozzi/RiskSpan (GBM, loan-level), both paywalled or proprietary.
- A cited "Davis, Machine Learning for Prepayment" **appears not to exist** — treat as misremembered.

### 1.4 Sequence models (LSTM/transformer)

- Ojha & Lee (2021), *Digital Finance* 3, 249–271: ~1.5M **Fannie Mae** loans 2005–09; LR/RF/MLP/CNN/RNN/LSTM comparison through the crisis. https://link.springer.com/article/10.1007/s42521-021-00036-4
- Huang & Yang (2024), arXiv:2501.00034 [VERIFIED]: **Fannie Mae data 2012–2022**, sequence models; shorter training windows + fewer features beat longer/more ("feature redundancy paradox"). https://arxiv.org/abs/2501.00034
- Yang et al. (2025), arXiv:2508.00415 [VERIFIED]: **Freddie Mac data, 44 cohorts**, ResE-BiLSTM + SHAP; default, not prepayment. https://arxiv.org/abs/2508.00415
- **No dedicated transformer-for-agency-prepayment paper surfaced** — a genuine open niche.

### 1.5 Fannie/Freddie public loan-level dataset papers (ML)

arXiv:2602.00120 (2026) — Fannie Mae 2023Q1–2024Q4, AutoML default benchmark [SEARCH-VERIFIED]; Kanimozhi et al. (IEEE ICSTSN 2023) — Freddie Mac prepayment, logistic (minor venue) [SEARCH-VERIFIED]. Pattern: public GSE datasets dominate recent academic ML work, **but overwhelmingly for default, not prepayment**.

---

## 2. The Mortgage Rate Lock-In Effect (Post-2022)

### 2.1 Batzer, Coste, Doerner & Seiler — "The Lock-In Effect of Rising Mortgage Rates" [VERIFIED]

- **Venue:** FHFA Staff Working Paper 24-03 (March 2024; revised extended-sample version on SSRN late 2024). Still a working paper as of mid-2026.
- **Data:** ~50 million active US mortgages (National Mortgage Database) joined with property transaction records, 1998–2024Q2 in the revision.
- **Headline results:** for each 1 pp that market rates exceed origination rate, **probability of sale falls 18.1%**; lock-in caused a **57% reduction in fixed-rate-mortgage home sales in 2023Q4**; prevented **~1.33M sales 2022Q2–2023Q4** (original) / **~1.72M through 2024Q2** (revision); supply reduction **raised home prices +5.7%** (original) / **+7.0%** (revision); average PV of locked-in rate benefit **~$49,050 per borrower**.
- **URLs:** https://www.fhfa.gov/research/papers/wp2403 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5021709

### 2.2 Fonseca & Liu — "Mortgage Lock-In, Mobility, and Labor Reallocation" [VERIFIED]

- **Venue:** *Journal of Finance*, Vol. 79 (2024), pp. 3729–3772, DOI 10.1111/jofi.13398.
- **Data:** Equifax CRISM merged with ICE/McDash servicing + HMDA; 56M+ loans originated 1992–present.
- **Headline results:** a **1 pp deepening of lock-in reduces moving probability by 9% overall, 16% during 2022–2024**.
- **URLs:** https://onlinelibrary.wiley.com/doi/10.1111/jofi.13398 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4399613

### 2.3 Other key lock-in papers

| Paper | Venue | Headline magnitudes | URL |
|---|---|---|---|
| Liebersohn & Rothstein, "Household Mobility and Mortgage Rate Lock" | **JFE Vol. 164 (2025), 1–13**; NBER w32781 [VERIFIED] | Mobility of mortgaged households **−16%** in 2022–23; **~$20B deadweight loss** | https://www.nber.org/papers/w32781 |
| Aladangady, Krimmel & Scharlemann, "Locked In: Mobility, Market Tightness, and House Prices" | FEDS WP 2024-088, rev. May 2025 [VERIFIED abstract] | Lock-in explains **44%** of the 2021→2022 decline in moves; time-on-market −29%; prices +8% | https://www.federalreserve.gov/econres/feds/files/2024088r1pap.pdf |
| Fonseca, Liu & Mabille, "Unlocking Mortgage Lock-In" | NBER w35237 (2026); SSRN 4874654 | "Missing downsizers" offset ~1/3 of rate-driven price decline; ~40% drop in existing-home sales 2022→2024 | https://www.nber.org/papers/w35237 |
| Gerardi, Qian & Zhang, "Mortgage Lock-in, Lifecycle Migration…" (2024) | SSRN 4933879 | Young buyers disproportionately hurt. **Counterfactual magnitudes conflict across drafts — [UNVERIFIED], check current draft before citing** | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4933879 |
| Gerardi, Qian & Zhang, "Mortgage Lock-in: A Review of the Literature" (May 2026) | SSRN 6820819 [partially UNVERIFIED — 403] | Best single literature review for this area | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6820819 |

Foundational pre-2022: Ferreira, Gyourko & Tracy (JUE 2010 / NBER w17405) — each $1,000 of extra real annual mortgage cost lowers mobility ~12%. NY Fed Liberty Street (May 2024): by end-2023 **~70% of outstanding mortgages were ≥3 pp below prevailing rates**. No peer-reviewed quantification of the 2025–26 lock-in unwind exists yet.

**Note for prepayment modeling:** this literature measures the **turnover component** (sales/moves), not refi response — the rate-gap elasticities (−18.1%/pp FHFA; −9%/pp Fonseca-Liu) are directly usable as turnover-suppression priors.

---

## 3. Prepayment Model Failure / Regime Shift Post-2022

### 3.1 The speed collapse and the out-of-sample problem

- **Aggregate CPRs at historic lows:** Feb 2023 1-month CPR: FN 4.2, FH 4.0, GN 5.2 (Ginnie Mae Global Markets Analysis, data by Recursion) [VERIFIED]. https://www.ginniemae.gov/data_and_reports/reporting/Documents/global_market_analysis_mar23.pdf
- **MSCI (Yihai Yu, Feb 2023), "Agency MBS in 2023: Uncharted Territory"** [VERIFIED]: 2022 mortgage rates jumped 378 bp to ~7%; 30yr CPR fell from >35 to below 5; explicitly frames post-2022 as an out-of-sample modeling problem; turnover replaced refi as the dominant model-accuracy driver. https://www.msci.com/research-and-insights/blog-post/agency-mbs-in-2023-uncharted-territory
- **NY Fed (Haughwout et al., May 2023), "The Great Pandemic Mortgage Refinance Boom"** [VERIFIED]: ~14M mortgages refinanced 2020Q2–2021Q4 (~1/3 of balances), $430B equity extracted. https://libertystreeteconomics.newyorkfed.org/2023/05/the-great-pandemic-mortgage-refinance-boom/
- **FEDS Notes (Sept 2024)** [VERIFIED]: >90% of the Fed's ~$2.3T MBS in coupons <4%; paydowns ~$18B/mo vs $35B cap. https://www.federalreserve.gov/econres/notes/feds-notes/the-evolution-of-the-federal-reserves-agency-mbs-holdings-20240920.html

### 3.2 Documented model errors (practitioner, month-by-month) [VERIFIED — SanCap/Santander Portfolio Strategy, Brian Landy]

- **Nov 2023:** "MBS entered uncharted prepayment territory"; BAM and Yield Book disagree on the **sign** of convexity below par; both underpredicted discount speeds. https://portfolio-strategy.apsec.com/2023/11/03/comparing-convexity-in-the-bam-and-yield-book-models/
- **Jan 2025:** Yield Book production model **10.8% too fast** on December speeds. https://portfolio-strategy.apsec.com/2025/01/10/fewer-refinances-slow-mbs-speeds-in-december/
- **Mar 2025:** February upside surprise — YB production off 11%, YB experimental off 21.3%, Ginnie models off ~15–25%. https://portfolio-strategy.apsec.com/2025/03/07/prepayment-speeds-surprise-to-the-upside-in-february/
- **Jun 2025:** "Bloomberg's BAM model failed to track the turnover pickup." https://portfolio-strategy.apsec.com/2025/06/06/refinancing-slows-as-turnover-accelerates/
- **Dec 2025:** October 2025 speeds "caught many market participants by surprise." https://portfolio-strategy.apsec.com/2025/12/19/lessons-learned-from-agency-mbs-in-2025/

**TCW (Katz & Li, Aug 2024), "Models Versus Reality"** [VERIFIED]: dealer/vendor model relative-value signals inverted vs realized returns (FN 2.0s/2.5s delivered ~1.5% excess return over three months against model preference for premiums). https://www.tcw.com/insights/2024/2024-08-29-securitized-spotlight

### 3.3 Burnout and the 2020–21 low-coupon universe

- **97.69% of the agency universe out-of-the-money** at end-2023; 2020–21 production in 2s/2.5s with gross WACs 2.75–3.13% (LSEG/Yield Book, Jan 2024) [VERIFIED]. https://www.lseg.com/en/insights/data-analytics/us-mbs-finishes-up-2023-strong
- FHFA Prepayment Monitoring Report Q1 2024 [VERIFIED]: even fastest-quartile 2021-cohort 2.5s/3s ran only **3–6 CPR** (3-month). https://www.fhfa.gov/sites/default/files/2024-05/Prepayment-Monitoring-Report_2024Q1.pdf
- Practitioner "cuspy" coupons in 2024–25: **5.5s–6.0s**, where model disagreement is largest (SanCap).
- Theory: Lesniewski (2026), arXiv:2603.12422 — burnout decomposed as individual hazard drift minus survival-selection variance in heterogeneous Cox models. https://arxiv.org/abs/2603.12422

### 3.4 High-coupon 2023–24 vintages as rates fell

- **Sep–Oct 2024 refi boomlet** (~6.1% 30yr rates): ~300,000 refis in two months; VA nearly one-third; >75% of rate/term refis from 2023–24 originations; **average age at refi 15 months** (ICE data via National Mortgage News, Dec 2024) [VERIFIED]. https://www.nationalmortgagenews.com/news/why-this-years-refi-boomlet-hints-at-mortgage-revival
- **Coupon-level speeds:** FHFA PMR Q1 2024 fastest-quartile 3M CPR — 2023-cohort 6.5s **13.2**, 7s **19.8**, ~7.5-WAC **27.1**. VA extremes: VA 6.5s approaching **80 CPR** in the Jan–Feb 2024 mini-wave; 2023-vintage VA 7s >80 CPR early 2024 then ~60 CPR late 2024 despite lower rates — **rapid burnout between the two 2024 waves** (SanCap) [VERIFIED]. MSCI (May 2024): VA S-curve "significantly steeper" than FHA/GSE. https://www.msci.com/research-and-insights/blog-post/high-speeds-in-slow-lanes-a-deep-dive-into-ginnie-mae-va-loans
- **S-curve shape controversy (central to E2):** SanCap (June 2024), "Don't be fooled by flatter S-curves" — apparent post-pandemic flattening is **composition bias (high SATO), not behavior**; controlling for SATO recovers pandemic-like steepness. https://portfolio-strategy.apsec.com/2024/06/07/dont-be-fooled-by-flatter-s-curves/ — But SanCap (Dec 2025) finds 2024/2025 S-curves still slower than 2020–21, while TCW (Jan 2026) reports **November 2025 pool-level S-curves surging above QE4-era levels** (fastest FN 6.5 pool 73 CPR; current-coupon convexity −3.3, most negative on record). Reconcilable (cohort vs pool level, SATO controls) — a live disagreement our model can adjudicate. https://www.tcw.com/insights/2026/2026-01-09-securitized-spotlight
- **Not found:** no public BIS/academic paper on the 2022 prepayment regime break per se; no verifiable dealer desk quotes; no dedicated Recursion post on the 2024 wave.

---

## 4. Interpretability of ML Prepayment Models

**(i) Vendor NN models validated via model-implied response curves:**
- **MSCI/Zhang et al. (JSF 2019)** [VERIFIED]: cohort-based sensitivity analysis recovers S-curves by loan size, inverted S-curve (media effect), burnout, seasonality, turnover-vs-refi differences. The canonical "open the black box by plotting what it learned" prepayment paper.
- **Sadhwani–Giesecke–Sirignano (JFEC 2021)**: PDP-style sensitivity/interaction plots; prepayment sensitivity to incentive "varies significantly, both in magnitude and sign."

**(ii) SHAP/PDP on tree models for prepayment:**
- **Zanders (Schepers & van Hees, Apr 2025)** [VERIFIED]: XGBoost CPR model at a Dutch bank with SHAP for validation/regulatory acceptance. https://zandersgroup.com/en/revealing-new-insights-through-machine-learning-an-application-in-prepayment-modelling/
- **Fernandez (2026), SSRN 6470598** [SEARCH-VERIFIED]: 148,938 Freddie Mac records; LR/RF/GBM vs Cox; SHAP recovers incentive, seasoning, credit score as top drivers; GBM AUC 0.776, Cox C-index 0.676. **No regime contrast.** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6470598
- Theses: Szolnoki (Tilburg 2021, Freddie Mac survival forests); Ewing (Erasmus 2022, PDP/ICE on Dutch data); Saito (TU Delft 2018, RF+PDP).

**(iii) SHAP on mortgage default (methodological templates on GSE data):**
- Bank of England SWP 816 (Bracke et al., 2019) — Shapley-based QII on UK mortgage default. https://www.bankofengland.co.uk/working-paper/2019/machine-learning-explainability-in-finance-an-application-to-default-risk-analysis
- NVIDIA (2020) — XGBoost + SHAP TreeExplainer on 11.2M Fannie Mae public loans (AUC ~0.90). https://developer.nvidia.com/blog/explaining-and-accelerating-machine-learning-for-loan-delinquencies/
- Kim & Shin (IJSPM 2021); Ozturkkal & Wahlstrøm (*Computational Economics* 2025); Purohit & Verma (JRFM 2024).

**Honest negatives:** no model-distillation-for-prepayment paper; GitHub search for "prepayment SHAP" repos returned zero results; formal journal papers doing SHAP specifically on prepayment (vs default) are scarce.

---

## 5. Gap Check

**Conclusion: no public study combines all three elements. Confidence ~85%** (~20 query formulations across arXiv full sweep, SSRN, Semantic Scholar, GitHub API, Kaggle, thesis repositories).

- **(a) ML on public GSE data spanning pre/post-2022:** samples almost always end 2020–2022. Only one anonymous GitHub repo (Apr 2026: Fannie 2019 vintages through end-2024, Cox/logistic — no ML/SHAP, no 2023–24 vintages) spans the lock-in era.
- **(b) Interpretability showing the regime shift:** documented by practitioners on proprietary models — never SHAP/PDP on a learned ML function with public code. Fernandez (2026) is nearest but has no regime contrast (data span [UNVERIFIED] — the biggest caveat).
- **(c) Cohort-level CPR forecasting for 2023–24 high-coupon vintages:** exists only inside proprietary vendor/dealer models.

**Supporting signals:** Joshi (2025, SSRN 5341000) flags interpretability as the open challenge in MBS AI; MSCI's 2019 paper flags regime shift as the known weakness of NN prepay models — no public post-2022 recalibration study followed.

**Residual blind spots:** SSRN full texts (403s), paywalled JFI/JSF/Journal of Risk content, non-indexed 2025–26 master's theses.
