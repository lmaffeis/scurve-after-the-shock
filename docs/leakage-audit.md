# Leakage Audit — Hazard Panel Features

Rule: every feature at month *t* must be knowable at the **start** of month *t*.
The target `y` is the voluntary prepayment event (Zero_Bal_Code == "01") **in** month *t*.

| Feature | Definition | Why no look-ahead |
|---|---|---|
| `incentive_bps` | (note rate − PMMS30 at *t*) × 100 | PMMS is a weekly survey observable intra-month; monthly average used. Borderline by a few weeks; acceptable for a monthly hazard and disclosed in the writeup. |
| `sato_bps` | (note rate − PMMS30 at origination month) × 100 | Fixed at origination. |
| `burnout_bps` | Σ over months *s* < *t* of max(incentive_s, 0) | Explicitly lagged: `cum_sum().shift(1)` per loan (see `features.add_burnout`); month *t*'s own incentive excluded. |
| `mtm_ltv` | OLTV × (UPB_entering / UPB_0) ÷ (HPI_t / HPI_orig) | UPB_entering is the balance **entering** the month (prior month's reported balance; see `panel.upb_entering`). Using the end-of-period balance would be a hard leak: Fannie reports CURRENT_UPB = 0 on the removal record, making LTV = 0 a perfect event predictor (caught in training — frozen AUC 0.99998 — and fixed; regression-tested in `test_upb_entering_no_zero_leak_at_event`). HPI is interpolated from the quarterly index published with a lag; disclosed as a limitation. |
| `loan_age` | Months since origination, **computed** from calendar months | Reported LOAN_AGE is blanked on removal records (null ⇔ event, the third leak caught at GBM AUC 0.9999); computed age cannot be blanked. Regression-tested (`test_computed_loan_age_never_null`). |
| `orig_upb_log`, `cscore_b`, `dti`, `oltv` | Origination attributes | Fixed at origination. |
| `month_of_year` | Calendar | Deterministic. |
| `channel`, `purpose`, `prop`, `occ_stat`, `state`, `num_bo_capped`, `first_flag` | Origination attributes | Fixed at origination. |
| `dlq_bucket` | Delinquency status **entering** *t* (lagged one month) | Fannie blanks DLQ_STATUS on the removal record, so the unlagged value's null-ness flags the event. Lagged in the panel builder; regression-tested (`test_state_fields_are_lagged_no_null_leak`). |
| `mod_flag` | Modification flag **entering** *t* (lagged one month) | Same removal-record blanking — unlagged, "mod_flag is null" separated events at AUC 0.9999. Caught in training, fixed, regression-tested. |

Structural guards:
- The event month's row carries the features as reported **in that month's record**; Fannie's performance file reports the status/UPB for the period, with removal fields (LAST_UPB, Zero_Bal_Code) populated on the final record. We never join future months backward.
- Walk-forward evaluation only: models are trained on months ≤ train_end and scored on strictly later months, so any residual within-month timing subtlety cannot inflate out-of-sample results across the train boundary.
- `burnout` correctness is unit-tested (`test_burnout_is_lagged_cumulative_positive_incentive`).

Known disclosed simplifications (also in the paper's Limitations):
1. Month-*t* PMMS in month-*t* incentive (weeks-level timing).
2. HPI interpolation uses the containing quarter (publication lag ignored).
3. `CURRENT_UPB` at *t* reflects the end-of-period balance for the reporting month; for the denominator of cohort SMM we use the **previous** month's balance (see `cohorts.py` lag), which is the industry convention.
