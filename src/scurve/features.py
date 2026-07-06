"""Feature expressions for the hazard panel.

Inputs are lower-case canonical columns produced by the panel builder:
loan_id, month, orig_rate, pmms30, pmms_orig, oltv, orig_upb, curr_upb,
hpi, hpi_orig, DLQ_STATUS. Frames must be sorted by (loan_id, month) before
window features (add_burnout) are applied.

Leakage rule: every feature at month t uses only information available at the
start of t (burnout is lagged; incentive uses month-t PMMS, observable
intra-month — acceptable for a monthly hazard, documented in the writeup).
"""
import polars as pl


def add_incentive() -> pl.Expr:
    return ((pl.col("orig_rate") - pl.col("pmms30")) * 100).alias("incentive_bps")


def add_sato() -> pl.Expr:
    return ((pl.col("orig_rate") - pl.col("pmms_orig")) * 100).alias("sato_bps")


def add_burnout() -> pl.Expr:
    """Cumulative positive incentive over months strictly before t (bps-months)."""
    return (
        pl.col("incentive_bps").clip(lower_bound=0)
        .cum_sum().shift(1).over("loan_id")
        .fill_null(0.0)
        .alias("burnout_bps")
    )


def add_mtm_ltv() -> pl.Expr:
    return (
        (pl.col("oltv") * (pl.col("curr_upb") / pl.col("orig_upb"))
         / (pl.col("hpi") / pl.col("hpi_orig")))
        .alias("mtm_ltv")
    )


def add_month_of_year() -> pl.Expr:
    return pl.col("month").str.slice(5, 2).cast(pl.Int8).alias("month_of_year")


def dlq_bucket() -> pl.Expr:
    """0 = current/unknown, 1 = 30dpd, 2 = 60+dpd."""
    d = pl.col("DLQ_STATUS").str.strip_chars()
    return (
        pl.when(d == "01").then(1)
        .when(d.str.contains(r"^(0[2-9]|[1-9]\d)$")).then(2)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("dlq_bucket")
    )


NUMERIC_FEATURES = [
    "incentive_bps", "sato_bps", "burnout_bps", "mtm_ltv", "loan_age",
    "orig_upb_log", "cscore_b", "dti", "oltv", "month_of_year",
]
CATEGORICAL_FEATURES = [
    "channel", "purpose", "prop", "occ_stat", "state", "num_bo_capped",
    "first_flag", "dlq_bucket", "mod_flag",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
