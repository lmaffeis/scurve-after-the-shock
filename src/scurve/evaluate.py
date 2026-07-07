"""Walk-forward evaluation: loan-level (AUC/logloss) and cohort-level CPR."""
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

from .dates import month_add, month_range


def walk_forward_splits(train_ends: list[str], horizon: int) -> list[tuple[str, list[str]]]:
    return [(te, month_range(month_add(te, 1), month_add(te, horizon)))
            for te in train_ends]


def loan_level_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y, p, sample_weight=w)),
        "logloss": float(log_loss(y, p, sample_weight=w)),
        "n": int(len(y)),
    }


def cohort_cpr_from_preds(df: pl.DataFrame) -> pl.DataFrame:
    """df: vintage_year, coupon_bucket, month, pred_hazard, weight, curr_upb.

    Predicted cohort SMM = UPB- and sample-weight-weighted mean hazard."""
    wcol = pl.col("weight") * pl.col("curr_upb")
    return (
        df.group_by(["vintage_year", "coupon_bucket", "month"])
        .agg(((pl.col("pred_hazard") * wcol).sum() / wcol.sum()).alias("pred_smm"),
             wcol.sum().alias("pred_upb_mass"))
        .with_columns((1 - (1 - pl.col("pred_smm")) ** 12).alias("pred_cpr"))
    )


def cpr_error_table(preds: pl.DataFrame, actuals: pl.DataFrame) -> pl.DataFrame:
    return (
        preds.join(actuals.select("vintage_year", "coupon_bucket", "month",
                                  "cpr", "begin_upb"),
                   on=["vintage_year", "coupon_bucket", "month"], how="inner")
        .with_columns((pl.col("pred_cpr") - pl.col("cpr")).alias("cpr_error"))
    )


def upb_weighted_mae(err: pl.DataFrame) -> float:
    """UPB-weighted mean absolute CPR error, in CPR points (x100).

    Non-finite errors (cohort-months with zero predicted UPB mass in the
    sample) are excluded rather than poisoning the aggregate."""
    w = err["begin_upb"].to_numpy()
    e = np.abs(err["cpr_error"].to_numpy())
    ok = np.isfinite(e) & np.isfinite(w)
    return float(100 * (e[ok] * w[ok]).sum() / w[ok].sum())
