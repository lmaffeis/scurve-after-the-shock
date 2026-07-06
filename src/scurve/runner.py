"""Walk-forward training orchestration over the panel parquet store."""
import json
from pathlib import Path

import polars as pl

from .dates import month_add, month_range
from .evaluate import (cohort_cpr_from_preds, cpr_error_table,
                       loan_level_metrics, upb_weighted_mae, walk_forward_splits)
from .models.baseline import LogisticHazard
from .models.gbm import GBMHazard


def split_panel(lf: pl.LazyFrame, train_end: str, test_months: list[str]):
    train = lf.filter(pl.col("month") <= train_end)
    test = lf.filter(pl.col("month").is_in(test_months))
    return train, test


def _xyw(df: pl.DataFrame):
    return df, df["y"].to_numpy(), df["weight"].to_numpy()


def make_model(name: str, cfg: dict):
    if name == "logistic":
        return LogisticHazard()
    if name == "gbm":
        return GBMHazard(cfg["models"]["gbm"])
    raise ValueError(name)


def run_walk_forward(panel_glob: str, cfg: dict, actuals: pl.DataFrame,
                     out_dir: Path, model_names=("logistic", "gbm"),
                     train_ends: list[str] | None = None,
                     frozen: bool = False) -> dict:
    """If frozen=True, train once at train_ends[0] and predict ALL later months."""
    lf = pl.scan_parquet(panel_glob).drop_nulls(subset=["incentive_bps", "mtm_ltv"])
    train_ends = train_ends or cfg["eval"]["train_ends"]
    horizon = cfg["eval"]["horizon_months"]
    splits = walk_forward_splits(train_ends, horizon)
    if frozen:
        last = lf.select(pl.col("month").max()).collect()["month"][0]
        splits = [(train_ends[0], month_range(month_add(train_ends[0], 1), last))]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in model_names:
        rows = []
        for train_end, test_months in splits:
            tr_lf, te_lf = split_panel(lf, train_end, test_months)
            tr, ytr, wtr = _xyw(tr_lf.collect())
            te, yte, wte = _xyw(te_lf.collect())
            if te.is_empty():
                continue
            model = make_model(name, cfg)
            # last 12 train months as early-stopping validation for GBM
            if name == "gbm":
                val_start = month_add(train_end, -11)
                mask = (tr["month"] >= val_start).to_numpy()
                model.fit(tr.filter(~pl.Series(mask)), ytr[~mask], wtr[~mask],
                          tr.filter(pl.Series(mask)), ytr[mask], wtr[mask])
            else:
                model.fit(tr, ytr, wtr)
            p = model.predict_hazard(te)
            preds = te.select("loan_id", "month", "y", "weight", "curr_upb",
                              "coupon_bucket", "vintage_year").with_columns(
                pl.Series("pred_hazard", p), pl.lit(train_end).alias("train_end"))
            preds.write_parquet(out_dir / f"preds_{name}_{train_end}.parquet")
            ll = loan_level_metrics(yte, p, wte)
            cohort = cohort_cpr_from_preds(preds)
            err = cpr_error_table(cohort, actuals)
            rows.append({"train_end": train_end, **ll,
                         "cpr_mae_pts": upb_weighted_mae(err)})
            print(name, train_end, rows[-1], flush=True)
        summary[name] = rows
    with open(out_dir / "walkforward_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
