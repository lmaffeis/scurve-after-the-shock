"""E1: frozen (trained <=2021-12) vs annually-retrained GBM, walk-forward on
2022+. Outputs: figure e1_frozen_vs_retrained.png, table e1.json."""
import json
from pathlib import Path

import polars as pl

from ..evaluate import cohort_cpr_from_preds, cpr_error_table, upb_weighted_mae
from ..plotting import error_timeseries, save_fig
from ..runner import run_walk_forward


def universe_cpr_series(err: pl.DataFrame) -> pl.DataFrame:
    w = pl.col("begin_upb")
    return (err.group_by("month")
            .agg(((pl.col("cpr") * w).sum() / w.sum()).alias("actual_cpr"),
                 ((pl.col("pred_cpr") * w).sum() / w.sum()).alias("pred_cpr"))
            .sort("month"))


def _cohort_errors(pred_files: list[Path], actuals: pl.DataFrame) -> pl.DataFrame:
    preds = pl.concat([pl.read_parquet(f) for f in pred_files])
    # a month can appear under several train_ends in expanding mode: keep latest
    preds = (preds.sort("train_end")
             .unique(subset=["loan_id", "month"], keep="last"))
    cohort = cohort_cpr_from_preds(preds)
    return cpr_error_table(cohort, actuals)


def run_e1(cfg: dict, actuals: pl.DataFrame) -> None:
    models_dir, fig_dir = cfg["paths"]["models_dir"], cfg["paths"]["figures_dir"]
    tab_dir = cfg["paths"]["tables_dir"]
    panel_glob = str(cfg["paths"]["panel_dir"] / "panel_*.parquet")
    frozen_dir = models_dir / "e1_frozen"

    # frozen: one GBM trained at frozen_train_end, predicting everything after
    if not list(frozen_dir.glob("preds_gbm_*.parquet")):
        run_walk_forward(panel_glob, cfg, actuals, frozen_dir, model_names=("gbm",),
                         train_ends=[cfg["eval"]["frozen_train_end"]], frozen=True)
    frozen_err = _cohort_errors(sorted(frozen_dir.glob("preds_gbm_*.parquet")), actuals)

    # retrained: reuse the expanding-window predictions for months > frozen end
    retrained_files = [f for f in sorted(models_dir.glob("preds_gbm_*.parquet"))
                       if f.stem.split("_")[-1] >= cfg["eval"]["frozen_train_end"]]
    retr_err = _cohort_errors(retrained_files, actuals)
    cutoff = cfg["eval"]["frozen_train_end"]
    frozen_err = frozen_err.filter(pl.col("month") > cutoff)
    retr_err = retr_err.filter(pl.col("month") > cutoff)

    fu, ru = universe_cpr_series(frozen_err), universe_cpr_series(retr_err)
    chart = pl.concat([
        fu.select("month", pl.col("actual_cpr").alias("value"))
          .with_columns(pl.lit("Actual").alias("series")),
        fu.select("month", pl.col("pred_cpr").alias("value"))
          .with_columns(pl.lit("Frozen model (trained through 2021)").alias("series")),
        ru.select("month", pl.col("pred_cpr").alias("value"))
          .with_columns(pl.lit("Retrained annually").alias("series")),
    ]).with_columns(pl.col("value") * 100)
    fig = error_timeseries(chart, "Aggregate CPR: the model that didn't see it coming")
    save_fig(fig, fig_dir, "e1_frozen_vs_retrained")

    by_year = {}
    for name, err in [("frozen", frozen_err), ("retrained", retr_err)]:
        by_year[name] = {
            "overall_mae_pts": upb_weighted_mae(err),
            "by_year": {yr: upb_weighted_mae(err.filter(pl.col("month").str.starts_with(yr)))
                        for yr in sorted({m[:4] for m in err["month"].to_list()})},
        }
    tab_dir.mkdir(parents=True, exist_ok=True)
    with open(tab_dir / "e1.json", "w", encoding="utf-8") as f:
        json.dump(by_year, f, indent=2)
    print(json.dumps(by_year, indent=2), flush=True)
