"""Robustness appendix: calibration, cohort-level fit, sampling sensitivity.

Outputs: r1_calibration.png, r2_cohort_scatter.png, robustness.json.
The SATO-bucket S-curve robustness figure is produced by e2 (e2_scurves_by_sato).
"""
import json

import polars as pl

from ..evaluate import (cohort_cpr_from_preds, cpr_error_table,
                        loan_level_metrics, upb_weighted_mae)
from ..features import FEATURES
from ..models.gbm import GBMHazard
from ..plotting import INK_2, MUTED, SERIES, _base, save_fig
from ..sample import _stable_frac

SPLIT = "2024-12"   # largest training set; test = 2025


def calibration_table(preds: pl.DataFrame, n_bins: int = 10) -> pl.DataFrame:
    """Weighted reliability: decile of predicted hazard -> mean pred vs
    weighted empirical event rate."""
    q = preds.select(
        pl.col("pred_hazard"), pl.col("y").cast(pl.Float64), pl.col("weight"))
    edges = [q["pred_hazard"].quantile(i / n_bins) for i in range(1, n_bins)]
    binned = q.with_columns(
        pl.col("pred_hazard").cut(edges, labels=[str(i) for i in range(n_bins)])
        .alias("bin"))
    return (binned.group_by("bin")
            .agg(((pl.col("pred_hazard") * pl.col("weight")).sum()
                  / pl.col("weight").sum()).alias("mean_pred"),
                 ((pl.col("y") * pl.col("weight")).sum()
                  / pl.col("weight").sum()).alias("empirical_rate"),
                 pl.len().alias("n"))
            .sort("mean_pred"))


def thin_nonevents(panel: pl.DataFrame, keep_frac: float) -> pl.DataFrame:
    """Further deterministic thinning of already-sampled non-events, with
    weights scaled by 1/keep_frac (salted hash so the cut is independent
    of the original 5% draw)."""
    keyed = panel.with_columns(
        (pl.col("loan_id") + "|" + pl.col("month") + "|salt")
        .map_elements(_stable_frac, return_dtype=pl.Float64).alias("_h"))
    kept = keyed.filter((pl.col("y") == 1) | (pl.col("_h") < keep_frac))
    return kept.with_columns(
        pl.when(pl.col("y") == 0)
        .then(pl.col("weight") / keep_frac)
        .otherwise(pl.col("weight")).alias("weight")).drop("_h")


def run_robustness(cfg: dict, actuals: pl.DataFrame) -> None:
    fig_dir, tab_dir = cfg["paths"]["figures_dir"], cfg["paths"]["tables_dir"]
    models_dir = cfg["paths"]["models_dir"]
    out = {}

    # --- R1: calibration of the saved GBM predictions (test year 2025) ---
    preds = pl.read_parquet(models_dir / f"preds_gbm_{SPLIT}.parquet")
    cal = calibration_table(preds)
    fig, ax = _base(figsize=(7, 7))
    lim = float(cal["empirical_rate"].max()) * 1.15 * 100
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1, linestyle="--")
    ax.plot(cal["mean_pred"] * 100, cal["empirical_rate"] * 100,
            marker="o", markersize=6, linewidth=2, color=SERIES[0])
    ax.set_xlabel("Mean predicted monthly hazard (%)", color=INK_2)
    ax.set_ylabel("Weighted empirical event rate (%)", color=INK_2)
    ax.set_title("Calibration by predicted-hazard decile (test year 2025)",
                 loc="left", fontweight="bold")
    save_fig(fig, fig_dir, "r1_calibration")
    out["calibration"] = cal.with_columns(pl.col("bin").cast(pl.Utf8)).to_dicts()

    # --- R2: cohort-level actual vs predicted scatter ---
    cohort = cohort_cpr_from_preds(preds)
    err = cpr_error_table(cohort, actuals).filter(pl.col("begin_upb") > 1e9)
    fig, ax = _base(figsize=(7, 7))
    top = float(max(err["cpr"].max(), err["pred_cpr"].max())) * 100 * 1.05
    ax.plot([0, top], [0, top], color=MUTED, linewidth=1, linestyle="--")
    ax.scatter(err["cpr"] * 100, err["pred_cpr"] * 100,
               s=(err["begin_upb"] / err["begin_upb"].max() * 220).to_list(),
               alpha=0.45, color=SERIES[0], edgecolors="none")
    ax.set_xlabel("Actual cohort CPR (%)", color=INK_2)
    ax.set_ylabel("Predicted cohort CPR (%)", color=INK_2)
    ax.set_title("Cohort-month fit, vintage x coupon (test year 2025; "
                 "marker size = cohort UPB)", loc="left", fontweight="bold")
    save_fig(fig, fig_dir, "r2_cohort_scatter")
    out["cohort_fit"] = {"n_cohort_months": err.shape[0],
                         "upb_weighted_mae_pts": upb_weighted_mae(err)}

    # --- R3: sensitivity to the non-event downsampling ratio ---
    lf = pl.scan_parquet(str(cfg["paths"]["panel_dir"] / "panel_*.parquet")) \
           .drop_nulls(subset=["incentive_bps", "mtm_ltv"])
    tr_full = lf.filter(pl.col("month") <= SPLIT).collect()
    te = lf.filter((pl.col("month") > SPLIT)).collect()
    results = {}
    for label, frac in [("base_5pct", None), ("thinned_2.5pct", 0.5)]:
        tr = tr_full if frac is None else thin_nonevents(tr_full, frac)
        model = GBMHazard(cfg["models"]["gbm"])
        model.fit(tr.select(FEATURES), tr["y"].to_numpy(), tr["weight"].to_numpy())
        p = model.predict_hazard(te)
        ll = loan_level_metrics(te["y"].to_numpy(), p, te["weight"].to_numpy())
        pp = te.select("loan_id", "month", "y", "weight", "curr_upb",
                       "coupon_bucket", "vintage_year").with_columns(
            pl.Series("pred_hazard", p))
        mae = upb_weighted_mae(cpr_error_table(cohort_cpr_from_preds(pp), actuals))
        results[label] = {"train_rows": tr.shape[0], "auc": round(ll["auc"], 4),
                          "cpr_mae_pts": round(mae, 3)}
        print("R3", label, results[label], flush=True)
    out["sampling_sensitivity"] = results

    tab_dir.mkdir(parents=True, exist_ok=True)
    with open(tab_dir / "robustness.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["sampling_sensitivity"], indent=2), flush=True)
