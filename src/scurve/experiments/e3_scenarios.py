"""E3: what does the current model imply for 2023-24 high coupons under rate
scenarios? One-month-ahead hazard annualized to CPR (documented simplification:
burnout and HPI held at latest observed values). Outputs: e3_scenarios.png, e3.json."""
import json

import numpy as np
import polars as pl

from ..features import FEATURES
from ..models.gbm import GBMHazard
from ..plotting import save_fig, scurve_chart

SHIFTS = [-150, -100, -50, 0, 50, 100]
TARGET = {(2023, 6.0), (2023, 6.5), (2023, 7.0),
          (2024, 6.0), (2024, 6.5), (2024, 7.0)}


def scenario_cpr(model, df: pl.DataFrame, shifts_bps=SHIFTS) -> pl.DataFrame:
    rows = []
    w = (df["weight"] * df["curr_upb"]).to_numpy()
    for s in shifts_bps:
        x = df.with_columns((pl.col("incentive_bps") - s).alias("incentive_bps"))
        p = model.predict_hazard(x)
        for (vy, cb) in sorted({(a, b) for a, b in
                                zip(df["vintage_year"], df["coupon_bucket"])}):
            m = ((df["vintage_year"] == vy) & (df["coupon_bucket"] == cb)).to_numpy()
            smm = float(np.average(p[m], weights=w[m]))
            rows.append({"shift_bps": s, "vintage_year": vy, "coupon_bucket": cb,
                         "smm": smm, "cpr": 1 - (1 - smm) ** 12})
    return pl.DataFrame(rows)


def run_e3(cfg: dict) -> None:
    lf = pl.scan_parquet(str(cfg["paths"]["panel_dir"] / "panel_*.parquet")) \
           .drop_nulls(subset=["incentive_bps", "mtm_ltv"])
    latest = lf.select(pl.col("month").max()).collect()["month"][0]

    # train on everything, score the latest cross-section of target cohorts
    full = lf.collect()
    model = GBMHazard(cfg["models"]["gbm"] | {"early_stopping_rounds": None,
                                              "n_estimators": 600})
    model.fit(full.select(FEATURES), full["y"].to_numpy(), full["weight"].to_numpy())

    is_target = pl.struct("vintage_year", "coupon_bucket").map_elements(
        lambda r: (r["vintage_year"], r["coupon_bucket"]) in TARGET,
        return_dtype=pl.Boolean)
    cross = full.filter((pl.col("month") == latest) & (pl.col("y") == 0) & is_target)
    del full
    out = scenario_cpr(model, cross)

    chart = out.with_columns(
        (pl.col("vintage_year").cast(pl.Utf8) + " " +
         pl.col("coupon_bucket").cast(pl.Utf8) + "s").alias("series"),
        pl.col("shift_bps").alias("incentive_bps"),   # x-axis reuse
    ).select("incentive_bps", "cpr", "series")
    fig = scurve_chart(chart, f"Projected CPR under parallel rate shifts (as of {latest})")
    fig.axes[0].set_xlabel("Parallel mortgage-rate shift (bps)")
    save_fig(fig, cfg["paths"]["figures_dir"], "e3_scenarios")

    cfg["paths"]["tables_dir"].mkdir(parents=True, exist_ok=True)
    with open(cfg["paths"]["tables_dir"] / "e3.json", "w", encoding="utf-8") as f:
        json.dump({"as_of": latest, "rows": out.to_dicts()}, f, indent=2)
    print(out.filter(pl.col("shift_bps").is_in([-100, 0])), flush=True)
