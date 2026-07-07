import polars as pl

from scurve.experiments.e1_frozen import universe_cpr_series


def test_universe_cpr_series_upb_weighted():
    err = pl.DataFrame({
        "month": ["2022-01", "2022-01"],
        "pred_cpr": [0.10, 0.30], "cpr": [0.05, 0.25],
        "begin_upb": [3e9, 1e9],
        "vintage_year": [2020, 2023], "coupon_bucket": [2.5, 6.5],
    })
    out = universe_cpr_series(err)
    row = out.filter(pl.col("month") == "2022-01")
    assert abs(row["actual_cpr"][0] - (0.05 * 3 + 0.25 * 1) / 4) < 1e-12
    assert abs(row["pred_cpr"][0] - (0.10 * 3 + 0.30 * 1) / 4) < 1e-12
