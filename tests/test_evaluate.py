import polars as pl

from scurve.evaluate import cohort_cpr_from_preds, cpr_error_table, walk_forward_splits


def test_walk_forward_splits():
    s = walk_forward_splits(["2021-12", "2022-12"], horizon=3)
    assert s[0] == ("2021-12", ["2022-01", "2022-02", "2022-03"])
    assert s[1][0] == "2022-12"


def test_cohort_cpr_from_preds_weighted_smm():
    df = pl.DataFrame({
        "vintage_year": [2023, 2023], "coupon_bucket": [6.5, 6.5],
        "month": ["2024-01", "2024-01"],
        "pred_hazard": [0.02, 0.01],
        "weight": [1.0, 1.0],
        "curr_upb": [100_000.0, 300_000.0],
    })
    out = cohort_cpr_from_preds(df)
    smm = (0.02 * 100_000 + 0.01 * 300_000) / 400_000
    assert abs(out["pred_cpr"][0] - (1 - (1 - smm) ** 12)) < 1e-12


def test_cpr_error_table_joins_actuals():
    preds = pl.DataFrame({
        "vintage_year": [2023], "coupon_bucket": [6.5], "month": ["2024-01"],
        "pred_cpr": [0.10],
    })
    actuals = pl.DataFrame({
        "vintage_year": [2023], "coupon_bucket": [6.5], "month": ["2024-01"],
        "cpr": [0.15], "begin_upb": [1e9],
    })
    out = cpr_error_table(preds, actuals)
    assert abs(out["cpr_error"][0] - (-0.05)) < 1e-12
