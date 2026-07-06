import polars as pl

from scurve.cohorts import cohort_actuals_for_file, combine_cohort_actuals, smm_to_cpr


def test_smm_to_cpr():
    assert abs(smm_to_cpr(0.0) - 0.0) < 1e-12
    assert abs(smm_to_cpr(0.01) - (1 - 0.99**12)) < 1e-12


def test_cohort_actuals_single_prepay(tmp_path):
    # Two loans, same cohort (vintage 2023, 6.5 coupon). L1 prepays in 2023-09.
    df = pl.DataFrame({
        "LOAN_ID":       ["L1", "L1", "L1", "L2", "L2", "L2"],
        "ACT_PERIOD":    ["072023", "082023", "092023", "072023", "082023", "092023"],
        "ORIG_RATE":     ["6.500"] * 6,
        "ORIG_DATE":     ["062023"] * 6,
        "CURRENT_UPB":   ["100000", "99000", "0", "200000", "199000", "198000"],
        "LAST_UPB":      ["", "", "98500", "", "", ""],
        "Zero_Bal_Code": ["", "", "01", "", "", ""],
    })
    p = tmp_path / "perf_2023Q3.parquet"
    df.write_parquet(p)
    out = cohort_actuals_for_file(p)
    sep = out.filter(pl.col("month") == "2023-09")
    # begin UPB entering Sep = 99000 + 199000; prepaid = LAST_UPB 98500
    assert abs(sep["prepaid_upb"][0] - 98500) < 1e-6
    assert abs(sep["begin_upb"][0] - 298000) < 1e-6
    assert sep["vintage_year"][0] == 2023
    assert abs(sep["coupon_bucket"][0] - 6.5) < 1e-9


def test_combine_cohort_actuals_resums_and_computes_cpr():
    a = pl.DataFrame({
        "vintage_year": [2023], "coupon_bucket": [6.5], "month": ["2023-09"],
        "prepaid_upb": [50_000.0], "begin_upb": [1_000_000.0], "loan_months": [10],
    })
    b = pl.DataFrame({
        "vintage_year": [2023], "coupon_bucket": [6.5], "month": ["2023-09"],
        "prepaid_upb": [50_000.0], "begin_upb": [1_000_000.0], "loan_months": [12],
    })
    out = combine_cohort_actuals([a, b])
    assert out.shape[0] == 1
    smm = 100_000 / 2_000_000
    assert abs(out["smm"][0] - smm) < 1e-12
    assert abs(out["cpr"][0] - (1 - (1 - smm) ** 12)) < 1e-12
