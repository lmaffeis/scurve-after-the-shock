import polars as pl

from scurve.features import (add_burnout, add_incentive, add_mtm_ltv,
                             add_month_of_year, add_sato, dlq_bucket)


def _frame():
    return pl.DataFrame({
        "loan_id":   ["A", "A", "A"],
        "month":     ["2023-07", "2023-08", "2023-09"],
        "orig_rate": [7.0, 7.0, 7.0],
        "pmms30":    [6.8, 6.5, 7.2],
        "pmms_orig": [6.7, 6.7, 6.7],
        "oltv":      [80.0, 80.0, 80.0],
        "orig_upb":  [400_000.0, 400_000.0, 400_000.0],
        "curr_upb":  [400_000.0, 398_000.0, 396_000.0],
        "hpi":       [300.0, 303.0, 306.0],
        "hpi_orig":  [300.0, 300.0, 300.0],
    })


def test_incentive_and_sato_bps():
    out = _frame().with_columns(add_incentive(), add_sato())
    assert [round(v, 6) for v in out["incentive_bps"].to_list()] == [20.0, 50.0, -20.0]
    assert round(out["sato_bps"][0], 6) == 30.0


def test_burnout_is_lagged_cumulative_positive_incentive():
    out = (_frame().with_columns(add_incentive())
           .with_columns(add_burnout()))
    # month1: no history -> 0; month2: max(20,0)=20; month3: 20+50=70
    assert [round(v, 6) for v in out["burnout_bps"].to_list()] == [0.0, 20.0, 70.0]


def test_mtm_ltv_uses_hpi_ratio():
    out = _frame().with_columns(add_mtm_ltv())
    # month2: 80 * (398000/400000) / (303/300)
    assert abs(out["mtm_ltv"][1] - 80 * (398_000 / 400_000) / (303 / 300)) < 1e-9


def test_month_of_year():
    out = _frame().with_columns(add_month_of_year())
    assert out["month_of_year"].to_list() == [7, 8, 9]


def test_dlq_bucket():
    df = pl.DataFrame({"DLQ_STATUS": ["00", "01", "03", "XX", ""]})
    assert df.with_columns(dlq_bucket())["dlq_bucket"].to_list() == [0, 1, 2, 0, 0]
