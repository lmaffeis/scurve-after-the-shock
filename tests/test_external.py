import polars as pl

from scurve.external import interpolate_quarterly_to_monthly, weekly_to_monthly


def test_weekly_to_monthly_averages_within_month():
    df = pl.DataFrame({
        "date": ["2023-01-05", "2023-01-12", "2023-02-02"],
        "value": [6.0, 7.0, 5.0],
    }).with_columns(pl.col("date").str.to_date())
    out = weekly_to_monthly(df)
    jan = out.filter(pl.col("month") == "2023-01")["value"][0]
    assert abs(jan - 6.5) < 1e-9
    assert out.filter(pl.col("month") == "2023-02")["value"][0] == 5.0


def test_hpi_interpolation_linear_between_quarters():
    df = pl.DataFrame({
        "state": ["CA", "CA"],
        "year": [2023, 2023],
        "quarter": [1, 2],
        "index_nsa": [100.0, 106.0],
    })
    out = interpolate_quarterly_to_monthly(df)
    # quarter anchors at 2023-03 and 2023-06; expect 102 at 2023-04, 104 at 2023-05
    apr = out.filter(pl.col("month") == "2023-04")["hpi"][0]
    assert abs(apr - 102.0) < 1e-9
