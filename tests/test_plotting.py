from pathlib import Path

import polars as pl

from scurve.plotting import error_timeseries, grouped_barh, save_fig, scurve_chart


def test_scurve_chart_writes_png(tmp_path):
    df = pl.DataFrame({
        "incentive_bps": [-200, 0, 200] * 2,
        "cpr": [0.04, 0.08, 0.30, 0.05, 0.10, 0.45],
        "series": ["pre-2022"] * 3 + ["lock-in"] * 3,
    })
    fig = scurve_chart(df, title="test")
    out = save_fig(fig, tmp_path, "scurve_test")
    assert Path(out).exists() and Path(out).stat().st_size > 10_000


def test_error_timeseries_writes_png(tmp_path):
    df = pl.DataFrame({
        "month": ["2022-01", "2022-02", "2022-03"] * 2,
        "value": [5.0, 6.0, 4.0, 5.5, 5.8, 4.2],
        "series": ["Actual"] * 3 + ["Model"] * 3,
    })
    fig = error_timeseries(df, title="test")
    out = save_fig(fig, tmp_path, "ts_test")
    assert Path(out).exists() and Path(out).stat().st_size > 10_000


def test_grouped_barh_writes_png(tmp_path):
    fig = grouped_barh(["a", "b", "c"], {"g1": [1, 2, 3], "g2": [2, 1, 2]},
                       title="test", xlabel="x")
    out = save_fig(fig, tmp_path, "bar_test")
    assert Path(out).exists() and Path(out).stat().st_size > 10_000
