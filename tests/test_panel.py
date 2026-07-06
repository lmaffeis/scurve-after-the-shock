import polars as pl

from scurve.panel import define_event, downsample_nonevents


def test_define_event_marks_zb01_only():
    df = pl.DataFrame({
        "Zero_Bal_Code": ["", "01", "03", None],
    })
    out = df.with_columns(define_event())
    assert out["y"].to_list() == [0, 1, 0, 0]


def test_downsample_keeps_all_events_and_weights_nonevents():
    df = pl.DataFrame({
        "loan_id": [f"L{i}" for i in range(2000)],
        "month": ["2023-01"] * 2000,
        "y": [1] * 100 + [0] * 1900,
        "loan_weight": [2.0] * 2000,
    })
    out = downsample_nonevents(df, keep_pct=10)
    assert out.filter(pl.col("y") == 1).shape[0] == 100          # all events kept
    n0 = out.filter(pl.col("y") == 0).shape[0]
    assert 100 < n0 < 300                                         # ~10% of 1900
    w1 = out.filter(pl.col("y") == 1)["weight"][0]
    w0 = out.filter(pl.col("y") == 0)["weight"][0]
    assert abs(w1 - 2.0) < 1e-9                                   # loan_weight only
    assert abs(w0 - 20.0) < 1e-9                                  # loan_weight * 10
    # determinism
    again = downsample_nonevents(df, keep_pct=10)
    assert out.shape == again.shape
