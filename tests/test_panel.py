import polars as pl

from scurve.panel import define_event, downsample_nonevents, upb_entering


def test_upb_entering_no_zero_leak_at_event():
    """Regression: at the removal month CURRENT_UPB=0; the entering balance
    must be the prior month's balance, never 0 (0 would leak the event)."""
    df = pl.DataFrame({
        "loan_id":  ["A", "A", "A", "B"],
        "month":    ["2023-07", "2023-08", "2023-09", "2023-07"],
        "curr_upb": [100_000.0, 99_000.0, 0.0, 200_000.0],
        "last_upb": [0.0, 0.0, 98_500.0, 0.0],
        "orig_upb": [110_000.0, 110_000.0, 110_000.0, 200_000.0],
    }).sort(["loan_id", "month"])
    out = df.with_columns(upb_entering())
    vals = out["upb_entering"].to_list()
    assert vals[0] == 100_000.0   # A first month -> reported balance fallback
    assert vals[1] == 100_000.0   # lag
    assert vals[2] == 99_000.0    # event month -> PRIOR balance, not 0
    assert vals[3] == 200_000.0   # B first month, curr_upb>0 fallback
    assert all(v > 0 for v in vals)  # never zero anywhere


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
