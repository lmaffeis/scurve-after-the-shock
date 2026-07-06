import polars as pl

from scurve.sample import assign_rate_bucket, keep_fracs, select_loans

EDGES = [0.0, 2.5, 3.0, 6.0, 99.0]


def test_assign_rate_bucket():
    df = pl.DataFrame({"orig_rate": [2.1, 2.75, 6.0, 7.5]})
    out = df.with_columns(assign_rate_bucket(EDGES))
    assert out["rate_bucket"].to_list() == ["0.0-2.5", "2.5-3.0", "6.0-99.0", "6.0-99.0"]


def test_keep_fracs_caps_at_one():
    counts = pl.DataFrame({
        "vintage_year": [2023, 2023], "rate_bucket": ["6.0-99.0", "2.5-3.0"],
        "n": [100_000, 5_000],
    })
    out = keep_fracs(counts, target=20_000)
    big = out.filter(pl.col("rate_bucket") == "6.0-99.0")["keep_frac"][0]
    small = out.filter(pl.col("rate_bucket") == "2.5-3.0")["keep_frac"][0]
    assert abs(big - 0.2) < 1e-9
    assert small == 1.0


def test_select_loans_deterministic_and_weighted():
    loans = pl.DataFrame({
        "LOAN_ID": [f"L{i:06d}" for i in range(10_000)],
        "vintage_year": [2023] * 10_000,
        "rate_bucket": ["6.0-99.0"] * 10_000,
    })
    fracs = pl.DataFrame({
        "vintage_year": [2023], "rate_bucket": ["6.0-99.0"], "keep_frac": [0.25],
    })
    a = select_loans(loans, fracs)
    b = select_loans(loans, fracs)
    assert a["LOAN_ID"].to_list() == b["LOAN_ID"].to_list()   # deterministic
    assert 0.20 < a.shape[0] / 10_000 < 0.30                   # ~25%
    assert abs(a["loan_weight"][0] - 4.0) < 1e-9
