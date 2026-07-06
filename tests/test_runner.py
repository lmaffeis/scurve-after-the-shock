import polars as pl

from scurve.runner import split_panel


def test_split_panel_respects_boundaries():
    df = pl.DataFrame({
        "month": ["2021-11", "2021-12", "2022-01", "2022-02"],
        "y": [0, 0, 1, 0],
    })
    train, test = split_panel(df.lazy(), "2021-12", ["2022-01", "2022-02"])
    assert train.collect()["month"].max() == "2021-12"
    assert set(test.collect()["month"].to_list()) == {"2022-01", "2022-02"}
