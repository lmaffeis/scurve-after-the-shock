import polars as pl
import pytest

from scurve.schema import KEEP, layout_from_frame, validate_layout


def test_layout_from_frame_sorts_by_position():
    df = pl.DataFrame({
        "Field Position": [2, 1, 3],
        "Field Name": ["LOAN_ID", "POOL_ID", "ACT_PERIOD"],
    })
    assert layout_from_frame(df) == ["POOL_ID", "LOAN_ID", "ACT_PERIOD"]


def test_validate_layout_passes_when_keep_subset():
    layout = list(KEEP) + ["SOMETHING_ELSE"]
    validate_layout(layout)  # should not raise


def test_validate_layout_reports_missing():
    with pytest.raises(ValueError, match="ORIG_RATE"):
        validate_layout(["LOAN_ID", "ACT_PERIOD"])
