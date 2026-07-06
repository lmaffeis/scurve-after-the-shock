import pytest

from scurve.dates import mmyyyy_to_month, month_add, month_range


def test_mmyyyy_to_month():
    assert mmyyyy_to_month("012018") == "2018-01"
    assert mmyyyy_to_month("122025") == "2025-12"


def test_mmyyyy_rejects_garbage():
    with pytest.raises(ValueError):
        mmyyyy_to_month("2018-01")


def test_month_add():
    assert month_add("2021-12", 1) == "2022-01"
    assert month_add("2022-03", -3) == "2021-12"


def test_month_range():
    assert month_range("2021-11", "2022-01") == ["2021-11", "2021-12", "2022-01"]
