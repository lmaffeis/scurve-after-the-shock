import zipfile

import polars as pl

from scurve.ingest import ingest_quarter, load_layout
from scurve.quality import quarter_report
from scurve.schema import KEEP

SYNTH_LAYOUT = KEEP + ["EXTRA_1", "EXTRA_2"]


def _make_row(loan_id, period, zb="", curr_upb="100000.00", last_upb=""):
    vals = {c: "" for c in SYNTH_LAYOUT}
    vals.update({
        "LOAN_ID": loan_id, "ACT_PERIOD": period, "ORIG_RATE": "6.500",
        "ORIG_UPB": "300000.00", "CURRENT_UPB": curr_upb, "ORIG_DATE": "062023",
        "LOAN_AGE": "1", "STATE": "CA", "OLTV": "80", "CSCORE_B": "760",
        "Zero_Bal_Code": zb, "LAST_UPB": last_upb,
    })
    return "|".join(vals[c] for c in SYNTH_LAYOUT)


def test_ingest_quarter_projects_keep_columns(tmp_path):
    raw = "\n".join([
        _make_row("L1", "072023"),
        _make_row("L1", "082023", zb="01", curr_upb="0.00", last_upb="99000.00"),
        _make_row("L2", "072023"),
    ])
    zpath = tmp_path / "2023Q3.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("2023Q3.csv", raw)
    out = ingest_quarter(zpath, SYNTH_LAYOUT, tmp_path / "pq", tmp_path / "tmp")
    df = pl.read_parquet(out)
    assert df.shape[0] == 3
    assert set(KEEP) <= set(df.columns)
    assert "EXTRA_1" not in df.columns
    assert df.filter(pl.col("Zero_Bal_Code") == "01").shape[0] == 1
    assert df["ACQ_QUARTER"][0] == "2023Q3"


def test_load_layout_skips_blank_lines(tmp_path):
    p = tmp_path / "layout.csv"
    p.write_text("A\nB\n\nC\n", encoding="utf-8")
    assert load_layout(p) == ["A", "B", "C"]


def test_quarter_report_flags_and_counts(tmp_path):
    df = pl.DataFrame({
        "LOAN_ID": ["L1", "L1", "L2"],
        "ACT_PERIOD": ["072023", "082023", "072023"],
        "CURRENT_UPB": ["100000.00", "0.00", "100000.00"],
        "Zero_Bal_Code": ["", "01", ""],
        "ORIG_RATE": ["6.500", "6.500", "7.125"],
    })
    p = tmp_path / "perf_2023Q3.parquet"
    df.write_parquet(p)
    rep = quarter_report(p)
    assert rep["rows"] == 3
    assert rep["loans"] == 2
    assert rep["null_loan_id"] == 0
    assert rep["zb_codes"]["01"] == 1
    assert rep["bad_act_period"] == 0
