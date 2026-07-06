"""Build configs/fannie_layout.csv, the official Fannie Mae SF Loan Performance
column order (one name per line, in file order).

Fannie Mae does not publish a single "Field Position / Field Name" workbook
specific to the Single-Family Loan Performance file's short-code column names.
Instead:

  * Fannie's own R ingestion script `LPPUB_Infile.R` (shipped inside
    FNMA_SF_Loan_Performance_r_Primary.zip, linked from the SF Loan
    Performance Data page) defines `lppub_column_names`, an ordered vector of
    the first 110 short-code field names -- this is the authoritative source
    for column *names* (POOL_ID, LOAN_ID, ACT_PERIOD, ...).
  * Fannie's "CRT File Layout and Glossary" workbook (crt-file-layout-and-
    glossary.xlsx, same page) has a `Field Position` / `Field Name` sheet
    ("Combined Glossary") covering fields 1-113 for CAS/CIRT/SF Loan
    Performance combined. Positions 1-110 line up 1:1 with the R script's
    codes (verified by hand); positions 111-113 are three FICO fields added
    after the R script was last updated (Fannie's notes say "Populated
    starting with the December 2025 activity period"), so they have no
    established short code yet -- we derive one from the descriptive name.

This script parses the R script for positions 1-110 and appends the three
newer fields for 111-113, producing a single ordered layout that we then
validate against KEEP and, ultimately, against the real pipe-delimited data
files (field count must match -- see the ground-truth check run separately).

Usage: python scripts/make_layout.py "data/raw/fannie/r_primary_extract/LPPUB_Infile.R"
"""
import re
import sys

import polars as pl

from scurve.config import REPO_ROOT
from scurve.schema import layout_from_frame, validate_layout

# Fields added by Fannie after the R script (last updated through 2023Q4/110
# fields) was published. Source: crt-file-layout-and-glossary.xlsx,
# "Combined Glossary" sheet, Field Position 111-113, Field Name column.
EXTRA_FIELDS = [
    (111, "ORIGINATION_CLASSIC_FICO"),  # "Origination Classic FICO(R)"
    (112, "ISSUANCE_CLASSIC_FICO"),     # "Issuance Classic FICO(R)"
    (113, "CURRENT_CLASSIC_FICO"),      # "Current Classic FICO(R)"
]


def parse_r_column_names(r_script: str) -> list[str]:
    """Extract the ordered `lppub_column_names <- c(...)` vector from the R script."""
    text = open(r_script, encoding="utf-8").read()
    m = re.search(r"lppub_column_names\s*<-\s*c\((.*?)\)\s*\n", text, re.S)
    if m is None:
        raise SystemExit(
            "Could not find 'lppub_column_names <- c(...)' in the R script. "
            "Open it and check the assignment still uses that exact name."
        )
    return re.findall(r'"([^"]+)"', m.group(1))


def main(r_script: str) -> None:
    names = parse_r_column_names(r_script)
    print(f"parsed {len(names)} field names from {r_script}")

    rows = [(i + 1, n) for i, n in enumerate(names)] + EXTRA_FIELDS
    df = pl.DataFrame(
        {"Field Position": [p for p, _ in rows], "Field Name": [n for _, n in rows]}
    )
    layout = layout_from_frame(df)

    out = REPO_ROOT / "configs" / "fannie_layout.csv"
    out.write_text("\n".join(layout), encoding="utf-8")
    print(f"{len(layout)} fields -> {out}")

    validate_layout(layout)
    print("validate_layout: OK")


if __name__ == "__main__":
    main(sys.argv[1])
