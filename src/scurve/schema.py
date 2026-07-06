"""Columns we keep from the Fannie Mae Single-Family Loan Performance files.

Names must match the official file layout exactly. `validate_layout` is the
guard: if Fannie's layout spells a field differently, it fails loudly and the
KEEP list gets fixed in one commit (never silently remap).
"""

KEEP = [
    "LOAN_ID",
    "ACT_PERIOD",       # reporting month, MMYYYY
    "CHANNEL",
    "ORIG_RATE",
    "CURR_RATE",
    "ORIG_UPB",
    "CURRENT_UPB",
    "ORIG_TERM",
    "ORIG_DATE",        # MMYYYY
    "FIRST_PAY",
    "LOAN_AGE",
    "OLTV",
    "OCLTV",
    "NUM_BO",
    "DTI",
    "CSCORE_B",
    "FIRST_FLAG",
    "PURPOSE",
    "PROP",
    "NO_UNITS",
    "OCC_STAT",
    "STATE",
    "MSA",
    "MI_PCT",
    "PRODUCT",
    "DLQ_STATUS",
    "MOD_FLAG",
    "Zero_Bal_Code",
    "ZB_DTE",           # MMYYYY of zero-balance event
    "LAST_UPB",         # UPB immediately prior to removal
]


def layout_from_frame(df) -> list[str]:
    """Official layout sheet (Field Position, Field Name) -> ordered name list."""
    rows = df.sort("Field Position")
    return [str(n).strip() for n in rows["Field Name"].to_list()]


def validate_layout(layout: list[str]) -> None:
    missing = [c for c in KEEP if c not in layout]
    if missing:
        raise ValueError(
            f"KEEP columns missing from layout: {missing}. "
            f"Layout has {len(layout)} fields; inspect configs/fannie_layout.csv "
            "and correct KEEP spelling to match the official names."
        )
