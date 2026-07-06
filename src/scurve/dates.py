"""Month-string helpers. Months are always 'YYYY-MM' strings project-wide."""


def mmyyyy_to_month(s: str) -> str:
    """Fannie ACT_PERIOD/ORIG_DATE format 'MMYYYY' -> 'YYYY-MM'."""
    s = s.strip()
    if len(s) != 6 or not s.isdigit() or not 1 <= int(s[:2]) <= 12:
        raise ValueError(f"not MMYYYY: {s!r}")
    return f"{s[2:]}-{s[:2]}"


def month_add(month: str, n: int) -> str:
    y, m = int(month[:4]), int(month[5:7])
    total = y * 12 + (m - 1) + n
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def month_range(start: str, end: str) -> list[str]:
    out, cur = [], start
    while cur <= end:
        out.append(cur)
        cur = month_add(cur, 1)
    return out
