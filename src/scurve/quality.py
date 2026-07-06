"""Per-quarter ingestion sanity report. Fails loudly on structural problems."""
import json
from pathlib import Path

import duckdb


def quarter_report(parquet_path: Path) -> dict:
    con = duckdb.connect()
    row = con.execute(f"""
        SELECT
            count(*) AS rows,
            count(DISTINCT LOAN_ID) AS loans,
            sum(CASE WHEN LOAN_ID IS NULL OR LOAN_ID = '' THEN 1 ELSE 0 END) AS null_loan_id,
            sum(CASE WHEN NOT regexp_matches(ACT_PERIOD, '^(0[1-9]|1[0-2])\\d{{4}}$')
                THEN 1 ELSE 0 END) AS bad_act_period
        FROM read_parquet('{parquet_path.as_posix()}')
    """).fetchone()
    zb = con.execute(f"""
        SELECT Zero_Bal_Code, count(*) FROM read_parquet('{parquet_path.as_posix()}')
        WHERE Zero_Bal_Code IS NOT NULL AND Zero_Bal_Code != ''
        GROUP BY 1
    """).fetchall()
    con.close()
    rep = {
        "file": parquet_path.name,
        "rows": row[0], "loans": row[1],
        "null_loan_id": row[2], "bad_act_period": row[3],
        "zb_codes": {code: n for code, n in zb},
    }
    if rep["null_loan_id"] > 0 or rep["bad_act_period"] > rep["rows"] * 0.001:
        raise ValueError(f"quality check failed: {rep}")
    return rep


def write_report(rep: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"quality_{rep['file']}.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
