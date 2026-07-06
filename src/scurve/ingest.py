"""Stream one quarterly zip through DuckDB into a ZSTD parquet with KEEP columns.

Everything stays varchar at this stage (raw files have blanks and surprises);
casting happens in downstream SQL where nulls are handled explicitly.
"""
import zipfile
from pathlib import Path

import duckdb

from .schema import KEEP


def ingest_quarter(zip_path: Path, layout: list[str], out_dir: Path, tmp_dir: Path) -> Path:
    quarter = zip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"perf_{quarter}.parquet"

    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith((".csv", ".txt")))
        extracted = Path(zf.extract(member, tmp_dir))

    try:
        con = duckdb.connect()
        con.execute("SET preserve_insertion_order=false; SET memory_limit='3GB';")
        cols = ", ".join(f'"{c}"' for c in KEEP)
        con.execute(f"""
            COPY (
                SELECT {cols}, '{quarter}' AS ACQ_QUARTER
                FROM read_csv('{extracted.as_posix()}', delim='|', header=false,
                              names={layout!r}, all_varchar=true)
            ) TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        con.close()
    finally:
        extracted.unlink(missing_ok=True)
    return out_path


def load_layout(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
