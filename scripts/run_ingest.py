"""Ingest quarterly zips. Usage:
python scripts/run_ingest.py            # all zips in raw_dir not yet ingested
python scripts/run_ingest.py 2023Q3 2023Q4
"""
import sys

from scurve.config import REPO_ROOT, load_config
from scurve.ingest import ingest_quarter, load_layout
from scurve.quality import quarter_report, write_report

if __name__ == "__main__":
    cfg = load_config()
    raw, pq = cfg["paths"]["raw_dir"], cfg["paths"]["parquet_dir"]
    layout = load_layout(REPO_ROOT / "configs" / "fannie_layout.csv")
    wanted = sys.argv[1:] or None
    zips = sorted(raw.glob("*.zip"))
    for z in zips:
        if wanted and z.stem not in wanted:
            continue
        out = pq / f"perf_{z.stem}.parquet"
        if out.exists():
            print("skip (exists):", out.name)
            continue
        print("ingesting", z.name, "...", flush=True)
        path = ingest_quarter(z, layout, pq, raw / "_tmp")
        rep = quarter_report(path)
        write_report(rep, pq / "_quality")
        print(f"  {rep['rows']:,} rows, {rep['loans']:,} loans, zb: {rep['zb_codes']}", flush=True)
