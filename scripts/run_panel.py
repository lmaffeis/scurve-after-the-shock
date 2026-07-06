import duckdb
import polars as pl

from scurve.config import load_config
from scurve.sample import assign_rate_bucket, keep_fracs, select_loans

if __name__ == "__main__":
    cfg = load_config()
    pq = cfg["paths"]["parquet_dir"]
    files = sorted(pq.glob("perf_*.parquet"))

    # 1. one row per loan (first record) across the full store
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB';")
    loans = con.execute(f"""
        SELECT LOAN_ID,
               TRY_CAST(substr(min(ORIG_DATE), 3, 4) AS INTEGER) AS vintage_year,
               TRY_CAST(min(ORIG_RATE) AS DOUBLE) AS orig_rate
        FROM read_parquet({[f.as_posix() for f in files]!r})
        GROUP BY LOAN_ID
    """).pl()
    con.close()
    loans = loans.drop_nulls().with_columns(
        assign_rate_bucket(cfg["sample"]["rate_bucket_edges"])
    )

    # 2. strata counts -> keep fractions -> deterministic selection
    counts = loans.group_by(["vintage_year", "rate_bucket"]).agg(pl.len().alias("n"))
    fracs = keep_fracs(counts, cfg["sample"]["loans_per_stratum"])
    sampled = select_loans(loans, fracs)
    cfg["paths"]["panel_dir"].mkdir(parents=True, exist_ok=True)
    sampled.write_parquet(cfg["paths"]["panel_dir"] / "sampled_loans.parquet")
    print("sampled loans:", sampled.shape[0], "of", loans.shape[0], flush=True)

    # 3. build panel per quarter
    from scurve.panel import build_panel
    pmms = pl.read_parquet(cfg["paths"]["external_dir"] / "pmms_monthly.parquet")
    hpi = pl.read_parquet(cfg["paths"]["external_dir"] / "hpi_state_monthly.parquet")
    build_panel(files, sampled, pmms, hpi, cfg["paths"]["panel_dir"],
                cfg["sample"]["nonevent_keep_pct"], cfg["panel"]["min_perf_month"])
