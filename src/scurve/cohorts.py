"""Full-population cohort CPR actuals from the ingested parquet store.

One acquisition-quarter file contains a loan's complete history, so per-file
aggregation followed by a cross-file re-sum is exact.
"""
from pathlib import Path

import duckdb
import polars as pl


def smm_to_cpr(smm: float) -> float:
    return 1.0 - (1.0 - smm) ** 12


def cohort_actuals_for_file(parquet_path: Path) -> pl.DataFrame:
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB';")
    out = con.execute(f"""
        WITH m AS (
            SELECT LOAN_ID,
                   substr(ACT_PERIOD, 3, 4) || '-' || substr(ACT_PERIOD, 1, 2) AS month,
                   TRY_CAST(ORIG_RATE AS DOUBLE) AS orig_rate,
                   TRY_CAST(substr(ORIG_DATE, 3, 4) AS INTEGER) AS vintage_year,
                   COALESCE(TRY_CAST(CURRENT_UPB AS DOUBLE), 0) AS curr_upb,
                   COALESCE(TRY_CAST(LAST_UPB AS DOUBLE), 0) AS last_upb,
                   Zero_Bal_Code AS zb
            FROM read_parquet('{parquet_path.as_posix()}')
            WHERE ORIG_RATE IS NOT NULL AND ORIG_RATE != ''
        ),
        lagged AS (
            SELECT *,
                   lag(curr_upb) OVER (PARTITION BY LOAN_ID ORDER BY month) AS prev_upb,
                   round(orig_rate * 2) / 2 AS coupon_bucket
            FROM m
        )
        SELECT vintage_year, coupon_bucket, month,
               sum(CASE WHEN zb = '01'
                        THEN CASE WHEN last_upb > 0 THEN last_upb ELSE prev_upb END
                        ELSE 0 END) AS prepaid_upb,
               sum(prev_upb) AS begin_upb,
               count(*) AS loan_months
        FROM lagged
        WHERE prev_upb IS NOT NULL AND prev_upb > 0
        GROUP BY 1, 2, 3
    """).pl()
    con.close()
    return out


def combine_cohort_actuals(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Re-sum per-file aggregates, then compute SMM/CPR."""
    return (
        pl.concat(frames)
        .group_by(["vintage_year", "coupon_bucket", "month"])
        .agg(pl.col("prepaid_upb").sum(), pl.col("begin_upb").sum(),
             pl.col("loan_months").sum())
        .with_columns((pl.col("prepaid_upb") / pl.col("begin_upb")).alias("smm"))
        .with_columns((1 - (1 - pl.col("smm")) ** 12).alias("cpr"))
        .sort(["vintage_year", "coupon_bucket", "month"])
    )
