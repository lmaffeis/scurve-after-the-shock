"""Build the loan-month hazard panel, one acquisition quarter at a time."""
from pathlib import Path

import duckdb
import polars as pl

from . import features as F
from .sample import _stable_frac


def define_event() -> pl.Expr:
    return (pl.col("Zero_Bal_Code").fill_null("").str.strip_chars() == "01") \
        .cast(pl.Int8).alias("y")


def downsample_nonevents(df: pl.DataFrame, keep_pct: int) -> pl.DataFrame:
    """Keep all y=1 rows; keep ~keep_pct% of y=0 rows deterministically.

    weight = loan_weight for events, loan_weight * (100/keep_pct) for kept
    non-events. Hash key is loan_id||month so a loan's kept months differ."""
    frac = keep_pct / 100.0
    keyed = df.with_columns(
        (pl.col("loan_id") + "|" + pl.col("month"))
        .map_elements(_stable_frac, return_dtype=pl.Float64).alias("_h")
    )
    kept = keyed.filter((pl.col("y") == 1) | (pl.col("_h") < frac))
    return kept.with_columns(
        pl.when(pl.col("y") == 1)
        .then(pl.col("loan_weight"))
        .otherwise(pl.col("loan_weight") / frac)
        .alias("weight")
    ).drop("_h")


def canonical_quarter(parquet_path: Path, sampled_ids: pl.DataFrame,
                      pmms: pl.DataFrame, hpi: pl.DataFrame) -> pl.DataFrame:
    """Read one ingested quarter, filter to sample, cast, join covariates."""
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB';")
    con.register("ids", sampled_ids.to_arrow())
    df = con.execute(f"""
        SELECT p.LOAN_ID AS loan_id,
               substr(p.ACT_PERIOD, 3, 4) || '-' || substr(p.ACT_PERIOD, 1, 2) AS month,
               substr(p.ORIG_DATE, 3, 4) || '-' || substr(p.ORIG_DATE, 1, 2) AS orig_month,
               TRY_CAST(substr(p.ORIG_DATE, 3, 4) AS INTEGER) AS vintage_year,
               TRY_CAST(p.ORIG_RATE AS DOUBLE) AS orig_rate,
               TRY_CAST(p.ORIG_UPB AS DOUBLE) AS orig_upb,
               COALESCE(TRY_CAST(p.CURRENT_UPB AS DOUBLE), 0) AS curr_upb,
               TRY_CAST(p.LOAN_AGE AS INTEGER) AS loan_age,
               TRY_CAST(p.OLTV AS DOUBLE) AS oltv,
               TRY_CAST(p.DTI AS DOUBLE) AS dti,
               TRY_CAST(p.CSCORE_B AS DOUBLE) AS cscore_b,
               p.CHANNEL AS channel, p.PURPOSE AS purpose, p.PROP AS prop,
               p.OCC_STAT AS occ_stat, p.STATE AS state,
               p.NUM_BO AS num_bo, p.FIRST_FLAG AS first_flag,
               p.MOD_FLAG AS mod_flag, p.DLQ_STATUS, p.Zero_Bal_Code,
               ids.loan_weight, ids.rate_bucket
        FROM read_parquet('{parquet_path.as_posix()}') p
        JOIN ids ON p.LOAN_ID = ids.LOAN_ID
        WHERE p.ORIG_RATE IS NOT NULL AND p.ORIG_RATE != ''
    """).pl()
    con.close()
    if df.is_empty():
        return df
    df = (
        df.join(pmms, on="month", how="left")
          .join(pmms.rename({"month": "orig_month", "pmms30": "pmms_orig"}),
                on="orig_month", how="left")
          .join(hpi, on=["state", "month"], how="left")
          .join(hpi.rename({"month": "orig_month", "hpi": "hpi_orig"}),
                on=["state", "orig_month"], how="left")
          .sort(["loan_id", "month"])
    )
    df = df.with_columns(
        F.add_incentive(), F.add_sato(), F.add_mtm_ltv(),
        F.add_month_of_year(), F.dlq_bucket(),
        pl.col("orig_upb").log().alias("orig_upb_log"),
        pl.col("num_bo").cast(pl.Int8, strict=False).fill_null(1)
          .clip(upper_bound=3).alias("num_bo_capped"),
        ((pl.col("orig_rate") * 2).round(0) / 2).alias("coupon_bucket"),
        define_event(),
    ).with_columns(F.add_burnout())
    return df


def build_panel(quarter_files: list[Path], sampled_ids: pl.DataFrame,
                pmms: pl.DataFrame, hpi: pl.DataFrame,
                out_dir: Path, keep_pct: int, min_month: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in quarter_files:
        dest = out_dir / f"panel_{f.stem.removeprefix('perf_')}.parquet"
        if dest.exists():
            print("skip:", dest.name, flush=True)
            continue
        df = canonical_quarter(f, sampled_ids, pmms, hpi)
        if df.is_empty():
            continue
        df = df.filter(pl.col("month") >= min_month)
        df = downsample_nonevents(df, keep_pct)
        keep_cols = (["loan_id", "month", "y", "weight", "curr_upb",
                      "coupon_bucket", "vintage_year", "rate_bucket"]
                     + F.FEATURES)
        df.select([c for c in keep_cols if c in df.columns]).write_parquet(dest)
        print(dest.name, df.shape, flush=True)
