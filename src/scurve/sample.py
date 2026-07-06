"""Deterministic stratified loan sampling.

Selection uses a stable string hash of LOAN_ID (no RNG, no seed drift between
runs/machines). Strata are vintage_year x rate_bucket; small strata are kept
in full, large ones thinned to `loans_per_stratum`, with inverse-probability
loan_weight so aggregates remain population-representative.
"""
import hashlib

import polars as pl


def _stable_frac(s: str) -> float:
    """Map a string to [0,1) deterministically (md5, first 8 hex digits)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 16**8


def assign_rate_bucket(edges: list[float]) -> pl.Expr:
    labels = [f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
    rate = pl.col("orig_rate")
    result = pl.when(rate < edges[1]).then(pl.lit(labels[0]))
    for lo, hi, lab in zip(edges[1:-1], edges[2:], labels[1:]):
        result = result.when((rate >= lo) & (rate < hi)).then(pl.lit(lab))
    return result.otherwise(pl.lit(labels[-1])).alias("rate_bucket")


def keep_fracs(stratum_counts: pl.DataFrame, target: int) -> pl.DataFrame:
    return stratum_counts.with_columns(
        pl.min_horizontal(pl.lit(1.0), target / pl.col("n")).alias("keep_frac")
    ).drop("n")


def select_loans(loans: pl.DataFrame, fracs: pl.DataFrame) -> pl.DataFrame:
    """loans: LOAN_ID, vintage_year, rate_bucket. Returns kept ids + loan_weight."""
    hashed = loans.with_columns(
        pl.col("LOAN_ID").map_elements(_stable_frac, return_dtype=pl.Float64).alias("_h")
    ).join(fracs, on=["vintage_year", "rate_bucket"], how="inner")
    return (
        hashed.filter(pl.col("_h") < pl.col("keep_frac"))
        .with_columns((1.0 / pl.col("keep_frac")).alias("loan_weight"))
        .select("LOAN_ID", "vintage_year", "rate_bucket", "loan_weight")
    )
