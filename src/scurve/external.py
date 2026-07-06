"""Download and tidy external covariates: PMMS 30y rate (FRED), FHFA state HPI."""
import io
from pathlib import Path

import polars as pl
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FHFA_STATE_HPI = "https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_at_state.csv"


def weekly_to_monthly(df: pl.DataFrame) -> pl.DataFrame:
    """Average weekly observations into calendar months. Input: date (Date), value."""
    return (
        df.sort("date")
        .group_by_dynamic("date", every="1mo")
        .agg(pl.col("value").mean())
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .select("month", "value")
    )


def interpolate_quarterly_to_monthly(df: pl.DataFrame) -> pl.DataFrame:
    """FHFA quarterly state HPI -> monthly by linear interpolation.

    Input columns: state, year, quarter, index_nsa. Quarter anchored to its
    final month (Q1 -> March). Output: state, month, hpi.
    """
    q = (
        df.with_columns(pl.date(pl.col("year"), pl.col("quarter") * 3, 1).alias("date"))
        .sort(["state", "date"])
    )
    return (
        q.upsample(time_column="date", every="1mo", group_by="state")
        .with_columns(
            pl.col("index_nsa").interpolate().alias("hpi"),
        )
        .drop_nulls("hpi")
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .select("state", "month", "hpi")
    )


def fetch_fred(series: str) -> pl.DataFrame:
    resp = requests.get(FRED_CSV.format(series=series), timeout=60)
    resp.raise_for_status()
    df = pl.read_csv(io.BytesIO(resp.content), null_values=".")
    df = df.rename({df.columns[0]: "date", df.columns[1]: "value"})
    return df.with_columns(pl.col("date").str.to_date()).drop_nulls()


def fetch_fhfa_state_hpi() -> pl.DataFrame:
    resp = requests.get(FHFA_STATE_HPI, timeout=120)
    resp.raise_for_status()
    df = pl.read_csv(
        io.BytesIO(resp.content),
        has_header=False,
        new_columns=["state", "year", "quarter", "index_nsa"],
    )
    # If FHFA ships a header row, first row will fail the int cast -- detect and drop.
    if df["year"].dtype == pl.Utf8:
        df = df.filter(pl.col("year").str.contains(r"^\d{4}$")).with_columns(
            pl.col("year").cast(pl.Int64),
            pl.col("quarter").cast(pl.Int64),
            pl.col("index_nsa").cast(pl.Float64),
        )
    return df


def build_all(external_dir: Path) -> None:
    external_dir.mkdir(parents=True, exist_ok=True)
    pmms = weekly_to_monthly(fetch_fred("MORTGAGE30US")).rename({"value": "pmms30"})
    pmms.write_parquet(external_dir / "pmms_monthly.parquet")
    hpi = interpolate_quarterly_to_monthly(fetch_fhfa_state_hpi())
    hpi.write_parquet(external_dir / "hpi_state_monthly.parquet")
