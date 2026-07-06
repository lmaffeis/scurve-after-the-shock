import polars as pl

from scurve.config import load_config
from scurve.runner import run_walk_forward

if __name__ == "__main__":
    cfg = load_config()
    actuals = pl.read_parquet(cfg["paths"]["parquet_dir"] / "cohort_actuals.parquet")
    run_walk_forward(str(cfg["paths"]["panel_dir"] / "panel_*.parquet"),
                     cfg, actuals, cfg["paths"]["models_dir"])
