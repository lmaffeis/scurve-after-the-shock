import sys

import polars as pl

from scurve.config import load_config
from scurve.experiments.e1_frozen import run_e1

if __name__ == "__main__":
    cfg = load_config()
    actuals = pl.read_parquet(cfg["paths"]["parquet_dir"] / "cohort_actuals.parquet")
    which = sys.argv[1:] or ["e1", "e2", "e3", "robustness"]
    if "e1" in which:
        run_e1(cfg, actuals)
    if "e2" in which:
        from scurve.experiments.e2_scurves import run_e2
        run_e2(cfg)
    if "e3" in which:
        from scurve.experiments.e3_scenarios import run_e3
        run_e3(cfg)
    if "robustness" in which:
        from scurve.experiments.robustness import run_robustness
        run_robustness(cfg, actuals)
