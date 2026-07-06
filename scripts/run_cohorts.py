from scurve.cohorts import cohort_actuals_for_file, combine_cohort_actuals
from scurve.config import load_config

if __name__ == "__main__":
    cfg = load_config()
    files = sorted(cfg["paths"]["parquet_dir"].glob("perf_*.parquet"))
    frames = []
    for f in files:
        print("cohorts:", f.name, flush=True)
        frames.append(cohort_actuals_for_file(f))
    out = combine_cohort_actuals(frames)
    dest = cfg["paths"]["parquet_dir"] / "cohort_actuals.parquet"
    out.write_parquet(dest)
    print(out.shape, "->", dest)
