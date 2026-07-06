"""Package the panel for Kaggle: single train parquet + module sources.

Upload data/kaggle_export/ as a private Kaggle Dataset, then run
notebooks/kaggle_nn_training.py as a Kaggle notebook with GPU on.
See docs/kaggle-instructions.md for the click-by-click guide.
"""
import shutil

import polars as pl

from scurve.config import REPO_ROOT, load_config

if __name__ == "__main__":
    cfg = load_config()
    out = REPO_ROOT / "data" / "kaggle_export"
    out.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(str(cfg["paths"]["panel_dir"] / "panel_*.parquet"))
    lf.collect(engine="streaming").write_parquet(out / "panel.parquet")
    shutil.copy(REPO_ROOT / "src" / "scurve" / "models" / "nn.py", out / "nn_module.py")
    shutil.copy(REPO_ROOT / "src" / "scurve" / "features.py", out / "features_module.py")
    shutil.copy(REPO_ROOT / "notebooks" / "kaggle_nn_training.py", out / "kaggle_nn_training.py")
    size_gb = (out / "panel.parquet").stat().st_size / 1e9
    print(f"export ready: {out}  (panel.parquet: {size_gb:.2f} GB)")
