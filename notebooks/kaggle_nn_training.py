# Kaggle notebook: train the SSG-style NN hazard model on the exported panel.
# Paste this whole script into ONE cell of a Kaggle notebook.
# Requirements: attach your uploaded dataset, enable GPU (Settings -> Accelerator).
# Adjust DATA below to your dataset's mount path (check the right-hand Data panel).
import importlib.util
import json
import sys

import polars as pl

DATA = "/kaggle/input/scurve-panel"          # <- adjust to your dataset slug


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# nn_module does `from ..features import ...`; shim a fake scurve package first
feats = _load("scurve.features", f"{DATA}/features_module.py")
pkg = type(sys)("scurve")
pkg.features = feats
sys.modules["scurve"] = pkg
nnmod = _load("scurve.models.nn", f"{DATA}/nn_module.py")

TRAIN_ENDS = ["2019-12", "2021-12", "2024-12"]   # NN at key splits only (GPU budget)


def month_add(month, n):
    y, m = int(month[:4]), int(month[5:7])
    total = y * 12 + (m - 1) + n
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


lf = pl.scan_parquet(f"{DATA}/panel.parquet").drop_nulls(
    subset=["incentive_bps", "mtm_ltv"])
results = {}
for train_end in TRAIN_ENDS:
    tr = lf.filter(pl.col("month") <= train_end).collect()
    te = lf.filter((pl.col("month") > train_end)
                   & (pl.col("month") <= month_add(train_end, 12))).collect()
    print(f"train_end {train_end}: train {tr.shape}, test {te.shape}", flush=True)
    model = nnmod.NNHazard(epochs=8)
    model.fit(tr, tr["y"].to_numpy(), tr["weight"].to_numpy())
    p = model.predict_hazard(te)
    te.select("loan_id", "month", "y", "weight", "curr_upb",
              "coupon_bucket", "vintage_year").with_columns(
        pl.Series("pred_hazard", p),
        pl.lit(train_end).alias("train_end")).write_parquet(
        f"/kaggle/working/preds_nn_{train_end}.parquet")
    results[train_end] = model.history
    print(train_end, "loss history:", model.history, flush=True)

with open("/kaggle/working/nn_history.json", "w") as f:
    json.dump(results, f)
print("DONE - download /kaggle/working/*.parquet and nn_history.json")
