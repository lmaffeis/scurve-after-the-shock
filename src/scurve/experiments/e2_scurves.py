"""E2: what did the prepayment function learn in each regime?

Per-regime GBM fits -> PDP over incentive on a COMMON reference population
(so curve differences are behavior, not composition) -> overlay S-curves.
SATO-tertile conditioning tests SanCap's composition-bias claim directly.
SHAP summaries quantify the driver-mix shift. Outputs: e2_scurves.png,
e2_scurves_by_sato.png, e2_shap.png, e2.json."""
import json

import numpy as np
import polars as pl

from ..features import FEATURES
from ..models.gbm import GBMHazard
from ..plotting import grouped_barh, save_fig, scurve_chart

GRID = list(range(-300, 301, 25))
REF_ROWS = 200_000
SHAP_ROWS = 50_000

# config keys -> reader-facing labels (figures only; JSON keeps config keys)
DISPLAY = {"normal": "Pre-pandemic (2018-19)",
           "refi_wave": "Refinancing wave (2020-21)",
           "lockin": "Lock-in era (2022-25)"}


def pdp_incentive(model, ref: pl.DataFrame, grid=GRID) -> pl.DataFrame:
    rows = []
    for g in grid:
        x = ref.with_columns(pl.lit(float(g)).alias("incentive_bps"))
        smm = float(np.average(model.predict_hazard(x)))
        rows.append({"incentive_bps": g, "smm": smm, "cpr": 1 - (1 - smm) ** 12})
    return pl.DataFrame(rows)


def _regime_frames(cfg: dict) -> dict[str, pl.LazyFrame]:
    lf = pl.scan_parquet(str(cfg["paths"]["panel_dir"] / "panel_*.parquet")) \
           .drop_nulls(subset=["incentive_bps", "mtm_ltv"])
    return {name: lf.filter((pl.col("month") >= lo) & (pl.col("month") <= hi))
            for name, (lo, hi) in cfg["regimes"].items()}


def run_e2(cfg: dict) -> None:
    fig_dir, tab_dir = cfg["paths"]["figures_dir"], cfg["paths"]["tables_dir"]
    regimes = _regime_frames(cfg)
    seed = cfg["models"]["gbm"]["seed"]

    # common reference population: lock-in era rows
    ref_full = regimes["lockin"].collect()
    ref = (ref_full.sample(n=min(REF_ROWS, ref_full.shape[0]), seed=seed)
           .select(FEATURES))
    del ref_full
    t1, t2 = ref["sato_bps"].quantile(1 / 3), ref["sato_bps"].quantile(2 / 3)

    curves, sato_curves, shap_summaries, metrics = [], [], {}, {}
    for name, lf in regimes.items():
        df = lf.collect()
        model = GBMHazard(cfg["models"]["gbm"] | {"early_stopping_rounds": None,
                                                  "n_estimators": 400})
        model.fit(df.select(FEATURES), df["y"].to_numpy(), df["weight"].to_numpy())
        n_rows = df.shape[0]
        del df

        pdp = pdp_incentive(model, ref).with_columns(pl.lit(DISPLAY[name]).alias("series"))
        curves.append(pdp)
        for i, (lo, hi) in enumerate([(None, t1), (t1, t2), (t2, None)]):
            sub = ref
            if lo is not None:
                sub = sub.filter(pl.col("sato_bps") >= lo)
            if hi is not None:
                sub = sub.filter(pl.col("sato_bps") < hi)
            sato_curves.append(
                pdp_incentive(model, sub)
                .with_columns(pl.lit(f"{DISPLAY[name]} / SATO T{i + 1}").alias("series")))

        import shap
        expl = shap.TreeExplainer(model.model)
        xs = model._pandas(ref.head(SHAP_ROWS))
        sv = expl.shap_values(xs)
        sv = sv[1] if isinstance(sv, list) else sv
        shap_summaries[name] = {c: float(np.abs(sv[:, i]).mean())
                                for i, c in enumerate(xs.columns)}

        p = pdp.sort("incentive_bps")
        cpr_at = dict(zip(p["incentive_bps"].to_list(), p["cpr"].to_list()))
        metrics[name] = {
            "turnover_cpr_at_minus200": cpr_at[-200],
            "cpr_at_0": cpr_at[0],
            "cpr_at_plus150": cpr_at[150],
            "refi_elasticity_pts_per_100bps": 100 * (cpr_at[150] - cpr_at[0]) / 1.5,
            "n_train_rows": n_rows,
        }
        print(f"E2 {name}: {metrics[name]}", flush=True)

    save_fig(scurve_chart(pl.concat(curves),
                          "The learned S-curve, by regime (common population)"),
             fig_dir, "e2_scurves")
    save_fig(scurve_chart(pl.concat(sato_curves),
                          "S-curves by regime and SATO tertile"),
             fig_dir, "e2_scurves_by_sato")

    top = sorted(shap_summaries["lockin"], key=shap_summaries["lockin"].get)[-10:]
    fig = grouped_barh(top, {DISPLAY[name]: [s[c] for c in top]
                             for name, s in shap_summaries.items()},
                       "Prepayment drivers by regime",
                       "mean |SHAP| (hazard scale)")
    save_fig(fig, fig_dir, "e2_shap")

    tab_dir.mkdir(parents=True, exist_ok=True)
    with open(tab_dir / "e2.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "shap_mean_abs": shap_summaries}, f, indent=2)
