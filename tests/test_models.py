import numpy as np
import polars as pl

from scurve.models.baseline import LogisticHazard
from scurve.models.gbm import GBMHazard


def _synthetic(n=20_000, seed=7):
    """Hazard driven by an S-curve in incentive; everything else noise."""
    rng = np.random.default_rng(seed)
    inc = rng.uniform(-300, 300, n)
    true_p = 0.005 + 0.06 / (1 + np.exp(-(inc - 50) / 40))
    y = (rng.uniform(0, 1, n) < true_p).astype(int)
    df = pl.DataFrame({
        "incentive_bps": inc,
        "sato_bps": rng.normal(0, 30, n),
        "burnout_bps": rng.uniform(0, 500, n),
        "mtm_ltv": rng.uniform(30, 95, n),
        "loan_age": rng.integers(1, 120, n),
        "orig_upb_log": rng.normal(12.5, 0.5, n),
        "cscore_b": rng.uniform(620, 820, n),
        "dti": rng.uniform(10, 50, n),
        "oltv": rng.uniform(40, 97, n),
        "month_of_year": rng.integers(1, 13, n),
        "pmms30": rng.uniform(2.5, 7.5, n),
        "channel": rng.choice(["R", "C", "B"], n),
        "purpose": rng.choice(["P", "C", "R"], n),
        "prop": rng.choice(["SF", "PU", "CO"], n),
        "occ_stat": rng.choice(["P", "S", "I"], n),
        "state": rng.choice(["CA", "TX", "FL", "NY"], n),
        "num_bo_capped": rng.integers(1, 4, n),
        "first_flag": rng.choice(["Y", "N"], n),
        "dlq_bucket": rng.integers(0, 3, n),
        "mod_flag": rng.choice(["Y", "N"], n),
    })
    return df, y, np.ones(n)


def _check(model):
    X, y, w = _synthetic()
    model.fit(X, y, w)
    p = model.predict_hazard(X)
    assert p.shape == (X.shape[0],)
    assert (p > 0).all() and (p < 1).all()
    # model recovers monotone incentive response: high-incentive mean hazard
    # must exceed low-incentive mean hazard by a wide margin
    inc = X["incentive_bps"].to_numpy()
    assert p[inc > 150].mean() > 2 * p[inc < -150].mean()


def test_logistic_hazard():
    _check(LogisticHazard())


def test_gbm_hazard():
    _check(GBMHazard({"num_leaves": 31, "learning_rate": 0.1,
                      "n_estimators": 100, "early_stopping_rounds": None,
                      "seed": 1}))
