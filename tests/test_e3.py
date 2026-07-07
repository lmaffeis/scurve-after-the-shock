import numpy as np
import polars as pl

from scurve.experiments.e3_scenarios import scenario_cpr


class FakeModel:
    def predict_hazard(self, X):
        return np.full(X.shape[0], 0.01) + np.clip(
            X["incentive_bps"].to_numpy(), 0, None) / 10_000


def test_scenario_cpr_shifts_incentive():
    df = pl.DataFrame({
        "incentive_bps": [0.0, 0.0], "weight": [1.0, 1.0],
        "curr_upb": [1e5, 1e5], "coupon_bucket": [6.5, 6.5],
        "vintage_year": [2023, 2023],
    })
    out = scenario_cpr(FakeModel(), df, shifts_bps=[0, -100])
    flat = out.filter(pl.col("shift_bps") == 0)["cpr"][0]
    rally = out.filter(pl.col("shift_bps") == -100)["cpr"][0]
    assert rally > flat            # rates down -> incentive up -> faster
