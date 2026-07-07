import numpy as np
import polars as pl

from scurve.experiments.e2_scurves import pdp_incentive


class FakeModel:
    def predict_hazard(self, X):
        # hazard = monotone function of incentive only
        return 1 / (1 + np.exp(-(X["incentive_bps"].to_numpy()) / 100)) * 0.05


def test_pdp_incentive_recovers_monotone_curve():
    ref = pl.DataFrame({
        "incentive_bps": [0.0] * 50,
        "other": np.arange(50.0),
    })
    out = pdp_incentive(FakeModel(), ref, grid=[-200, 0, 200])
    cprs = out["cpr"].to_list()
    assert cprs[0] < cprs[1] < cprs[2]
    assert out["incentive_bps"].to_list() == [-200, 0, 200]
