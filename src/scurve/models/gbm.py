"""LightGBM hazard model with native categorical handling."""
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, early_stopping

from ..features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class GBMHazard:
    def __init__(self, params: dict):
        self.params = params
        self.model = LGBMClassifier(
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            min_child_samples=params.get("min_child_samples", 20),
            colsample_bytree=params.get("feature_fraction", 1.0),
            random_state=params["seed"],
            n_jobs=-1,
            verbose=-1,
        )

    def _pandas(self, X: pl.DataFrame):
        pdf = X.select(NUMERIC_FEATURES + CATEGORICAL_FEATURES).to_pandas()
        for c in CATEGORICAL_FEATURES:
            pdf[c] = pdf[c].astype("category")
        return pdf

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray,
            X_val: pl.DataFrame | None = None, y_val=None, w_val=None):
        kwargs = {"sample_weight": w}
        esr = self.params.get("early_stopping_rounds")
        if X_val is not None and esr:
            kwargs["eval_set"] = [(self._pandas(X_val), y_val)]
            kwargs["eval_sample_weight"] = [w_val]
            kwargs["callbacks"] = [early_stopping(esr, verbose=False)]
        self.model.fit(self._pandas(X), y, **kwargs)
        return self

    def predict_hazard(self, X: pl.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._pandas(X))[:, 1]
