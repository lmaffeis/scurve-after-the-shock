"""Plain logistic hazard: the 'old desk model' benchmark.

Deliberately linear in the numeric features (that is the point of the
baseline); categoricals one-hot encoded, numerics standardized.
Because non-events are downsampled with weight 100/keep_pct, weighted
training preserves the true base hazard rate — predicted probabilities are
on the true monthly SMM scale.
"""
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class LogisticHazard:
    def __init__(self):
        self.pipe = Pipeline([
            ("prep", ColumnTransformer([
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ])),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ])

    def _pandas(self, X: pl.DataFrame):
        return X.select(NUMERIC_FEATURES + CATEGORICAL_FEATURES).to_pandas()

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray):
        self.pipe.fit(self._pandas(X), y, clf__sample_weight=w)
        return self

    def predict_hazard(self, X: pl.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(self._pandas(X))[:, 1]
