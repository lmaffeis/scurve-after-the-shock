"""Feed-forward hazard net (SSG-style): categorical embeddings + dense ReLU.

CPU-capable for tests/smoke; real training happens on Kaggle GPU via
notebooks/kaggle_nn_training.py, which loads this module from the uploaded
dataset. Keep this file free of package-relative runtime dependencies beyond
scurve.features (shipped alongside as features_module.py in the export).
"""
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class _Net(nn.Module):
    def __init__(self, n_num, cat_cards, hidden):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(c, min(16, (c + 1) // 2)) for c in cat_cards])
        emb_dim = sum(e.embedding_dim for e in self.embs)
        dims = [n_num + emb_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(0.1)]
        layers += [nn.Linear(dims[-1], 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        return self.mlp(torch.cat([x_num, *embs], dim=1)).squeeze(1)


class NNHazard:
    def __init__(self, epochs=10, batch_size=8192, hidden=(128, 128, 64), lr=1e-3):
        self.epochs, self.batch_size, self.hidden, self.lr = epochs, batch_size, hidden, lr
        self.history: list[float] = []

    def _encode(self, X: pl.DataFrame, fit: bool):
        if fit:
            self.num_mean = {c: float(X[c].cast(pl.Float64).mean()) for c in NUMERIC_FEATURES}
            self.num_std = {c: max(float(X[c].cast(pl.Float64).std() or 1.0), 1e-9)
                            for c in NUMERIC_FEATURES}
            self.cat_maps = {
                c: {v: i + 1 for i, v in enumerate(
                    sorted(X[c].cast(pl.Utf8).fill_null("_NA_").unique().to_list()))}
                for c in CATEGORICAL_FEATURES}          # 0 reserved for unseen
        x_num = np.stack([
            ((X[c].cast(pl.Float64).fill_null(self.num_mean[c]).to_numpy()
              - self.num_mean[c]) / self.num_std[c])
            for c in NUMERIC_FEATURES], axis=1).astype(np.float32)
        x_cat = np.stack([
            np.array([self.cat_maps[c].get(v, 0)
                      for v in X[c].cast(pl.Utf8).fill_null("_NA_").to_list()])
            for c in CATEGORICAL_FEATURES], axis=1).astype(np.int64)
        return torch.from_numpy(x_num), torch.from_numpy(x_cat)

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x_num, x_cat = self._encode(X, fit=True)
        cards = [len(self.cat_maps[c]) + 1 for c in CATEGORICAL_FEATURES]
        self.net = _Net(x_num.shape[1], cards, self.hidden).to(dev)
        ds = TensorDataset(x_num, x_cat,
                           torch.from_numpy(y.astype(np.float32)),
                           torch.from_numpy(w.astype(np.float32)))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.BCEWithLogitsLoss(reduction="none")
        for _ in range(self.epochs):
            tot, wsum = 0.0, 0.0
            for xn, xc, yb, wb in dl:
                xn, xc, yb, wb = xn.to(dev), xc.to(dev), yb.to(dev), wb.to(dev)
                opt.zero_grad()
                loss = (lossf(self.net(xn, xc), yb) * wb).sum() / wb.sum()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * float(wb.sum())
                wsum += float(wb.sum())
            self.history.append(tot / wsum)
        return self

    @torch.no_grad()
    def predict_hazard(self, X: pl.DataFrame) -> np.ndarray:
        dev = next(self.net.parameters()).device
        x_num, x_cat = self._encode(X, fit=False)
        out = []
        for i in range(0, len(x_num), 65536):
            logits = self.net(x_num[i:i + 65536].to(dev), x_cat[i:i + 65536].to(dev))
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)
