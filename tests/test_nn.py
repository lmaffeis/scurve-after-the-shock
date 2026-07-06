from scurve.models.nn import NNHazard
from tests.test_models import _synthetic


def test_nn_smoke_cpu():
    X, y, w = _synthetic(n=5_000)
    model = NNHazard(epochs=3, batch_size=1024, hidden=(32, 32))
    model.fit(X, y, w)
    p = model.predict_hazard(X)
    assert p.shape == (5_000,) and (p > 0).all() and (p < 1).all()
    assert model.history[-1] < model.history[0]  # training loss decreased
