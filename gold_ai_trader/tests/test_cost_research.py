import numpy as np
import pandas as pd
import pytest

from gold_ai_trader.cost_research import features, period, prepare


def sample(n=1400):
    rng = np.random.default_rng(19)
    price = 2000 + np.cumsum(rng.normal(0, .8, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
        "open": price, "high": price + 1, "low": price - 1,
        "close": price, "spread": np.full(n, 40), "tick_volume": np.full(n, 100),
    })


def test_features_and_outcomes_are_causal():
    frame = sample()
    x, y, valid = prepare(frame, point=.01)
    changed = frame.copy()
    changed.loc[1000:, ["open", "high", "low", "close"]] += 50
    x2, y2, valid2 = prepare(changed, point=.01)
    pd.testing.assert_frame_equal(x.iloc[:1000], x2.iloc[:1000])
    pd.testing.assert_series_equal(y.iloc[:995], y2.iloc[:995])
    pd.testing.assert_series_equal(valid.iloc[:995], valid2.iloc[:995])
    assert y.iloc[-5:].isna().all()


def test_purged_period_does_not_read_later_prices():
    frame = sample()
    x, y, valid = prepare(frame, point=.01)
    original = period(x, y, valid, 650, 850, 1050)
    changed = frame.copy()
    changed.loc[1050:, ["open", "high", "low", "close"]] += 100
    x2, y2, valid2 = prepare(changed, point=.01)
    assert period(x2, y2, valid2, 650, 850, 1050) == original
    assert set(original["results"]) == {"Baseline", "Logistic", "RandomForest", "XGBoost", "EMA20/50"}


def test_invalid_volume_and_point_are_rejected():
    frame = sample()
    with pytest.raises(ValueError):
        prepare(frame, point=0)
    frame.loc[50, "tick_volume"] = -1
    with pytest.raises(ValueError, match="tick_volume"):
        features(frame, point=.01)
