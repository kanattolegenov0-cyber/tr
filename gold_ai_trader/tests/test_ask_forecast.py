import numpy as np
import pandas as pd
import pytest

from gold_ai_trader.ask_forecast import analyze, format_report, levels


def history():
    rng = np.random.default_rng(21)
    n = 10000
    price = 2000 + np.cumsum(rng.normal(0, .8, n))
    now = pd.Timestamp("2026-11-02T12:31:00Z")
    frame = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp("2026-11-02T12:15:00Z"), periods=n, freq="15min"),
        "open": price, "high": price + 1, "low": price - 1, "close": price,
        "spread": np.full(n, 40), "tick_volume": np.full(n, 100),
    })
    spec = {"name": "GOLD", "point": .01, "trade_tick_size": .01, "digits": 2}
    quote = {"bid": 2000.03, "ask": 2000.43, "time_utc": "2026-11-02T12:30:00Z"}
    return frame, spec, quote, now


def test_levels_are_ordered_and_tick_aligned():
    result = levels(2000.03, 2000.43, 2, .01)
    assert result["BUY"] == {"entry": 2000.43, "sl": 1997.43, "tp": 2006.43}
    assert result["SELL"] == {"entry": 2000.03, "sl": 2003.03, "tp": 1994.03}


def test_command_core_outputs_three_probabilities_and_context():
    frame, spec, quote, now = history()
    report = analyze(frame, spec, quote, now=now)
    assert report["bars"] == 10000
    assert report["validation"]["test"] > 1000
    assert set(report["probabilities"]) == {"SELL", "HOLD", "BUY"}
    assert sum(report["probabilities"].values()) == pytest.approx(1)
    text = format_report(report)
    assert "Статус: HOLD" in text
    assert "SL" in text and "TP" in text


def test_stale_or_short_history_is_rejected_before_model_fit():
    frame, spec, quote, now = history()
    with pytest.raises(ValueError, match="10000"):
        analyze(frame.iloc[1:], spec, quote, now=now)
    old = frame.copy()
    old["timestamp"] -= pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="устарела") as error:
        analyze(old, spec, quote, now=now)
    assert "Сейчас UTC" in str(error.value)
    assert "2026-11-02T12:30:00+00:00" in str(error.value)
    assert "Котировка свежая, но история нет" in str(error.value)
    quote["time_utc"] = "2026-11-02T12:20:00Z"
    with pytest.raises(ValueError, match="котировка устарела"):
        analyze(frame, spec, quote, now=now)
