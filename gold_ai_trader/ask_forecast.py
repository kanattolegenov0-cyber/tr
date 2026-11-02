"""One read-only command for a fresh GOLD M15 research forecast from demo MT5."""
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cost_research import CLASSES, period, prepare
from .data import validate
from .indicators import calculate
from .instrument import specification
from .mt5_reader import MT5Reader, MT5Unavailable


def _price(value: float, tick: float, direction: str) -> float:
    quantum = Decimal(str(tick))
    rounding = ROUND_FLOOR if direction == "down" else ROUND_CEILING
    return float((Decimal(str(value)) / quantum).to_integral_value(rounding=rounding) * quantum)


def levels(bid: float, ask: float, atr: float, tick: float,
           sl_atr: float = 1.5, reward_risk: float = 2.0) -> dict:
    """Illustrative levels around the current quote, with conservative tick rounding."""
    if not all(math.isfinite(v) and v > 0 for v in (bid, ask, atr, tick, sl_atr, reward_risk)):
        raise ValueError("Котировка, ATR, размер тика и параметры TP/SL должны быть положительными")
    if ask <= bid:
        raise ValueError("Ask должен быть выше Bid")
    stop = atr * sl_atr
    buy_entry = _price(ask, tick, "up")
    sell_entry = _price(bid, tick, "down")
    result = {
        "BUY": {"entry": buy_entry, "sl": _price(buy_entry - stop, tick, "down"),
                "tp": _price(buy_entry + stop * reward_risk, tick, "down")},
        "SELL": {"entry": sell_entry, "sl": _price(sell_entry + stop, tick, "up"),
                 "tp": _price(sell_entry - stop * reward_risk, tick, "up")},
    }
    if result["BUY"]["sl"] <= 0 or result["SELL"]["tp"] <= 0:
        raise ValueError("Расчётный TP/SL вышел за пределы положительной цены")
    return result


def analyze(frame: pd.DataFrame, spec: dict, quote: dict, *, now=None,
            horizon: int = 4, atr_factor: float = .1, slippage: float = 0,
            commission_per_unit: float = 0, sl_atr: float = 1.5,
            reward_risk: float = 2.0) -> dict:
    """Check freshness, evaluate a fixed model, then fit it for a current estimate.

    The first 60% trains, next 20% calibrates, and last 20% tests the method.
    For the current estimate, the first 80% trains and last 20% calibrates.
    Neither fit can learn from a target whose exit is beyond the data boundary.
    """
    frame = validate(frame)
    if len(frame) != 10000:
        raise ValueError(f"Нужно ровно 10000 закрытых свечей M15; получено {len(frame)}")
    now = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if now.tzinfo is None:
        raise ValueError("Текущее время должно содержать часовой пояс")
    now = now.tz_convert("UTC")
    last_close = frame.timestamp.iloc[-1].to_pydatetime() + timedelta(minutes=15)
    if not 0 <= (now.to_pydatetime() - last_close).total_seconds() <= 1800:
        raise ValueError("Последняя закрытая свеча устарела или имеет время из будущего")
    tick_time = pd.Timestamp(quote["time_utc"])
    if tick_time.tzinfo is None:
        raise ValueError("Время котировки должно содержать часовой пояс")
    if not 0 <= (now - tick_time).total_seconds() <= 300:
        raise ValueError("Текущая котировка устарела или имеет время из будущего")
    point = float(spec["point"])
    tick_size = float(spec["trade_tick_size"])
    digits = spec["digits"]
    if not math.isfinite(point) or point <= 0 or not math.isfinite(tick_size) or tick_size <= 0:
        raise ValueError("В спецификации MT5 отсутствует корректный point или tick size")
    if not isinstance(digits, int) or not 0 <= digits <= 10:
        raise ValueError("В спецификации MT5 отсутствует корректный digits")
    bid, ask = float(quote["bid"]), float(quote["ask"])
    x, y, valid = prepare(frame, point, horizon, atr_factor, slippage, commission_per_unit)
    n = len(frame)
    check = period(x, y, valid, int(n*.6), int(n*.8), n, horizon)
    # Fixed Logistic model: the last test block is used for evaluation, not selection.
    purge = horizon + 1
    training = valid & (x.index < int(n*.8) - purge)
    calibration = valid & (x.index >= int(n*.8))
    if min(int(training.sum()), int(calibration.sum())) < 60:
        raise ValueError("Недостаточно наблюдений для текущего прогноза")
    if set(y.loc[training].astype(int)) != {0, 1, 2} or set(y.loc[calibration].astype(int)) != {0, 1, 2}:
        raise ValueError("В обучении или калибровке отсутствует один из трёх классов")
    latest = x.iloc[[-1]]
    if latest.isna().any().any():
        raise ValueError("Для последней закрытой свечи не хватает признаков")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(x.loc[training], y.loc[training].astype(int))
    calibrator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    calibrator.fit(model.predict_proba(x.loc[calibration]), y.loc[calibration].astype(int))
    probability = calibrator.predict_proba(model.predict_proba(latest))[0]
    if not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1):
        raise ValueError("Некорректные вероятности модели")
    atr = float(calculate(frame).atr.iloc[-1])
    suggested_levels = levels(bid, ask, atr, tick_size, sl_atr, reward_risk)
    baseline = check["results"]["Baseline"]
    logistic = check["results"]["Logistic"]
    reasons = []
    if logistic["log_loss"] >= baseline["log_loss"] or logistic["brier"] >= baseline["brier"]:
        reasons.append("модель не улучшила обе метрики относительно базового прогноза")
    if min(check["class_counts"].values()) < 30:
        reasons.append("на проверке слишком мало примеров одного из классов")
    if slippage == 0 or commission_per_unit == 0:
        reasons.append("проскальзывание или комиссия указаны нулевыми")
    return {
        "symbol": spec["name"], "last_bar_utc": frame.timestamp.iloc[-1].isoformat(),
        "quote_utc": tick_time.isoformat(), "bid": bid, "ask": ask,
        "bars": len(frame), "valid": int(valid.sum()), "point": point, "digits": digits,
        "atr": atr, "horizon": horizon, "atr_factor": atr_factor,
        "probabilities": dict(zip(CLASSES, map(float, probability))),
        "validation": {"test": check["test"], "class_counts": check["class_counts"],
                       "baseline": baseline, "logistic": logistic},
        "levels": suggested_levels, "sl_atr": sl_atr, "reward_risk": reward_risk,
        "reasons": reasons,
    }


def format_report(result: dict) -> str:
    decimals = result["digits"]
    probabilities = result["probabilities"]
    validation = result["validation"]
    lines = [
        f"{result['symbol']} M15: {result['bars']} закрытых свечей; пригодно {result['valid']}.",
        f"Последняя свеча UTC: {result['last_bar_utc']}; котировка UTC: {result['quote_utc']}.",
        f"Bid={result['bid']:.{decimals}f}; Ask={result['ask']:.{decimals}f}; point={result['point']}; ATR={result['atr']:.{decimals}f}.",
        f"Оценка движения через {result['horizon']} свечи после следующего открытия: " +
        ", ".join(f"{name}={probabilities[name]:.1%}" for name in CLASSES),
        f"Наибольшая оценка модели: {max(CLASSES, key=probabilities.get)}; это не торговая команда.",
        "HOLD в этой строке означает, что ни BUY, ни SELL не превысили порог ATR после оценённых издержек.",
        f"Проверка на {validation['test']} более поздних примерах: " +
        f"Logistic log loss={validation['logistic']['log_loss']:.4f}, Brier={validation['logistic']['brier']:.4f}; " +
        f"baseline log loss={validation['baseline']['log_loss']:.4f}, Brier={validation['baseline']['brier']:.4f} (меньше лучше).",
        "Классы проверки: " + ", ".join(f"{name}={validation['class_counts'][name]}" for name in CLASSES),
        f"Расчётные уровни от текущей котировки: SL={result['sl_atr']} ATR, TP={result['reward_risk']}:1 к риску.",
    ]
    for name in ("BUY", "SELL"):
        item = result["levels"][name]
        lines.append(f"  {name}: вход {item['entry']:.{decimals}f}; SL {item['sl']:.{decimals}f}; TP {item['tp']:.{decimals}f}")
    lines.append("Это ориентиры от текущих Bid/Ask; цена следующего входа, исполнение SL/TP и прибыль не гарантированы.")
    lines.append("Статус: HOLD — " + ("; ".join(result["reasons"]) if result["reasons"] else
                                       "качество вероятностей и доходность реальной сделки не доказаны"))
    lines.append("Программа не отправляет ордера в MT5.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Прогноз GOLD из последних 10000 закрытых свечей demo MT5")
    parser.add_argument("--symbol", default="GOLD")
    parser.add_argument("--slippage", type=float, default=0, help="потери на одну сторону в цене за единицу")
    parser.add_argument("--commission-per-unit", type=float, default=0, help="комиссия за полный цикл в цене за единицу")
    parser.add_argument("--sl-atr", type=float, default=1.5)
    parser.add_argument("--reward-risk", type=float, default=2.0)
    args = parser.parse_args()
    reader = None
    try:
        reader = MT5Reader()
        reader.connect()
        frame = reader.candles(args.symbol, "M15", 10000)
        spec = specification(reader, args.symbol)
        quote = reader.quote(args.symbol)
        print(format_report(analyze(frame, spec, quote, slippage=args.slippage,
                                    commission_per_unit=args.commission_per_unit,
                                    sl_atr=args.sl_atr, reward_risk=args.reward_risk)))
    except (MT5Unavailable, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Прогноз недоступен: {exc}\n")
    finally:
        if reader is not None:
            reader.close()


if __name__ == "__main__":
    main()
