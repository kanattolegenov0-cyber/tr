"""Walk-forward research on outcomes that include a delayed entry and estimated costs.

This module reads closed M15 bars only. It never calls the MT5 trading API.
"""
import argparse
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data import load
from .indicators import calculate
from .labeling import label

CLASSES = ("SELL", "HOLD", "BUY")
TARGET = {name: index for index, name in enumerate(CLASSES)}


def features(frame: pd.DataFrame, point: float) -> pd.DataFrame:
    """Return features available when the current closed bar becomes known."""
    if not math.isfinite(point) or point <= 0:
        raise ValueError("Требуется фактический положительный point из MT5")
    d = calculate(frame)
    x = pd.DataFrame(index=d.index)
    for period in (20, 50, 200):
        x[f"ema_{period}"] = d.close / d[f"ema{period}"] - 1
    for column in ("rsi", "return_1", "return_3", "return_5", "return_10", "volatility"):
        x[column] = d[column]
    for column in ("atr", "macd", "body", "upper_wick", "lower_wick"):
        x[column] = d[column] / d.close
    x["spread_fraction"] = d.spread * point / d.close
    if "tick_volume" in d:
        volume = pd.to_numeric(d.tick_volume, errors="coerce")
        if volume.isna().any() or not np.isfinite(volume).all() or (volume < 0).any():
            raise ValueError("tick_volume должен быть неотрицательным и конечным")
        x["tick_volume_log"] = np.log1p(volume)
    hour = d.timestamp.dt.hour + d.timestamp.dt.minute / 60
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # An observed gap is known at the current bar. Earlier indicators may span it.
    gaps = d.timestamp.diff().dt.total_seconds().ne(900)
    last_gap = pd.Series(np.where(gaps, np.arange(len(d)), np.nan), index=d.index).ffill()
    x["bars_since_gap"] = (np.arange(len(d)) - last_gap).clip(0, 200)
    return x.replace([np.inf, -np.inf], np.nan)


def prepare(frame: pd.DataFrame, point: float, horizon: int = 4,
            atr_factor: float = .1, slippage: float = 0,
            commission_per_unit: float = 0, entry_delay: int = 1):
    outcomes = label(frame, point, horizon, atr_factor, slippage, commission_per_unit, entry_delay)
    x = features(frame, point)
    y = outcomes.label.map(TARGET).astype("float")
    valid = y.notna() & x.notna().all(axis=1)
    return x, y, valid


def _scores(actual, probabilities):
    predicted = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1, 2], zero_division=0
    )
    return {
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((probabilities - np.eye(3)[actual]) ** 2, axis=1))),
        "f1_macro": float(f1_score(actual, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "by_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                            "f1": float(f1[i]), "support": int(support[i])}
                     for i, name in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1, 2]).tolist(),
    }


def period(x, y, valid, start_cal: int, start_test: int, end_test: int,
           horizon: int = 4, entry_delay: int = 1):
    """Fit on the past, calibrate later, and score an untouched later block.

    The purge covers the delayed entry and its exit after the horizon.
    """
    if horizon < 1 or entry_delay < 1 or not (0 < start_cal < start_test < end_test <= len(x)):
        raise ValueError("Неверные границы временных блоков")
    purge = horizon + entry_delay
    index = x.index
    masks = (
        valid & (index < start_cal - purge),
        valid & (index >= start_cal) & (index < start_test - purge),
        valid & (index >= start_test) & (index < end_test - purge),
    )
    splits = [(x.loc[m], y.loc[m].astype(int)) for m in masks]
    if min(len(values) for _, values in splits) < 60:
        raise ValueError("В одном из временных блоков менее 60 наблюдений")
    (train, yt), (cal, yc), (test, ye) = splits
    if set(yt) != {0, 1, 2} or set(yc) != {0, 1, 2}:
        raise ValueError("В обучении или калибровке отсутствует один из трёх классов")
    # Laplace smoothing keeps baseline probabilities finite even for a rare class.
    frequency = (np.bincount(yt, minlength=3) + 1) / (len(yt) + 3)
    baseline = np.tile(frequency, (len(ye), 1))
    results = {"Baseline": _scores(ye, baseline)}
    models = {
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=20,
                                                random_state=42, n_jobs=1),
        "XGBoost": XGBClassifier(n_estimators=80, max_depth=2, learning_rate=.05,
                                 objective="multi:softprob", num_class=3, n_jobs=1, random_state=42),
    }
    for name, model in models.items():
        model.fit(train, yt)
        calibration = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
        calibration.fit(model.predict_proba(cal), yc)
        probabilities = calibration.predict_proba(model.predict_proba(test))
        results[name] = _scores(ye, probabilities)
    ema_prediction = np.where(test.ema_20 > test.ema_50, 2, 0)
    results["EMA20/50"] = {
        "f1_macro": float(f1_score(ye, ema_prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(ye, ema_prediction, labels=[0, 1, 2]).tolist(),
    }
    return {"train": len(yt), "calibration": len(yc), "test": len(ye),
            "class_counts": {name: int((ye == i).sum()) for i, name in enumerate(CLASSES)},
            "results": results}


def main():
    parser = argparse.ArgumentParser(description="Исследование GOLD M15 с издержками; ордера не отправляются")
    parser.add_argument("file")
    parser.add_argument("--point", type=float, required=True, help="фактический point из MT5")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--atr-factor", type=float, default=.1)
    parser.add_argument("--slippage", type=float, default=0)
    parser.add_argument("--commission-per-unit", type=float, default=0)
    args = parser.parse_args()
    try:
        frame = load(args.file)
        x, y, valid = prepare(frame, args.point, args.horizon, args.atr_factor,
                              args.slippage, args.commission_per_unit)
        print(f"Свечей: {len(frame)}; пригодных примеров: {int(valid.sum())}")
        print("Метки:", {name: int((y[valid] == i).sum()) for i, name in enumerate(CLASSES)})
        n = len(frame)
        for name, a, b, c in (("Период 1", .4, .5, .6), ("Период 2", .5, .6, .7),
                              ("Период 3", .6, .7, .8), ("Последний изученный период", .6, .8, 1.0)):
            print(name)
            try:
                report = period(x, y, valid, int(n*a), int(n*b), int(n*c), args.horizon)
                print("  train/cal/test:", report["train"], report["calibration"], report["test"])
                print("  Классы теста:", report["class_counts"])
                for model, score in report["results"].items():
                    print(f"  {model}: F1={score['f1_macro']:.4f}" +
                          (f"; log loss={score['log_loss']:.4f}; Brier={score['brier']:.4f}"
                           if "log_loss" in score else ""))
            except ValueError as exc:
                print("  Не выполнено:", exc)
        print("HOLD: независимость последнего периода зависит от прежних экспериментов; это не оценка прибыли.")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
