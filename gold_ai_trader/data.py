"""Загрузка и проверка свечей. timestamp обозначает время открытия в UTC."""
from pathlib import Path
import numpy as np
import pandas as pd


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    needed = ['timestamp', 'open', 'high', 'low', 'close']
    if not set(needed).issubset(frame.columns):
        raise ValueError('Нужны столбцы timestamp, open, high, low, close')
    df = frame.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    for c in needed[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[needed].isna().any().any() or not np.isfinite(df[needed[1:]].to_numpy()).all():
        raise ValueError('Некорректные или пропущенные данные')
    if df.timestamp.duplicated().any() or not df.timestamp.is_monotonic_increasing:
        raise ValueError('Временные метки должны строго возрастать без повторений')
    if (df[needed[1:]] <= 0).any().any() or (df.high < df[['open', 'close', 'low']].max(axis=1)).any() or (df.low > df[['open', 'close', 'high']].min(axis=1)).any():
        raise ValueError('Некорректные цены OHLC')
    return df.reset_index(drop=True)


def load(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == '.csv':
        return validate(pd.read_csv(source))
    if source.suffix.lower() == '.parquet':
        return validate(pd.read_parquet(source))
    raise ValueError('Поддерживаются только CSV и Parquet')
