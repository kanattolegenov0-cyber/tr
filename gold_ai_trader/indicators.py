"""Индикаторы используют только текущую и предыдущие закрытые свечи."""
import numpy as np
import pandas as pd


def calculate(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df.close
    for period in (20, 50, 200):
        df[f'ema{period}'] = close.ewm(span=period, adjust=False, min_periods=period).mean()
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df.loc[(loss == 0) & (gain > 0), 'rsi'] = 100
    df.loc[(loss == 0) & (gain == 0), 'rsi'] = 50
    df['macd'] = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    tr = pd.concat([df.high-df.low, (df.high-close.shift()).abs(), (df.low-close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    mid = close.rolling(20).mean(); sd = close.rolling(20).std(ddof=0)
    df['bb_upper'], df['bb_lower'] = mid + 2*sd, mid - 2*sd
    for period in (1, 3, 5, 10):
        df[f'return_{period}'] = close.pct_change(period, fill_method=None)
    df['volatility'] = df.return_1.rolling(20).std()
    df['body'] = (df.close-df.open).abs()
    df['upper_wick'] = df.high-df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1)-df.low
    df['distance_high_20'] = df.high.rolling(20).max()-close
    df['distance_low_20'] = close-df.low.rolling(20).min()
    df['hour_utc'] = df.timestamp.dt.hour
    return df
