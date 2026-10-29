"""Публичные свечи и строгая проверка CSV; искусственные котировки не используются."""
import io
import numpy as np
import pandas as pd

INTERVALS = {'15m': 900, '1h': 3600, '4h': 14400, '1d': 86400}
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

def validate(frame, timeframe, now=None):
    if timeframe not in INTERVALS:
        raise ValueError('Неизвестный таймфрейм')
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not set(required).issubset(frame.columns):
        raise ValueError('CSV должен содержать timestamp,open,high,low,close,volume')
    df = frame[required].copy()
    if pd.api.types.is_numeric_dtype(df.timestamp):
        df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms', utc=True, errors='coerce')
    else:
        df['timestamp'] = pd.to_datetime(df.timestamp, utc=True, errors='coerce')
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.isna().any().any() or not np.isfinite(df[required[1:]].to_numpy()).all():
        raise ValueError('Обнаружены пропуски или некорректные числа')
    if df.timestamp.duplicated().any():
        raise ValueError('Повторяющиеся временные метки')
    df = df.sort_values('timestamp').reset_index(drop=True)
    if (df[['open','high','low','close']] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError('Цены должны быть положительными, объём — неотрицательным')
    if ((df.high < df[['open','close','low']].max(axis=1)) | (df.low > df[['open','close','high']].min(axis=1))).any():
        raise ValueError('Нарушены границы свечей OHLC')
    step = pd.Timedelta(seconds=INTERVALS[timeframe])
    if len(df) > 1 and not df.timestamp.diff().iloc[1:].eq(step).all():
        raise ValueError('Есть разрывы или неверный интервал свечей; исправьте CSV')
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if (df.timestamp > now).any():
        raise ValueError('Свечи из будущего недопустимы')
    df['closed'] = df.timestamp + step <= now
    return df

def read_csv(raw, timeframe):
    try:
        return validate(pd.read_csv(io.BytesIO(raw)), timeframe)
    except (pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError('Не удалось прочитать CSV в UTF-8') from exc

def fetch(symbol, timeframe):
    if symbol not in SYMBOLS or timeframe not in INTERVALS:
        raise ValueError('Неподдерживаемый актив или таймфрейм')
    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 15000, 'options': {'fetchMarkets': {'types': ['spot']}}})
        rows = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
    except Exception as exc:
        raise ValueError('Биржа недоступна. Загрузите CSV с историей свечей.') from exc
    return validate(pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume']), timeframe)
