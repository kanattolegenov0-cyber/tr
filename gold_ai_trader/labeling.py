"""Учебная маркировка будущих результатов с приближёнными издержками."""
import argparse
import math
import numpy as np
import pandas as pd
from .data import load, validate
from .indicators import calculate


def label(frame: pd.DataFrame, point: float, horizon: int = 4, atr_factor: float = .1, slippage: float = 0, commission_per_unit: float = 0) -> pd.DataFrame:
    # Спред MT5 в пунктах нельзя переводить в цену без реальной спецификации.
    if not math.isfinite(point) or point <= 0 or not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Требуются положительные point и целый горизонт')
    if not all(math.isfinite(v) and v >= 0 for v in (atr_factor, slippage, commission_per_unit)):
        raise ValueError('Параметры риска и издержек должны быть конечными и неотрицательными')
    df = validate(frame)
    if 'spread' not in df:
        raise ValueError('Нет исторического столбца spread в пунктах MT5')
    spread = pd.to_numeric(df.spread, errors='coerce')
    if spread.isna().any() or not np.isfinite(spread).all() or (spread < 0).any():
        raise ValueError('Исторический spread должен содержать неотрицательные конечные числа')
    indicators = calculate(df)
    # Прокси: цены OHLC — Bid, исторический spread свечи используется как оценка на её открытии.
    entry_bid = df.open.shift(-1)
    exit_bid = df.open.shift(-(horizon + 1))
    entry_spread = spread.shift(-1) * point
    exit_spread = spread.shift(-(horizon + 1)) * point
    long_net = exit_bid - (entry_bid + entry_spread) - 2*slippage - commission_per_unit
    short_net = entry_bid - (exit_bid + exit_spread) - 2*slippage - commission_per_unit
    threshold = indicators.atr * atr_factor
    # Проверяем каждую пару свечей: суммарной длительности окна недостаточно.
    irregular = (df.timestamp.diff().dt.total_seconds() != 900).astype(int).cumsum()
    future_irregular = irregular.shift(-(horizon+1)) - irregular
    valid = future_irregular.eq(0) & threshold.notna() & long_net.notna() & short_net.notna()
    outcome = pd.Series(pd.NA, index=df.index, dtype='string')
    outcome.loc[valid] = 'HOLD'
    outcome.loc[valid & (long_net > threshold)] = 'BUY'
    outcome.loc[valid & (short_net > threshold)] = 'SELL'
    return pd.DataFrame({'timestamp': df.timestamp, 'entry_time':df.timestamp.shift(-1), 'exit_time':df.timestamp.shift(-(horizon+1)), 'atr':indicators.atr, 'threshold':threshold, 'long_net':long_net.where(valid), 'short_net':short_net.where(valid), 'label':outcome})


def main():
    parser = argparse.ArgumentParser(description='Учебные метки GOLD M15: без торговых операций')
    parser.add_argument('file'); parser.add_argument('--point', type=float, required=True, help='point из спецификации MT5')
    parser.add_argument('--horizon', type=int, default=4); parser.add_argument('--atr-factor', type=float, default=.1, help='учебный коэффициент, не оптимизирован')
    parser.add_argument('--slippage', type=float, default=0, help='издержка на сторону в цене за единицу')
    parser.add_argument('--commission-per-unit', type=float, default=0, help='сумма комиссии за вход и выход в цене за единицу')
    args = parser.parse_args()
    try:
        result = label(load(args.file), args.point, args.horizon, args.atr_factor, args.slippage, args.commission_per_unit)
    except ValueError as exc:
        parser.error(str(exc))
    counts = result.label.value_counts()
    print('Всего свечей:',len(result),'; размечено:',int(result.label.notna().sum()),'; исключено:',int(result.label.isna().sum()))
    for direction in ('SELL', 'HOLD', 'BUY'):
        print(f'{direction}: {int(counts.get(direction, 0))}')
    print('Учебная оценка с приближённым спредом; не прогноз и не реальные сделки. Статус HOLD.')

if __name__ == '__main__':
    main()
