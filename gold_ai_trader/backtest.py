"""Учебный тест EMA: только синтетическое исполнение следующего открытия."""
from dataclasses import dataclass
import math
import pandas as pd
from .indicators import calculate

@dataclass(frozen=True)
class Costs:
    spread: float
    slippage: float = 0.0
    commission_per_side: float = 0.0
    units: float = 1.0
    initial_balance: float = 10000.0

    def __post_init__(self):
        values = (self.spread, self.slippage, self.commission_per_side, self.units, self.initial_balance)
        if not all(math.isfinite(v) for v in values) or min(values[:3]) < 0 or min(values[3:]) <= 0:
            raise ValueError('Издержки должны быть конечными и неотрицательными; объём и баланс положительными')


def run(frame: pd.DataFrame, costs: Costs) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = calculate(frame)
    # Сигнал свечи i применяется только на открытии свечи i+1.
    desired = pd.Series(0, index=df.index, dtype=int)
    ready = df.ema20.notna() & df.ema50.notna()
    desired.loc[ready & (df.ema20 > df.ema50)] = 1
    desired.loc[ready & (df.ema20 < df.ema50)] = -1
    position = 0; entry = 0.0; balance = costs.initial_balance; trades = []; equity = []
    for i, row in df.iterrows():
        target = int(desired.iloc[i-1]) if i > 0 else 0
        # Считаем OHLC ценами Bid, Ask = Bid + фиксированный спред.
        if target != position:
            if position:
                exit_price = row.open - costs.slippage if position == 1 else row.open + costs.spread + costs.slippage
                pnl = (exit_price-entry)*position*costs.units - 2*costs.commission_per_side
                balance += pnl
                trades.append({'entry_time':entry_time, 'exit_time':row.timestamp, 'direction':position, 'entry':entry, 'exit':exit_price, 'pnl_usd':pnl})
            position = target
            if position:
                entry = row.open + costs.spread + costs.slippage if position == 1 else row.open - costs.slippage
                entry_time = row.timestamp
        mark = row.close if position == 1 else row.close + costs.spread
        floating = (mark-entry)*position*costs.units - 2*costs.commission_per_side if position else 0
        equity.append({'timestamp':row.timestamp, 'equity_usd':balance+floating})
    # Открытая позиция отражается в equity, но не считается закрытой сделкой.
    return pd.DataFrame(trades, columns=['entry_time','exit_time','direction','entry','exit','pnl_usd']), pd.DataFrame(equity)
