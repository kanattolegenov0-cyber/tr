"""CLI offline-анализа; не вызывает исходный бот."""
import argparse
from pathlib import Path
from .data import load
from .indicators import calculate
from .backtest import Costs, run


def main():
    p = argparse.ArgumentParser(description='Gold AI Trader: offline исследование XAUUSD')
    p.add_argument('file', help='CSV или Parquet с закрытыми свечами Bid в UTC')
    p.add_argument('--spread', type=float, required=True, help='фиксированный спред в цене XAUUSD')
    p.add_argument('--slippage', type=float, default=0.0)
    p.add_argument('--commission', type=float, default=0.0, help='USD за сторону')
    p.add_argument('--units', type=float, default=1.0, help='условные единицы золота, не лоты')
    p.add_argument('--output', default='gold-output', help='папка результата')
    args = p.parse_args()
    df = load(args.file)
    if len(df) < 51:
        p.error('Нужно не менее 51 закрытой свечи')
    costs = Costs(args.spread, args.slippage, args.commission, args.units)
    trades, equity = run(df, costs)
    target = Path(args.output); target.mkdir(parents=True, exist_ok=True)
    trades.to_csv(target/'trades.csv', index=False); equity.to_csv(target/'equity.csv', index=False)
    indicators = calculate(df)
    print(f'Свечей: {len(df)}; закрытых виртуальных сделок: {len(trades)}')
    print(f'Последняя EMA20: {indicators.ema20.iloc[-1]:.3f}; EMA50: {indicators.ema50.iloc[-1]:.3f}')
    print(f'Последняя оценка капитала: {equity.equity_usd.iloc[-1]:.2f} USD (условная модель)')
    print(f'Результаты: {target}; исторический тест не предсказывает будущую доходность.')

if __name__ == '__main__':
    main()
