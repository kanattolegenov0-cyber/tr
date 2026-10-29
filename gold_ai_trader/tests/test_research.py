import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from gold_ai_trader.data import validate
from gold_ai_trader.indicators import calculate
from gold_ai_trader.backtest import Costs, run
from gold_ai_trader.mt5_reader import MT5Reader, MT5Unavailable


def frame(n=120):
    values = 2000 + np.sin(np.arange(n)/6)*10 + np.arange(n)*.01
    return validate(pd.DataFrame({'timestamp': pd.date_range('2025-01-01',periods=n,freq='15min',tz='UTC'), 'open': values, 'high': values+1, 'low': values-1, 'close': values}))


def test_indicators_do_not_use_future():
    df = frame(); changed = df.copy(); changed.loc[90:, 'close'] += 30
    pd.testing.assert_frame_equal(calculate(df).iloc[:90], calculate(changed).iloc[:90])


def test_invalid_data_rejected():
    bad = frame(); bad.loc[4,'high'] = 1
    with pytest.raises(ValueError): validate(bad)
    bad = frame(); bad.loc[4,'timestamp'] = bad.loc[3,'timestamp']
    with pytest.raises(ValueError): validate(bad)


def test_future_prices_do_not_change_prior_equity():
    df = frame(); changed = df.copy()
    changed.loc[90:, ['open','close','low','high']] += 100
    _, first = run(df, Costs(spread=.3))
    _, second = run(changed, Costs(spread=.3))
    pd.testing.assert_frame_equal(first.iloc[:90], second.iloc[:90])


def test_spread_commission_and_next_open():
    df = frame(); trades, _ = run(df, Costs(spread=.3, slippage=.1, commission_per_side=2, units=1))
    assert len(trades) > 0
    for trade in trades.itertuples():
        pnl = (trade.exit-trade.entry)*trade.direction - 4
        assert trade.pnl_usd == pytest.approx(pnl)
        idx = df.index[df.timestamp == trade.entry_time][0]
        assert idx >= 50
        open_price = df.loc[idx,'open']
        assert trade.entry == pytest.approx(open_price + .4 if trade.direction == 1 else open_price - .1)


def test_real_account_rejected_and_closed(monkeypatch):
    fake = SimpleNamespace(initialize=lambda:True, account_info=lambda:SimpleNamespace(trade_mode=2), ACCOUNT_TRADE_MODE_DEMO=0, shutdown=lambda: calls.append('closed'))
    calls=[]
    monkeypatch.setattr('gold_ai_trader.mt5_reader.importlib.import_module', lambda _:fake)
    with pytest.raises(MT5Unavailable): MT5Reader().connect()
    assert calls == ['closed']


def test_missing_mt5(monkeypatch):
    def missing(_): raise ImportError('missing')
    monkeypatch.setattr('gold_ai_trader.mt5_reader.importlib.import_module', missing)
    with pytest.raises(MT5Unavailable): MT5Reader()


def test_research_code_contains_no_execution_api():
    base = Path(__file__).resolve().parents[1]
    forbidden = {'order_send', 'order_check', 'OrderSend'}
    for path in base.glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        assert not any(isinstance(node, ast.Attribute) and node.attr in forbidden for node in ast.walk(tree)), str(path)
        assert not any(isinstance(node, ast.ImportFrom) and node.module and (node.module == 'src' or node.module.startswith('src.')) for node in ast.walk(tree))


def test_finds_xm_gold_symbol(monkeypatch):
    fake = SimpleNamespace(initialize=lambda: True, account_info=lambda: SimpleNamespace(trade_mode=0), ACCOUNT_TRADE_MODE_DEMO=0, shutdown=lambda: None, symbols_get=lambda: [SimpleNamespace(name=n) for n in ('GOLD', 'XAUUSD.a', 'EURUSD')])
    monkeypatch.setattr('gold_ai_trader.mt5_reader.importlib.import_module', lambda _: fake)
    reader = MT5Reader()
    reader.connect()
    assert reader.symbols() == ['GOLD', 'XAUUSD.a']
    reader.close()
