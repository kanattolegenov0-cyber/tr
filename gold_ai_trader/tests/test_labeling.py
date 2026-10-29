from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from gold_ai_trader.labeling import label
from gold_ai_trader.instrument import specification
from gold_ai_trader.mt5_reader import MT5Reader, MT5Unavailable


def data():
    n=40
    price=np.arange(n,dtype=float)+2000
    return pd.DataFrame({'timestamp':pd.date_range('2025-01-01',periods=n,freq='15min',tz='UTC'),'open':price,'high':price+1,'low':price-1,'close':price,'spread':np.repeat(20,n)})


def test_entry_next_bar_and_correct_side_spreads():
    df=data(); out=label(df,point=.01,horizon=4,atr_factor=0,slippage=.05,commission_per_unit=.1)
    i=20
    assert out.loc[i,'entry_time']==df.timestamp.iloc[i+1]
    assert out.loc[i,'exit_time']==df.timestamp.iloc[i+5]
    assert out.loc[i,'long_net']==pytest.approx(4-.2-.1-.1)
    assert out.loc[i,'short_net']==pytest.approx(-4-.2-.1-.1)
    assert out.loc[i,'label']=='BUY'
    assert out.label.iloc[-5:].isna().all()


def test_zero_movement_and_costs_returns_hold():
    df=data(); df[['open','close']]=2000.; df['high']=2001.; df['low']=1999.
    out=label(df,point=.01,atr_factor=.1)
    assert out.label.iloc[20]=='HOLD'
    assert out.long_net.iloc[20] < 0 and out.short_net.iloc[20] < 0


def test_future_change_does_not_change_prior_target():
    df=data(); a=label(df,point=.01)
    df.loc[30:,['open','high','low','close']] += 100
    b=label(df,point=.01)
    pd.testing.assert_frame_equal(a.iloc[:25],b.iloc[:25])


def test_gap_and_missing_spread_rejected():
    df=data(); df.loc[25:,'timestamp']+=pd.Timedelta(hours=1)
    out=label(df,point=.01)
    assert out.label.iloc[20:25].isna().all()
    with pytest.raises(ValueError): label(df.drop(columns='spread'),point=.01)
    with pytest.raises(ValueError): label(df,point=0)
    df.loc[0,'spread']=-1
    with pytest.raises(ValueError): label(df,point=.01)


def test_instrument_only_returns_safe_specification(monkeypatch):
    info=SimpleNamespace(name='GOLD',point=.01,digits=2,trade_tick_size=.01,trade_tick_value=1.,trade_contract_size=100.,currency_base='XAU',currency_profit='USD',currency_margin='USD',volume_min=.01,volume_max=100.,volume_step=.01, password='must-not-be-returned')
    fake=SimpleNamespace(initialize=lambda:True, account_info=lambda:SimpleNamespace(trade_mode=0), ACCOUNT_TRADE_MODE_DEMO=0, shutdown=lambda:None, symbols_get=lambda:[SimpleNamespace(name='GOLD')],symbol_info=lambda _:info)
    monkeypatch.setattr('gold_ai_trader.mt5_reader.importlib.import_module',lambda _:fake)
    reader=MT5Reader(); reader.connect(); result=specification(reader,'GOLD'); reader.close()
    assert result['point']==.01 and 'password' not in result
    info.point=0
    reader.connect()
    with pytest.raises(MT5Unavailable): specification(reader,'GOLD')
    reader.close()


def test_compensating_interval_errors_are_not_treated_as_contiguous():
    df=data()
    df.loc[22,'timestamp'] += pd.Timedelta(minutes=5)
    out=label(df,point=.01)
    assert pd.isna(out.label.iloc[20])
