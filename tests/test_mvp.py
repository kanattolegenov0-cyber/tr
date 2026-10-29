import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app, analyze
from backend.market_data import validate
from backend.indicators import calculate
from backend.feature_engineering import features
from backend.ml_models import run
from backend.signal_engine import decide

def fixture(n=1000):
    rng=np.random.default_rng(42)
    price=100*np.exp(np.cumsum(rng.normal(0,.01,n)))
    times=pd.date_range(end=pd.Timestamp.now(tz='UTC').floor('h')-pd.Timedelta(hours=1),periods=n,freq='h')
    return pd.DataFrame({'timestamp':times,'open':price,'close':price,'high':price*1.01,'low':price*.99,'volume':rng.uniform(1,100,n)})

class MVP(unittest.TestCase):
    def test_causal_features(self):
        df=fixture()
        before=features(calculate(df)).iloc[:800]
        df.loc[800:,'close']*=3
        pd.testing.assert_frame_equal(before,features(calculate(df)).iloc[:800])
    def test_invalid_ohlc(self):
        df=fixture();df.loc[0,'high']=0
        with self.assertRaises(ValueError): validate(df,'1h')
    def test_gap(self):
        with self.assertRaises(ValueError): validate(fixture().drop(index=20),'1h')
    def test_flat_rsi(self):
        df=fixture();df['close']=100
        self.assertEqual(calculate(df).rsi.iloc[-1],50)
    def test_model_and_gates(self):
        model=run(calculate(fixture()))
        self.assertTrue(0<=model['probability']<=1)
        self.assertGreater(model['metrics']['test_count'],100)
        self.assertEqual(decide(model,True)[0],'HOLD')
        self.assertEqual(decide(None,False)[0],'HOLD')
    def test_open_candle_excluded(self):
        df=fixture();extra=df.iloc[[-1]].copy();extra['timestamp']=pd.Timestamp.now(tz='UTC').floor('h')
        df=pd.concat([df,extra],ignore_index=True)
        result=analyze(validate(df,'1h'),'BTC/USDT','1h','ТЕСТ: синтетические данные')
        self.assertTrue(result['open_candle'])
        self.assertEqual(result['signal']['close_price'],float(df.close.iloc[-2]))
    def test_api_csv(self):
        client=TestClient(app)
        self.assertEqual(client.get('/api/health').status_code,200)
        df=fixture();df.timestamp=df.timestamp.map(lambda t:t.isoformat())
        response=client.post('/api/analysis/csv',files={'file':('test.csv',df.to_csv(index=False),'text/csv')})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(len(response.json()['candles']),1000)
        self.assertEqual(client.post('/api/analysis/csv',files={'file':('bad.csv','a,b\n1,2')}).status_code,400)

if __name__=='__main__': unittest.main()
