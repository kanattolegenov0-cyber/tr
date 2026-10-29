import numpy as np

FEATURES = ['return_1','trend20','trend50','trend200','rsi_scaled','macd_scaled','atr_scaled','volatility','volume_change']

def features(df):
    x = df[['return_1','volatility','volume_change']].copy()
    for n in (20,50,200):
        x[f'trend{n}'] = df.close / df[f'sma{n}'] - 1
    x['rsi_scaled'] = df.rsi / 100
    x['macd_scaled'] = df.macd / df.close
    x['atr_scaled'] = df.atr / df.close
    return x[FEATURES].replace([np.inf,-np.inf], np.nan)
