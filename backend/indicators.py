import numpy as np

EXPLANATIONS = {
 'sma20': 'Средняя цена последних 20 свечей сглаживает колебания. Не гарантирует продолжение тренда.',
 'sma50': 'Средняя цена 50 свечей помогает оценить средний тренд.',
 'sma200': 'Средняя цена 200 свечей показывает более длительный тренд.',
 'ema12': 'Средняя за 12 свечей с большим весом недавних цен.',
 'ema26': 'Средняя за 26 свечей с большим весом недавних цен.',
 'rsi': 'RSI сравнивает недавний рост и падение. Выше 70 — возможная перекупленность, а не обещание падения.',
 'macd': 'Разность EMA 12 и EMA 26 описывает импульс цены.',
 'macd_signal': 'Средняя MACD за 9 периодов для сравнения с текущим импульсом.',
 'bb_upper': 'Верхняя граница: средняя за 20 свечей плюс два стандартных отклонения.',
 'bb_lower': 'Нижняя граница: средняя за 20 свечей минус два стандартных отклонения.',
 'atr': 'Средний истинный диапазон за 14 свечей измеряет размер колебаний цены.',
 'volatility': 'Стандартное отклонение доходностей 20 свечей; это риск колебаний, а не направление.',
 'volume_change': 'Относительное изменение объёма к предыдущей свече.',
 'return_1': 'Относительное изменение цены за одну свечу.',
}

def calculate(df):
    out = df.copy()
    close = out.close
    for n in (20, 50, 200):
        out[f'sma{n}'] = close.rolling(n).mean()
    for n in (12, 26):
        out[f'ema{n}'] = close.ewm(span=n, adjust=False, min_periods=n).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    out['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out.loc[(loss == 0) & (gain > 0), 'rsi'] = 100
    out.loc[(loss == 0) & (gain == 0), 'rsi'] = 50
    out['macd'] = out.ema12 - out.ema26
    out['macd_signal'] = out.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    sd = close.rolling(20).std(ddof=0)
    out['bb_upper'] = out.sma20 + 2*sd
    out['bb_lower'] = out.sma20 - 2*sd
    tr = __import__('pandas').concat([out.high-out.low, (out.high-close.shift()).abs(), (out.low-close.shift()).abs()], axis=1).max(axis=1)
    out['atr'] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    out['return_1'] = close.pct_change(fill_method=None)
    out['volatility'] = out.return_1.rolling(20).std()
    out['volume_change'] = out.volume.pct_change(fill_method=None).replace([np.inf,-np.inf], np.nan)
    return out
