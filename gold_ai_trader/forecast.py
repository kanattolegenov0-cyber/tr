"""Исследовательский прогноз: обучение, калибровка и проверка разделены по времени."""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix, f1_score
from xgboost import XGBClassifier
from .data import load
from .indicators import calculate

NAMES = ['SELL', 'HOLD', 'BUY']

def prepare(df, horizon, cost):
    if horizon < 1 or not np.isfinite(cost) or cost <= 0:
        raise ValueError('Горизонт и порог издержек должны быть положительными')
    d = calculate(df)
    x = pd.DataFrame(index=d.index)
    for n in (20, 50, 200):
        x[f'ema{n}'] = d.close / d[f'ema{n}'] - 1
    for col in ('rsi','return_1','return_3','return_5','return_10','volatility'):
        x[col] = d[col]
    x['atr'] = d.atr / d.close
    x['macd'] = d.macd / d.close
    x['body'] = d.body / d.close
    x['upper_wick'] = d.upper_wick / d.close
    x['lower_wick'] = d.lower_wick / d.close
    # Цель — движение close, а не исполнимая доходность. Пропуски в будущем горизонте исключаются.
    change = d.close.shift(-horizon) - d.close
    elapsed = d.timestamp.shift(-horizon) - d.timestamp
    target = pd.Series(np.where(change > cost, 2, np.where(change < -cost, 0, 1)), index=d.index).where(change.notna() & (elapsed.dt.total_seconds() == 900*horizon))
    valid = x.notna().all(axis=1) & target.notna()
    return x, target, valid

def evaluate(df, horizon=4, cost=0.30):
    x, y, valid = prepare(df, horizon, cost)
    # Разделяем по исходным индексам; зазор исключает пересечение целевых горизонтов.
    n = len(df); a = int(n*.6); b = int(n*.8)
    masks = [valid & (x.index < a-horizon), valid & (x.index >= a) & (x.index < b-horizon), valid & (x.index >= b)]
    parts = [(x.loc[m], y.loc[m].astype(int)) for m in masks]
    if min(len(p[0]) for p in parts) < 60 or any(set(p[1]) != {0,1,2} for p in parts):
        raise ValueError('Недостаточно данных каждого из трёх классов в обучении, калибровке или тесте; загрузите больше M15 свечей или проверьте порог')
    (train, yt), (cal, yc), (test, ytest) = parts
    model = XGBClassifier(n_estimators=80, max_depth=2, learning_rate=.05, objective='multi:softprob', num_class=3, n_jobs=1, random_state=42)
    model.fit(train, yt)
    # Калибратор учится только на отдельном последующем временном блоке.
    calibrator = LogisticRegression(max_iter=2000, random_state=42)
    calibrator.fit(model.predict_proba(cal), yc)
    p = calibrator.predict_proba(model.predict_proba(test))
    if list(calibrator.classes_) != [0,1,2]:
        raise ValueError('Калибратор не содержит все классы')
    labels = np.argmax(p, axis=1)
    baseline = np.tile(np.bincount(yt, minlength=3) / len(yt), (len(ytest),1))
    brier = float(np.mean(np.sum((p-np.eye(3)[ytest])**2, axis=1)))
    metrics = {'train':len(yt), 'calibration':len(yc), 'test':len(ytest), 'log_loss':float(log_loss(ytest,p,labels=[0,1,2])), 'baseline_log_loss':float(log_loss(ytest,baseline,labels=[0,1,2])), 'brier':brier, 'f1_macro':float(f1_score(ytest,labels,average='macro',zero_division=0)), 'confusion_matrix':confusion_matrix(ytest,labels,labels=[0,1,2]).tolist()}
    latest = x.iloc[[-1]]
    if latest.isna().any().any():
        raise ValueError('Не хватает признаков для последней свечи')
    current = calibrator.predict_proba(model.predict_proba(latest))[0]
    return metrics, dict(zip(NAMES, map(float,current)))

def main():
    parser = argparse.ArgumentParser(description='Исследовательский XGBoost-прогноз на четыре свечи M15')
    parser.add_argument('file'); parser.add_argument('--cost',type=float,required=True,help='порог изменения цены в USD за единицу, включая оценку издержек')
    parser.add_argument('--horizon',type=int,default=4)
    args=parser.parse_args()
    try:
        df=load(args.file); metrics, forecast=evaluate(df,args.horizon,args.cost)
    except ValueError as exc:
        parser.error(str(exc))
    print('Последняя свеча (время открытия UTC):',df.timestamp.iloc[-1].isoformat())
    print('Учебные оценки, не торговый сигнал:', ', '.join(f'{k}={v:.1%}' for k,v in forecast.items()))
    print('Проверка на отложенном периоде:',metrics)
    print('HOLD: прогноз исследовательский, надёжность и исполнимая доходность не подтверждены. Данные могут быть устаревшими.')

if __name__ == '__main__': main()
