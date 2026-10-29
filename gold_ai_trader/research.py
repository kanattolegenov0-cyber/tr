"""Оценка моделей по последовательным периодам без оптимизации на последнем тесте."""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .data import load
from .forecast import prepare


def period(frame, start_cal, start_test, end_test, horizon=4, cost=.30):
    x, y, valid = prepare(frame, horizon, cost)
    idx = x.index
    masks = [valid & (idx < start_cal-horizon), valid & (idx >= start_cal) & (idx < start_test-horizon), valid & (idx >= start_test) & (idx < end_test-horizon)]
    splits = [(x.loc[mask], y.loc[mask].astype(int)) for mask in masks]
    if min(len(item[0]) for item in splits) < 60 or any(set(item[1]) != {0,1,2} for item in splits[:2]):
        raise ValueError('недостаточно наблюдений или не представлены все три класса в обучении и калибровке')
    (train, yt), (cal, yc), (test, ye) = splits
    baseline = np.tile(np.bincount(yt, minlength=3)/len(yt), (len(ye), 1))
    models = {'XGBoost': XGBClassifier(n_estimators=80, max_depth=2, learning_rate=.05, objective='multi:softprob', num_class=3, n_jobs=1, random_state=42), 'Logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))}
    results = {'Baseline': {'log_loss': float(log_loss(ye, baseline, labels=[0,1,2])), 'brier': float(np.mean(np.sum((baseline-np.eye(3)[ye])**2,axis=1)))}}
    for name, model in models.items():
        model.fit(train, yt)
        calibration = LogisticRegression(max_iter=2000, random_state=42)
        calibration.fit(model.predict_proba(cal), yc)
        p = calibration.predict_proba(model.predict_proba(test))
        results[name] = {'log_loss':float(log_loss(ye,p,labels=[0,1,2])), 'brier':float(np.mean(np.sum((p-np.eye(3)[ye])**2,axis=1))), 'f1_macro':float(f1_score(ye,np.argmax(p,axis=1),labels=[0,1,2],average='macro',zero_division=0))}
    return {'train':len(yt),'calibration':len(yc),'test':len(ye),'test_from':str(frame.timestamp.iloc[start_test]),'test_to':str(frame.timestamp.iloc[end_test-1]),'results':results}


def main():
    p=argparse.ArgumentParser(description='Проверка прогноза на нескольких временных периодах M15')
    p.add_argument('file'); p.add_argument('--cost', type=float, required=True); p.add_argument('--horizon',type=int,default=4)
    args=p.parse_args()
    try:
        frame=load(args.file); n=len(frame)
        # Проверочные периоды не пересекают финальный тест; параметры не подбираются.
        for label,a,b,c in [('Период 1',.4,.5,.6),('Период 2',.5,.6,.7),('Период 3',.6,.7,.8),('Финальный тест',.6,.8,1.0)]:
            print(label)
            try:
                report=period(frame,int(n*a),int(n*b),int(n*c),args.horizon,args.cost)
                print('Период UTC:',report['test_from'],'—',report['test_to'], '; строк:',report['test'])
                for name, metric in report['results'].items():
                    print(f"  {name}: log loss={metric['log_loss']:.4f}; Brier={metric['brier']:.4f}" + (f"; F1={metric['f1_macro']:.4f}" if 'f1_macro' in metric else ''))
            except ValueError as exc:
                print('  Не выполнено:',exc)
        print('Меньше log loss и Brier — лучше. Это не оценка прибыли; статус HOLD.')
    except ValueError as exc:
        p.error(str(exc))

if __name__ == '__main__': main()
