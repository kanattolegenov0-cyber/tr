"""Обучение → отдельная калибровка → нетронутый тест. Между блоками удаляем одну цель."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from .feature_engineering import features

def run(df):
    x = features(df)
    # Последняя строка не имеет известной следующей цены и не становится ложным классом 0.
    y = (df.close.shift(-1) > df.close).astype(float)
    y.iloc[-1] = np.nan
    valid = x.notna().all(axis=1) & y.notna()
    xx, yy = x[valid], y[valid].astype(int)
    if len(xx) < 300:
        raise ValueError('Нужно не менее 500 закрытых свечей для обучения и независимой оценки')
    a, b = int(len(xx)*.6), int(len(xx)*.8)
    train, cal, test = slice(0,a-1), slice(a,b-1), slice(b,None)
    for part in (train,cal,test):
        if yy.iloc[part].nunique() < 2:
            raise ValueError('В одном из временных блоков отсутствует один класс; модель ненадёжна')
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(xx.iloc[train], yy.iloc[train])
    calibrator = LogisticRegression(C=1, random_state=42)
    calibrator.fit(model.decision_function(xx.iloc[cal]).reshape(-1,1), yy.iloc[cal])
    probability = calibrator.predict_proba(model.decision_function(xx.iloc[test]).reshape(-1,1))[:,1]
    truth = yy.iloc[test]
    pred = probability >= .5
    baseline_p = float(yy.iloc[train].mean())
    baseline_brier = brier_score_loss(truth, np.full(len(truth),baseline_p))
    metrics = {
      'accuracy': accuracy_score(truth,pred), 'precision': precision_score(truth,pred,zero_division=0),
      'recall': recall_score(truth,pred,zero_division=0), 'f1': f1_score(truth,pred,zero_division=0),
      'roc_auc': roc_auc_score(truth,probability), 'brier': brier_score_loss(truth,probability),
      'baseline_brier': baseline_brier, 'baseline_accuracy': accuracy_score(truth,np.full(len(truth),baseline_p>=.5)),
      'confusion_matrix': confusion_matrix(truth,pred,labels=[0,1]).tolist(),
      'train_count': len(xx.iloc[train]), 'calibration_count': len(xx.iloc[cal]), 'test_count':len(truth),
    }
    observed, predicted = calibration_curve(truth, probability, n_bins=5, strategy='quantile')
    metrics['calibration_curve'] = {'predicted':predicted.tolist(),'observed':observed.tolist()}
    latest = x.iloc[[-1]]
    if latest.isna().any().any():
        raise ValueError('Последняя свеча не содержит всех признаков')
    p = float(calibrator.predict_proba(model.decision_function(latest).reshape(-1,1))[0,1])
    reliable = bool(metrics['brier'] < baseline_brier and metrics['roc_auc'] > .5)
    # Вклад относится к базовой логистической модели, не является причинным объяснением.
    contributions = model.named_steps['standardscaler'].transform(latest)[0] * model.named_steps['logisticregression'].coef_[0]
    factors = sorted(zip(x.columns, contributions), key=lambda item:abs(item[1]), reverse=True)[:3]
    return {'probability':p,'reliable':reliable,'metrics':metrics,'factors':[{'feature':k,'contribution':float(v)} for k,v in factors], 'version':'logistic-chronological-v1'}
