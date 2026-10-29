from .market_data import INTERVALS

def decide(model, stale):
    if stale:
        return 'HOLD', 'Данные устарели. Получите свежие свечи.'
    if not model or not model['reliable']:
        return 'HOLD', 'Качество модели не подтверждено на отложенном периоде.'
    p = model['probability']
    direction = 'BUY' if p >= .60 else 'SELL' if p <= .40 else 'HOLD'
    return direction, f'Оценка вероятности роста следующей свечи: {p:.1%}. Пороги 60% и 40% учебные. Это не гарантия результата.'
