"""Получение неперсональной спецификации инструмента из подключённого demo MT5."""
import argparse
import json
import math
from .mt5_reader import MT5Reader, MT5Unavailable


def specification(reader, symbol):
    if symbol not in reader.symbols():
        raise MT5Unavailable('Инструмент не найден в списке терминала')
    info = reader._mt.symbol_info(symbol)
    if info is None:
        raise MT5Unavailable('Спецификация инструмента недоступна')
    # Только свойства инструмента. Данные аккаунта и реквизиты не сохраняются.
    fields = ('name', 'point', 'digits', 'trade_tick_size', 'trade_tick_value', 'trade_contract_size', 'currency_base', 'currency_profit', 'currency_margin', 'volume_min', 'volume_max', 'volume_step')
    result = {key: getattr(info, key, None) for key in fields}
    point = result['point']
    if point is None or not isinstance(point, (int, float)) or not math.isfinite(point) or point <= 0:
        raise MT5Unavailable('Размер пункта отсутствует или некорректен')
    return result


def main():
    parser = argparse.ArgumentParser(description='Чтение спецификации из MT5, без торговых операций')
    parser.add_argument('--symbol', default='GOLD')
    args = parser.parse_args()
    reader = None
    try:
        reader = MT5Reader()
        reader.connect()
        result = specification(reader, args.symbol)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (MT5Unavailable, ValueError) as exc:
        parser.exit(1, f'Ошибка MT5: {exc}\n')
    finally:
        if reader is not None:
            reader.close()

if __name__ == '__main__':
    main()
