"""Только чтение. Изолированный API не предоставляет торговые операции."""
import importlib
import pandas as pd
from .data import validate

class MT5Unavailable(RuntimeError):
    pass

class MT5Reader:
    def __init__(self):
        try:
            self._mt = importlib.import_module('MetaTrader5')
        except ImportError as exc:
            raise MT5Unavailable('MetaTrader5 недоступен; используйте CSV/Parquet') from exc
        self._connected = False

    def connect(self):
        if not self._mt.initialize():
            raise MT5Unavailable('Терминал MT5 недоступен')
        account = self._mt.account_info()
        if account is None or account.trade_mode != self._mt.ACCOUNT_TRADE_MODE_DEMO:
            self._mt.shutdown()
            raise MT5Unavailable('Для чтения требуется подключённый демосчёт MT5')
        self._connected = True

    def close(self):
        if self._connected:
            self._mt.shutdown(); self._connected = False

    def symbols(self):
        if not self._connected:
            raise MT5Unavailable('Нет подключения')
        values = self._mt.symbols_get()
        if values is None:
            raise MT5Unavailable('Не удалось получить инструменты')
        return [s.name for s in values if 'XAUUSD' in s.name.upper() or s.name.upper() == 'GOLD']

    def candles(self, symbol: str, timeframe: str, count: int):
        if not self._connected:
            raise MT5Unavailable('Нет подключения')
        if symbol not in self.symbols() or timeframe not in ('M5','M15','H1','H4') or not 1 <= count <= 100000:
            raise ValueError('Неверный инструмент, интервал или количество')
        # Начинаем с индекса 1: текущая незакрытая свеча исключается.
        rates = self._mt.copy_rates_from_pos(symbol, getattr(self._mt, 'TIMEFRAME_'+timeframe), 1, count)
        if rates is None or len(rates) == 0:
            raise MT5Unavailable('История недоступна; проверьте инструмент в терминале')
        frame = pd.DataFrame(rates).rename(columns={'time':'timestamp'})
        frame['timestamp'] = pd.to_datetime(frame.timestamp, unit='s', utc=True)
        return validate(frame.sort_values('timestamp'))

    def quote(self, symbol: str):
        if symbol not in self.symbols():
            raise ValueError('Неизвестный инструмент')
        tick = self._mt.symbol_info_tick(symbol)
        if tick is None:
            raise MT5Unavailable('Котировка недоступна')
        return {'bid':tick.bid, 'ask':tick.ask, 'spread':tick.ask-tick.bid, 'time_utc':pd.to_datetime(tick.time, unit='s', utc=True).isoformat()}
