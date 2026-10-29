# AI Trading Assistant — этап 1

Русскоязычный исследовательский MVP. Анализ криптовалют через публичный Binance/ccxt или CSV, свечной график и объём, 14 индикаторов с объяснениями, Logistic Regression с калибровкой, независимые метрики и BUY/SELL/HOLD. Никаких реальных ордеров или обещаний прибыли.

## Запуск

Python 3.11+ и Node.js 20.19+ (или 22.12+).

```bash
cd /workspace/tr
python -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Во втором терминале:

```bash
cd /workspace/tr/frontend
npm install
npm run dev
```

Frontend порт 5173, backend порт 8000, API документация `/docs`. Vite проксирует `/api` на backend. В production API не следует открывать публично без аутентификации и ограничений нагрузки. API-ключи для котировок не нужны.

## Проверки

```bash
cd /workspace/tr
python -m unittest discover -s tests -v
cd frontend
npm run build
```

Тесты используют явно синтетические данные только для проверки кода, не для интерфейса или заявлений о рынке.

## CSV

UTF-8: `timestamp,open,high,low,close,volume`. Время открытия свечи — ISO 8601 UTC или Unix **миллисекунды**, строки без пропусков интервала и дублей. Максимум 5 МБ. От 500 закрытых свечей для модели (рекомендуется 1000). Пропуски не заполняются искусственными ценами; неверный файл отклоняется. Объём в единицах базового актива.

Цена на панели может относиться к открытой свече; цена сигнала всегда относится к последней закрытой. Время сигнала — момент закрытия, не момент запроса. Источник и время показываются явно. Прогноз не равен рекомендации купить; даже прохождение минимального фильтра качества не доказывает надёжность в будущем.

## Файлы и ограничения

`backend/market_data.py` — адаптер и проверка; `indicators.py` — расчёты; `feature_engineering.py` — причинные признаки; `ml_models.py` — обучение и метрики; `signal_engine.py` — пороги и блокировки; `ai_analyst.py` — объяснение; `main.py` — API; `frontend/src` — интерфейс; `tests` — проверки; `docs/architecture.md` — план следующих этапов.

Пока отсутствуют paper trading, backtesting, сохранение моделей, walk-forward, сравнение нескольких моделей, чат, Academy, Telegram и Docker. Модель обучается заново при запросе, данные и результаты не сохраняются. CSV не подтверждает подлинность котировок. Frontend собран успешно; npm-зависимости закреплены в package-lock.json. Семь тестов backend прошли; HTTP-загрузка 1000 тестовых свечей вернула 200. Публичный Binance в текущей среде блокируется сетевой политикой (CONNECT 403); до разрешения домена используйте CSV. Визуальная проверка в браузере ещё не выполнена.

## Запуск в Windows (командная строка CMD)

Установите Python 3.11 или новее и Node.js 20.19+ / 22.12+. При установке Python включите добавление Python в PATH. Распакуйте архив проекта, например в `C:\Users\KM1\tr`, и откройте CMD:

```bat
cd /d C:\Users\KM1\tr
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```

Оставьте этот терминал открытым. В другом окне CMD:

```bat
cd /d C:\Users\KM1\tr\frontend
npm ci
npm run dev
```

Откройте в браузере адрес, который выведет Vite (обычно `http://localhost:5173`). Для остановки нажмите Ctrl+C в каждом терминале. Если `py` или `npm` не распознаны, установите соответствующую программу и откройте новое окно CMD. Путь в командах должен совпадать с фактической папкой после распаковки. Создавайте `.venv` именно в Windows: окружение из Linux несовместимо.

## Gold AI Trader

Отдельное исследовательское приложение XAUUSD находится в [gold_ai_trader](gold_ai_trader/README.md). Оно содержит offline-анализ, учебный EMA backtest и интерфейс чтения MT5; аудит исходного xaubot-ai — в [AUDIT.md](gold_ai_trader/AUDIT.md).
