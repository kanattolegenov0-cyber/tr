import json
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .market_data import fetch, read_csv, SYMBOLS, INTERVALS
from .indicators import calculate, EXPLANATIONS
from .ml_models import run
from .signal_engine import decide
from .ai_analyst import explain

app = FastAPI(title='AI Trading Assistant', version='0.1.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'], allow_methods=['GET','POST'],allow_headers=['*'])

@app.get('/api/health')
def health():
    return {'status':'ok','mode':'research-only'}

@app.get('/api/config')
def config():
    return {'symbols':SYMBOLS,'timeframes':list(INTERVALS),'explanations':EXPLANATIONS}

def analyze(df, symbol, timeframe, source):
    closed = calculate(df[df.closed].reset_index(drop=True))
    if closed.empty:
        raise ValueError('Нет закрытых свечей')
    last = closed.iloc[-1]
    step = pd.Timedelta(seconds=INTERVALS[timeframe])
    stale = bool(pd.Timestamp.now(tz='UTC') > last.timestamp + 2*step)
    model, issue = None, None
    try:
        model = run(closed)
    except ValueError as exc:
        issue = str(exc)
    direction, reason = decide(model, stale)
    if issue:
        reason += ' ' + issue
    prior_time = last.timestamp - pd.Timedelta(hours=24)
    prior = closed.loc[closed.timestamp == prior_time, 'close']
    change = float((last.close/prior.iloc[0]-1)*100) if len(prior) else None
    candles = json.loads(df.drop(columns='closed').assign(timestamp=df.timestamp.map(lambda t:int(t.timestamp()))).to_json(orient='records'))
    values = {key: None if pd.isna(last[key]) else float(last[key]) for key in EXPLANATIONS}
    return {'symbol':symbol,'timeframe':timeframe,'source':source,'stale':stale,
      'open_candle':bool((~df.closed).any()),'last_price':float(df.close.iloc[-1]),
      'updated_at':df.timestamp.iloc[-1].isoformat(),'change_24h':change,'candles':candles,
      'indicators':values,'explanations':EXPLANATIONS,'model':model,'model_issue':issue,
      'signal':{'direction':direction,'at':(last.timestamp+step).isoformat(),'close_price':float(last.close),
      'horizon':timeframe,'explanation':explain(direction,reason),'uncertainty':'Высокая: рынок меняется, историческая оценка не гарантирует точность.'}}

@app.get('/api/analysis')
def market(symbol: str='BTC/USDT', timeframe: str='1h'):
    try:
        return analyze(fetch(symbol,timeframe),symbol,timeframe,'Публичный API Binance')
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc

@app.post('/api/analysis/csv')
async def csv(file:UploadFile=File(...), symbol:str='BTC/USDT', timeframe:str='1h'):
    if symbol not in SYMBOLS or timeframe not in INTERVALS:
        raise HTTPException(400,'Неподдерживаемый актив или таймфрейм')
    raw = await file.read(5_000_001)
    if len(raw)>5_000_000:
        raise HTTPException(413,'Максимальный размер CSV — 5 МБ')
    try:
        return analyze(read_csv(raw,timeframe),symbol,timeframe,'CSV пользователя; источник не подтверждён')
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from exc
