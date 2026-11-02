@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-gold\Scripts\python.exe" (
  echo Python environment .venv-gold is missing. See gold_ai_trader\README.md.
  exit /b 1
)
".venv-gold\Scripts\python.exe" -m gold_ai_trader.ask_forecast %*
exit /b %errorlevel%
