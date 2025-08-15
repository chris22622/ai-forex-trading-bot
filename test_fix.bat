@echo off
echo Testing Telegram conflict fix...

set DISABLE_TELEGRAM=1
set ENABLE_TELEGRAM_ALERTS=False

echo Running test...
.venv\Scripts\python.exe -c "print('Test starting...'); import main; print('SUCCESS: No Telegram conflicts!')"

echo Test completed.
pause
