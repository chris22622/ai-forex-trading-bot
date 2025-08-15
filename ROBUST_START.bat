@echo off
echo 🚀 ROBUST TRADING BOT LAUNCHER
echo ================================
echo.

echo 📅 Date: %DATE%
echo 🕒 Time: %TIME%
echo.

echo 🔍 Checking Python...
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo.

echo 🔍 Checking current directory...
if not exist "main.py" (
    echo ❌ main.py not found! Run this from the bot directory
    pause
    exit /b 1
)
echo.

echo 🤖 Starting robust trading bot...
echo ⚠️ This will handle MT5 connection issues automatically
echo.

python robust_start.py

echo.
echo 🔚 Bot startup completed
pause
