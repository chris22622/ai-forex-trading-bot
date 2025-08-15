@echo off
title Deriv Trading Bot - Godlike Edition
echo ================================================
echo 🚀 DERIV TRADING BOT - GODLIKE EDITION  
echo 🤖 The Sophisticated AI-Powered Trading System
echo ================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo 🚀 Starting the sophisticated trading bot...
echo.

REM Run the main sophisticated bot
python main.py

echo.
echo ⏹️ Bot session ended.
pause
