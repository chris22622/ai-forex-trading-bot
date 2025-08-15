@echo off
title Deriv MT5 Trading Bot Launcher
echo.
echo ================================
echo   DERIV MT5 TRADING BOT
echo ================================
echo.
echo Starting MT5 bot launcher...
echo.

cd /d "%~dp0"
python launch_mt5_bot.py

echo.
echo Bot session ended.
pause
