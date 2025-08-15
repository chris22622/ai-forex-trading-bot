@echo off
title Deriv Trading Bot Launcher
color 0A

echo ================================================
echo       DERIV TRADING BOT LAUNCHER
echo ================================================
echo.

echo [INFO] Checking Python environment...
cd /d "%~dp0"

echo [INFO] Starting trading bot...
echo.
echo ================================================
echo       BOT IS STARTING...
echo ================================================
echo.

C:/Users/Chris/scalping_deriv_bot/.venv/Scripts/python.exe main.py

echo.
echo ================================================
echo       BOT HAS STOPPED
echo ================================================
echo.
pause
