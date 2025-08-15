@echo off
title Deriv Trading Bot - Unified Launcher
echo ================================================
echo 🚀 DERIV TRADING BOT - UNIFIED LAUNCHER
echo 🤖 Choose: WebSocket API, MT5, or Demo
echo ================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.9+ and add it to PATH
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Run validation first
echo 🔍 Validating setup...
python validate_setup.py
if errorlevel 1 (
    echo ❌ Setup validation failed!
    echo Please check your configuration and try again.
    pause
    exit /b 1
)

echo.
echo ✅ Validation passed!
echo.

REM Ask user for confirmation
echo 🎯 Ready to start the trading bot!
echo.
echo IMPORTANT:
echo - Make sure your API credentials are configured in config.py
echo - Check that PAPER_TRADING is set correctly
echo - Ensure you understand the risks involved
echo.
set /p confirm="Do you want to continue? (y/N): "
if /i not "%confirm%"=="y" (
    echo ❌ Startup cancelled by user
    pause
    exit /b 0
)

echo.
echo 🚀 Starting Deriv Trading Bot...
echo 📊 Press Ctrl+C to stop the bot
echo.

REM Start the unified launcher
python unified_launcher.py

REM If we get here, the bot has stopped
echo.
echo ⏹️ Bot session ended
echo.
pause