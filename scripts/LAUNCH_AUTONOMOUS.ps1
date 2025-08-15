# Autonomous Trading Bot Launcher
Write-Host "============================================" -ForegroundColor Green
Write-Host "  AUTONOMOUS TRADING BOT - AUTOPILOT MODE" -ForegroundColor Green  
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "🤖 Starting completely autonomous trading bot..." -ForegroundColor Yellow
Write-Host "✅ Runs 24/7 without any input needed" -ForegroundColor Green
Write-Host "✅ Automatically handles all errors" -ForegroundColor Green
Write-Host "✅ Uses smart simulation mode" -ForegroundColor Green
Write-Host ""

# Set environment variables for autonomous mode
$env:AUTONOMOUS_MODE = "true"
$env:FORCE_SIMULATION = "true"

# Start the bot
Write-Host "🚀 Launching bot..." -ForegroundColor Cyan
python main.py

Write-Host ""
Write-Host "Bot has stopped. Press any key to exit..." -ForegroundColor Yellow
Read-Host
