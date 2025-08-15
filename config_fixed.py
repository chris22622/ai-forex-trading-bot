#!/usr/bin/env python3
"""
🚀 AGGRESSIVE TRADING CONFIGURATION FOR SMALL ACCOUNT GROWTH
Optimized for accounts under $100 to enable actual growth
"""

import os
from datetime import datetime

# 🛡️ SAFER TRADING SETTINGS FOR $42 ACCOUNT (Post-Drawdown Protection)
TRADE_AMOUNT = 0.5          # $0.50 trades (0.01 lots) - ULTRA SAFE
AUTO_TAKE_PROFIT = 0.75     # $0.75 profit target - Quick profits
AUTO_STOP_LOSS = 1.0        # $1.00 stop loss - Tight control
MIN_PROFIT_THRESHOLD = 0.25 # $0.25 minimum profit - Accept small wins

# 🛡️ SAFER RISK MANAGEMENT (For $42 Account)
MAX_DAILY_LOSS = 8.0        # $8 daily loss limit (20% of $42)
MAX_PORTFOLIO_LOSS = 3.0    # $3 portfolio loss (7% protection)
MAX_CONSECUTIVE_LOSSES = 100   # Stop after 100 losses in a row
MAX_CONCURRENT_TRADES = 1    # Only 1 trade at a time for safety

# 🛡️ SAFER TIMING (Faster Exits)
MAX_TRADE_AGE_MINUTES = 15   # 15 minutes per trade (was 45) - Quick exits
AUTO_TIME_LIMIT = 15         # Auto time limit - Fast management
MIN_TRADE_INTERVAL = 60      # 60 seconds between trades (was 10) - Less frequent

# 🛡️ SAFER SYMBOL SELECTION (Avoid Crash Indices)
PRIMARY_TRADING_SYMBOLS = [
    'Volatility 10 Index',    # Most stable
    'Volatility 25 Index'     # Moderately safe
    # Removed volatile symbols: Volatility 75, 100, Boom, Crash
]
MAX_SYMBOLS_CONCURRENT = 1   # Only 1 symbol at a time for safety
DEFAULT_SYMBOL = 'Volatility 10 Index'  # Safest symbol

# 🛡️ SAFER AI SETTINGS (Higher Confidence Required)
AI_CONFIDENCE_THRESHOLD = 0.65  # Higher threshold for better trades (was 0.05)
RSI_BUY_THRESHOLD = 30         # Buy only when RSI < 30 (oversold)
RSI_SELL_THRESHOLD = 70        # Sell only when RSI > 70 (overbought)

# 🛡️ SAFER PROFIT PROTECTION (For $42 Account)
PROFIT_PROTECTION_ENABLED = True
PROFIT_PROTECTION_THRESHOLD = 2.0     # Stop at +$2 profit (5% of account)
PROFIT_PROTECTION_MAX_THRESHOLD = 5.0  # Emergency stop at +$5 (12% gain)

# 🛡️ MAXIMUM DRAWDOWN PROTECTION
MAX_DRAWDOWN_PERCENTAGE = 0.15  # Stop at 15% drawdown (was 20%)

# MT5 Connection Settings
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '62233085'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', 'Goon22622$')
MT5_SERVER = os.getenv('MT5_SERVER', 'DerivVU-Server')
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_DEMO_MODE = True  # Set to False for live trading

# 🛡️ SAFER POSITION SIZING (For $42 Account)
BASE_RISK_PERCENTAGE = 0.05  # 5% risk only (was 15%)
MAX_LOT_SIZE = 0.02          # Maximum 0.02 lots (was 0.10)
MIN_LOT_SIZE = 0.01          # Minimum 0.01 lots (back to ultra safe)
MT5_LOT_SIZE = 0.01          # Default lot size for MT5 integration

# Telegram Settings
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '8024680623:AAEc0wNz5ALSrY8yb4PMWS0Wh95zz9IHUXE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '7543942571')

# Technical Indicators Settings
EMA_FAST = 12
EMA_SLOW = 26
MA_SHORT = 20
MA_LONG = 50
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Buffer Settings
PRICE_BUFFER_SIZE = 100

# Auto Trade Management
ENABLE_AUTO_TRADE_MANAGEMENT = True
ENABLE_HEDGE_PREVENTION = True
ENABLE_MULTI_SYMBOL_TRADING = False  # Single symbol for safety

# Smart Risk Management
ENABLE_SMART_RISK_MANAGEMENT = True
MAX_LOSS_PER_TRADE = 1.0    # $1 max loss per trade
RISK_CHECK_INTERVAL = 30

# Position Management
POSITION_SIZE_MODE = 'fixed'
DYNAMIC_POSITION_SIZING = False
