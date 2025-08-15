#!/usr/bin/env python3
"""
MT5 AI Trading Bot Configuration
Copy this file to src/config.py and update with your settings
"""

import os

# 📱 TELEGRAM NOTIFICATIONS CONFIGURATION
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")
ENABLE_TELEGRAM_NOTIFICATIONS = True

# ⚠️ CRITICAL SAFETY SETTING ⚠️
# Set to False for LIVE TRADING with real money!
DEMO_MODE = True  # True = Demo training (unlimited risk), False = Live trading (protected)

# 🎯 TRADING SETTINGS
TRADE_AMOUNT = 50.0  # Trade amount in USD
AUTO_TAKE_PROFIT = 1.5  # Take profit target
AUTO_STOP_LOSS = 2.0  # Stop loss amount
MIN_PROFIT_THRESHOLD = 0.5  # Minimum profit threshold

# 🎯 RISK MANAGEMENT
MAX_DAILY_LOSS = 50.0  # Maximum daily loss limit
MAX_PORTFOLIO_LOSS = 20.0  # Maximum portfolio loss
MAX_CONSECUTIVE_LOSSES = 10  # Stop after consecutive losses
MAX_CONCURRENT_TRADES = 8  # Maximum concurrent trades

# 🛡️ TIMING SETTINGS
MAX_TRADE_AGE_MINUTES = 15  # Maximum trade duration
AUTO_TIME_LIMIT = 15  # Auto time limit
MIN_TRADE_INTERVAL = 60  # Minimum interval between trades

# 🎯 TRADING SYMBOLS
PRIMARY_TRADING_SYMBOLS = [
    "Volatility 10 Index",
    "Volatility 25 Index",
    "Volatility 75 Index",
    "Volatility 100 Index",
    "Crash 500 Index",
    "Boom 500 Index",
]
MAX_SYMBOLS_CONCURRENT = 8
DEFAULT_SYMBOL = "Volatility 25 Index"

# 🎯 AI SETTINGS
MIN_CONFIDENCE = 0.05
MIN_FREE_MARGIN = 5.0
AI_MODEL_TYPE = "randomforest"
RSI_BUY_THRESHOLD = 35
RSI_SELL_THRESHOLD = 65

# MT5 Connection Settings
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "12345678"))  # Your MT5 account ID
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "YOUR_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER", "Deriv-Demo")  # MT5 server
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_DEMO_MODE = True
MT5_REAL_TRADING = True
MT5_MAX_LOT_SIZE = 1.0
MT5_SLIPPAGE = 3
MT5_MAGIC_NUMBER = 123456
MT5_DEFAULT_SYMBOL = "Volatility 25 Index"

# Position Sizing
BASE_RISK_PERCENTAGE = 0.5
MAX_LOT_SIZE = 1.0
MIN_LOT_SIZE = 0.01
MT5_LOT_SIZE = 0.1

# Technical Indicators
EMA_FAST = 12
EMA_SLOW = 26
MA_SHORT = 20
MA_LONG = 50
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Technical Analysis Constants
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_NEUTRAL_LOW = 40
RSI_NEUTRAL_HIGH = 60

# Buffer Settings
PRICE_BUFFER_SIZE = 100

# Feature Flags
ENABLE_AUTO_TRADE_MANAGEMENT = True
ENABLE_HEDGE_PREVENTION = True
ENABLE_MULTI_SYMBOL_TRADING = True
ENABLE_SMART_RISK_MANAGEMENT = True
ENFORCE_MARKET_HOURS = False
DEMO_MODE_ENABLED = True

# Risk Management
MAX_LOSS_PER_TRADE = 1.0
RISK_CHECK_INTERVAL = 30
MAX_DRAWDOWN_PERCENTAGE = 0.15
PROFIT_PROTECTION_ENABLED = True
PROFIT_PROTECTION_THRESHOLD = 2.0
PROFIT_PROTECTION_MAX_THRESHOLD = 5.0

# Volume Management
VOLUME_GUARD_MODE = "smart_filter"
MAX_SAFE_LOT_SIZE = 2.0

# Symbol Preferences
PREFERRED_SYMBOLS = [
    "Volatility 10 Index",
    "Volatility 25 Index",
    "Volatility 75 Index",
    "Volatility 100 Index",
    "Crash 500 Index",
    "Boom 500 Index",
]

AVOID_HIGH_MIN_LOT_SYMBOLS = [
    "Volatility 50 Index",
    "Crash 1000 Index",
    "Boom 1000 Index",
]
