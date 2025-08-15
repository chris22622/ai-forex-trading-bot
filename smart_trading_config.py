"""
Smart Trading Configuration for Maximum Profitability
This file contains optimized settings for better trading performance
"""

import os

# ==================== ESSENTIAL CONSTANTS ====================
# Trading mode
DEMO_MODE = False  # Live trading for real profits

# API Credentials
DERIV_LIVE_API_TOKEN = "g9kxuLTUWxrHGNe"  # Live token with ALL permissions
DERIV_DEMO_API_TOKEN = "s8Sww40XKyKMp0x"  # Demo token with ALL permissions
DERIV_API_TOKEN = DERIV_DEMO_API_TOKEN if DEMO_MODE else DERIV_LIVE_API_TOKEN

# WebSocket URLs
DERIV_WS_URLS = [
    "wss://ws.derivws.com/websockets/v3",
    "wss://ws.binaryws.com/websockets/v3",
    "wss://ws.deriv.com/websockets/v3"
]
DERIV_WS_URL = DERIV_WS_URLS[0]

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
ENABLE_TELEGRAM_ALERTS = True

# Check if Telegram should be disabled
TELEGRAM_ENABLED = os.getenv('DISABLE_TELEGRAM', '0').lower() not in ['1', 'true', 'yes']
if not TELEGRAM_ENABLED:
    TELEGRAM_BOT_TOKEN = ""
    ENABLE_TELEGRAM_ALERTS = False

# MT5 Configuration
LIVE_ACCOUNT_NUMBER = 62233085
LIVE_PASSWORD = "Goon22622$"
LIVE_SERVER = "DerivVU-Server"

MT5_DEMO_MODE = False
MT5_REAL_TRADING = True
MT5_LOGIN = LIVE_ACCOUNT_NUMBER
MT5_PASSWORD = LIVE_PASSWORD
MT5_SERVER = LIVE_SERVER
MT5_LOT_SIZE = 0.01

# Technical indicator periods (Required by indicators.py)
RSI_PERIOD = 14
EMA_FAST = 12
EMA_SLOW = 26
MA_SHORT = 10
MA_LONG = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# AI Model Configuration (Required by ai_model.py)
AI_MODEL_TYPE = "enhanced"  # Type of AI model to use

# Trading timing
MIN_TRADE_INTERVAL = 30  # Minimum seconds between trades

# AI Confidence threshold - Higher threshold = More selective trades
AI_CONFIDENCE_THRESHOLD = 0.6  # Increased from 0.5 for better quality trades

# Auto-management settings for optimal profit/loss ratios
ENABLE_AUTO_TRADE_MANAGEMENT = True
AUTO_TAKE_PROFIT = 1.2        # Lower target for quicker wins ($1.20)
AUTO_STOP_LOSS = 1.8          # Tighter stop loss ($1.80)
AUTO_TIME_LIMIT = 12          # Shorter time limit (12 minutes)
MIN_PROFIT_THRESHOLD = 0.4    # Minimum profit to hold ($0.40)

# Enhanced risk management
MAX_CONSECUTIVE_LOSSES = 100  # Keep high to prevent blocking
MAX_DAILY_LOSS = 50.0        # Daily loss limit ($50)
TRADE_AMOUNT = 3.0           # Base trade amount ($3)

# Smart concurrent trading (Balanced approach)
MAX_CONCURRENT_TRADES = 6    # Reduced from 10 for better management
ENABLE_HEDGE_PREVENTION = False  # Keep disabled for flexibility
ALLOW_SAME_DIRECTION_STACKING = True

# Multi-symbol trading (Optimized selection)
ENABLE_MULTI_SYMBOL_TRADING = True
MAX_SYMBOLS_CONCURRENT = 6   # Reduced for better focus

# Symbol configuration
DEFAULT_SYMBOL = "Volatility 75 Index"  # Fallback symbol

# 🎯 SMART SYMBOL SELECTION (High-performance symbols)
PRIMARY_TRADING_SYMBOLS = [
    "Volatility 10 Index",      # Low volatility, steady moves
    "Volatility 25 Index",      # Medium volatility
    "Volatility 75 Index",      # Higher volatility for bigger moves
    "Volatility 100 (1s) Index", # High-frequency opportunities
    "Boom 500 Index",           # Spike patterns
    "Crash 500 Index",          # Crash patterns
]

# Symbol-specific lot multipliers (Optimized for each symbol)
SYMBOL_LOT_MULTIPLIERS = {
    "Volatility 10 Index": 1.2,      # Slightly larger positions for stable symbol
    "Volatility 25 Index": 1.1,      # Standard+ sizing
    "Volatility 75 Index": 0.9,      # Smaller for higher volatility
    "Volatility 100 (1s) Index": 0.8, # Smaller for high frequency
    "Boom 500 Index": 0.7,           # Smaller for spike patterns
    "Crash 500 Index": 0.7,          # Smaller for crash patterns
}

# 📊 PERFORMANCE TRACKING SETTINGS
ENABLE_PERFORMANCE_TRACKING = True
TRACK_WIN_RATE_HISTORY = 20    # Track last 20 trades for win rate
ADAPTIVE_SIZING_ENABLED = True  # Enable smart position sizing
STREAK_BONUS_ENABLED = True     # Enable winning streak bonuses

# 🚨 SAFETY FEATURES
ENABLE_BREAKEVEN_PROTECTION = True    # Close at breakeven after 10 minutes
ENABLE_TRAILING_STOPS = True          # Use trailing stops for profitable trades
ENABLE_DYNAMIC_TARGETS = True         # Adjust targets based on symbol volatility

# Smart trading hours (Optional - can be disabled)
ENABLE_TRADING_HOURS = False
TRADING_START_HOUR = 8        # Start trading at 8 AM
TRADING_END_HOUR = 22         # Stop trading at 10 PM

# Notification settings
ENABLE_DETAILED_NOTIFICATIONS = True
NOTIFY_ON_PERFORMANCE_MILESTONES = True
PERFORMANCE_NOTIFICATION_INTERVAL = 10  # Every 10 trades

print("🧠 Smart Trading Configuration Loaded - Optimized for Profitability!")
