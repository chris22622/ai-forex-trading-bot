"""
Configuration file for Deriv Trading Bot - LIVE TRADING ENABLED
ULTRA-CONSERVATIVE SETTINGS FOR $100 ACCOUNT PROTECTION
"""

import os

# ==================== LIVE TRADING MODE ====================
# 🚨 LIVE TRADING ENABLED - ULTRA CONSERVATIVE SETTINGS
DEMO_MODE = False  # LIVE TRADING

# ==================== API CREDENTIALS ====================
# Deriv API Tokens (Get from app.deriv.com -> API Token)
DERIV_LIVE_API_TOKEN = "g9kxuLTUWxrHGNe"  # Live token with ALL permissions
DERIV_DEMO_API_TOKEN = "s8Sww40XKyKMp0x"  # Demo token with ALL permissions

# Select the appropriate token based on mode
DERIV_API_TOKEN = DERIV_DEMO_API_TOKEN if DEMO_MODE else DERIV_LIVE_API_TOKEN

# ==================== LIVE MT5 ACCOUNT CREDENTIALS ====================
LIVE_ACCOUNT_NUMBER = 62233085
LIVE_PASSWORD = "Goon22622$"
LIVE_SERVER = "DerivVU-Server"

# ==================== ULTRA-CONSERVATIVE LIVE SETTINGS ====================
# Protect $100 account with minimal risk settings
LIVE_TRADE_AMOUNT = 0.01          # Minimal lot size (1 cent per pip)
LIVE_AUTO_TAKE_PROFIT = 1.0       # Take profit at $1 (1% of account)
LIVE_AUTO_STOP_LOSS = 1.5         # Stop loss at $1.50 (1.5% of account)
LIVE_MAX_CONCURRENT = 1           # Only 1 trade at a time
LIVE_MAX_DAILY_LOSS = 10.0        # Stop if lose $10 in a day (10% of account)
LIVE_MAX_CONSECUTIVE_LOSSES = 8   # Very conservative loss limit
LIVE_EMERGENCY_STOP = 20.0        # Emergency stop if lose $20 (20% of account)

# Live trading safety features
LIVE_REQUIRE_CONFIRMATION = False  # Auto-trade (but with tight limits)
LIVE_BALANCE_CHECK_INTERVAL = 30   # Check balance every 30 seconds
LIVE_MIN_BALANCE = 80.0           # Stop trading if balance drops below $80

# ==================== TELEGRAM CONFIGURATION ====================
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
ENABLE_TELEGRAM_ALERTS = True  # ✅ ENABLED for live trading notifications

# Check if Telegram should be disabled
TELEGRAM_ENABLED = os.getenv('DISABLE_TELEGRAM', '0').lower() not in ['1', 'true', 'yes']

if not TELEGRAM_ENABLED:
    TELEGRAM_BOT_TOKEN = ""
    ENABLE_TELEGRAM_ALERTS = False

# ==================== TRADING SETTINGS ====================
# WebSocket URLs for Deriv
DERIV_WS_URLS = [
    "wss://ws.derivws.com/websockets/v3",
    "wss://ws.binaryws.com/websockets/v3",
    "wss://ws.deriv.com/websockets/v3"
]
DERIV_WS_URL = DERIV_WS_URLS[0]

# Trading Parameters - ULTRA-CONSERVATIVE FOR LIVE $100 ACCOUNT
TRADE_AMOUNT = 0.01  # 🛡️ MINIMAL: 0.01 lots for live trading safety
CURRENCY = "USD"
DURATION = 1
DURATION_UNIT = "t"

# Symbol Configuration - CONSERVATIVE FOR LIVE TRADING
DEFAULT_SYMBOL = "Volatility 10 Index"  # 🛡️ SAFEST: Most stable for live trading
AVAILABLE_SYMBOLS = [
    "Volatility 10 Index",    # PRIMARY: Most conservative, lowest volatility
    "Volatility 25 Index",    # SECONDARY: Moderate volatility  
    "Volatility 50 Index",    # TERTIARY: Medium volatility
    "Volatility 75 Index",    # QUATERNARY: Higher volatility (backup only)
    "Volatility 100 Index"    # QUINARY: Highest volatility (emergency only)
]

# ==================== MULTI-SYMBOL TRADING SETTINGS ====================
ENABLE_MULTI_SYMBOL_TRADING = True
PRIMARY_TRADING_SYMBOLS = [
    "Volatility 10 Index",    # Primary: Ultra-conservative
    "Volatility 25 Index"     # Secondary: Moderate volatility
]

MAX_SYMBOLS_CONCURRENT = 1  # Only 1 symbol for live trading safety
SYMBOL_ROTATION_ENABLED = False  # Disable rotation for stability

# ==================== TECHNICAL ANALYSIS SETTINGS ====================
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

EMA_FAST = 12
EMA_SLOW = 26
EMA_SIGNAL = 9

BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

MA_SHORT = 10
MA_LONG = 20

# ==================== ULTRA-CONSERVATIVE RISK MANAGEMENT ====================
MAX_CONSECUTIVE_LOSSES = 8  # Very conservative for live trading
MAX_DAILY_LOSS = 10.0       # Maximum $10 daily loss (10% of account)
MIN_BALANCE = 80.0          # Stop trading if balance drops below $80
STOP_LOSS_PERCENT = 0.015   # 1.5% stop loss

# ==================== AUTOMATIC TRADE MANAGEMENT ====================
ENABLE_AUTO_TRADE_MANAGEMENT = True
AUTO_TAKE_PROFIT = 1.0      # Take profit at $1
AUTO_STOP_LOSS = 1.5        # Stop loss at $1.50
AUTO_TIME_LIMIT = 10        # Close trades after 10 minutes
MIN_PROFIT_THRESHOLD = 0.25 # Minimum $0.25 profit to consider profitable
MAX_CONCURRENT_TRADES = 1   # Only 1 trade at a time for live trading

# ==================== HEDGE PREVENTION SETTINGS ====================
ENABLE_HEDGE_PREVENTION = True
HEDGE_CONFLICT_ACTION = "block"
ALLOW_SAME_DIRECTION_STACKING = False  # No stacking for live trading

# ==================== BOT BEHAVIOR SETTINGS ====================
MIN_TRADE_INTERVAL = 30  # 30 seconds between trades for safety
AI_CONFIDENCE_THRESHOLD = 0.65  # High confidence threshold (65%) for live trading
PRICE_BUFFER_SIZE = 100

# Feature flags
ENABLE_AI_PREDICTIONS = True
ENABLE_RISK_MANAGEMENT = True
ENABLE_LOGGING = True

# EXECUTION MODE
EXECUTION_MODE = "MT5"  # Use MT5 for live trading

# ==================== MT5 CONFIGURATION ====================
# Live MT5 Settings
MT5_DEMO_MODE = False  # Live trading mode
MT5_REAL_TRADING = True

# Use live account credentials
MT5_LOGIN = LIVE_ACCOUNT_NUMBER
MT5_PASSWORD = LIVE_PASSWORD
MT5_SERVER = LIVE_SERVER

# Ultra-conservative MT5 settings
MT5_LOT_SIZE = 0.01         # Minimum lot size
MT5_MAX_LOT_SIZE = 0.01     # Maximum lot size (same as minimum for safety)
MT5_SLIPPAGE = 3
MT5_MAGIC_NUMBER = 234000

# Conservative symbols for live trading
MT5_SYMBOLS = {
    "Volatility 10 Index": "Volatility 10 Index",
    "Volatility 25 Index": "Volatility 25 Index"
}
MT5_DEFAULT_SYMBOL = "Volatility 10 Index"

# ==================== LOGGING SETTINGS ====================
LOG_DIR = "logs"
TRADE_LOG_FILE = f"{LOG_DIR}/trades.csv"
BOT_LOG_FILE = f"{LOG_DIR}/trading_bot.log"
ERROR_LOG_FILE = f"{LOG_DIR}/errors.log"
LOG_LEVEL = "INFO"

# ==================== AI/ML SETTINGS ====================
AI_MODEL_TYPE = "ensemble"
TRAIN_DATA_DAYS = 7  # Shorter training for live trading
RETRAIN_INTERVAL_HOURS = 12  # More frequent retraining

USE_TECHNICAL_INDICATORS = True
USE_PRICE_PATTERNS = True
USE_VOLUME_ANALYSIS = False

# ==================== ADVANCED SETTINGS ====================
MAX_RECONNECTION_ATTEMPTS = 5
RECONNECTION_DELAY = 5

PERFORMANCE_REVIEW_INTERVAL = 50  # Review after 50 trades
STRATEGY_ADJUSTMENT_THRESHOLD = 0.6  # Higher threshold for stability

PAPER_TRADING = False

# ==================== DEVELOPMENT SETTINGS ====================
DEBUG_MODE = True
VERBOSE_LOGGING = True
TEST_MODE = False
SIMULATION_SPEED = 1.0

# ==================== VALIDATION ====================
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if not DERIV_API_TOKEN or len(DERIV_API_TOKEN) < 10:
        errors.append("❌ DERIV_API_TOKEN not properly set")
    
    if TELEGRAM_ENABLED and (not TELEGRAM_BOT_TOKEN or len(TELEGRAM_BOT_TOKEN) < 10):
        errors.append("❌ TELEGRAM_BOT_TOKEN not properly set")
    
    if TELEGRAM_ENABLED and (not TELEGRAM_CHAT_ID or len(str(TELEGRAM_CHAT_ID)) < 5):
        errors.append("❌ TELEGRAM_CHAT_ID not properly set")
    
    if TRADE_AMOUNT <= 0:
        errors.append("❌ TRADE_AMOUNT must be positive")
    
    if DEFAULT_SYMBOL not in AVAILABLE_SYMBOLS:
        errors.append(f"❌ DEFAULT_SYMBOL '{DEFAULT_SYMBOL}' not in AVAILABLE_SYMBOLS list")
    
    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    return errors

if __name__ == "__main__":
    errors = validate_config()
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(error)
    else:
        print("✅ Configuration validation passed!")
        print(f"🚨 LIVE TRADING MODE: {'ENABLED' if not DEMO_MODE else 'DISABLED'}")
        print(f"🎯 Trading Symbol: {DEFAULT_SYMBOL}")
        print(f"💰 Trade Amount: ${TRADE_AMOUNT}")
        print(f"📱 Telegram Alerts: {'Enabled' if ENABLE_TELEGRAM_ALERTS else 'Disabled'}")
        print(f"🛡️ Max Daily Loss: ${MAX_DAILY_LOSS}")
        print(f"🔒 Max Concurrent Trades: {MAX_CONCURRENT_TRADES}")
        print(f"⚡ AI Confidence Threshold: {AI_CONFIDENCE_THRESHOLD}")
