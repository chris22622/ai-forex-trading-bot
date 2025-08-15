#!/usr/bin/env python3
"""
🚀 AGGRESSIVE CONFIG FOR SMALL ACCOUNTS
Update config.py with settings optimized for small account growth
"""

import os

def update_config_for_small_accounts():
    """Update config.py with aggressive settings for small accounts"""
    
    config_content = '''#!/usr/bin/env python3
"""
🚀 AGGRESSIVE CONFIGURATION FOR SMALL ACCOUNT GROWTH
Optimized for accounts under $100 to enable actual growth
"""

import os
from datetime import datetime

# 🚀 AGGRESSIVE TRADING SETTINGS FOR SMALL ACCOUNTS
TRADE_AMOUNT = 3.0          # $3 trades (was $1)
AUTO_TAKE_PROFIT = 3.0      # $3 profit target (was $0.75) 
AUTO_STOP_LOSS = 4.0        # $4 stop loss (was $1.50)
MIN_PROFIT_THRESHOLD = 1.5  # $1.50 minimum profit (was $0.25)

# 🚀 AGGRESSIVE RISK MANAGEMENT
MAX_DAILY_LOSS = 25.0       # $25 daily loss limit (was $15)
MAX_PORTFOLIO_LOSS = 20.0   # $20 portfolio loss (was $8)
MAX_CONSECUTIVE_LOSSES = 100 # No limit on consecutive losses
MAX_CONCURRENT_TRADES = 5    # 5 concurrent trades (was 3)

# 🚀 AGGRESSIVE TIMING
MAX_TRADE_AGE_MINUTES = 45   # 45 minutes per trade (was 10)
MIN_TRADE_INTERVAL = 10      # 10 seconds between trades (was 30)

# 🚀 AGGRESSIVE AI SETTINGS
AI_CONFIDENCE_THRESHOLD = 0.4  # Lower confidence needed (was 0.6)
RSI_BUY_THRESHOLD = 45         # Buy when RSI < 45 (was < 40)
RSI_SELL_THRESHOLD = 55        # Sell when RSI > 55 (was > 60)

# 🚀 AGGRESSIVE PROFIT PROTECTION
PROFIT_PROTECTION_ENABLED = True
PROFIT_PROTECTION_THRESHOLD = 15.0    # Stop at +$15 profit (was $5)
PROFIT_PROTECTION_MAX_THRESHOLD = 25.0 # Emergency stop at +$25

# MT5 Connection Settings
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '62233085'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', 'ChrisAI2024!')
MT5_SERVER = os.getenv('MT5_SERVER', 'DerivVU-Server')
MT5_PATH = r"C:\\Program Files\\Deriv MT5\\terminal64.exe"

# 🚀 AGGRESSIVE POSITION SIZING
BASE_RISK_PERCENTAGE = 0.15  # 15% risk for tiny accounts
MAX_LOT_SIZE = 0.10          # Maximum 0.10 lots
MIN_LOT_SIZE = 0.02          # Minimum 0.02 lots (was 0.01)

# Telegram Settings
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your-telegram-bot-token-here')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your-chat-id-here')

# Trading Symbols (Deriv Synthetic Indices)
SYMBOLS = [
    'Volatility 10 Index',
    'Volatility 25 Index', 
    'Volatility 75 Index',
    'Volatility 100 (1s) Index',
    'Boom 500 Index',
    'Crash 500 Index'
]

# Technical Indicators
EMA_FAST = 8    # Fast EMA period
EMA_SLOW = 21   # Slow EMA period
RSI_PERIOD = 14 # RSI period
MA_SHORT = 10   # Short moving average
MA_LONG = 20    # Long moving average

# 🚀 AGGRESSIVE FEATURES
ENABLE_AUTO_TRADE_MANAGEMENT = True
ENABLE_SMART_RISK_MANAGEMENT = True
ENABLE_MULTI_SYMBOL_TRADING = True
ENABLE_HEDGE_PREVENTION = False    # Disable for maximum opportunities

print(f"🚀 AGGRESSIVE CONFIG LOADED - Optimized for Small Account Growth!")
print(f"📊 Trade Amount: ${TRADE_AMOUNT}")
print(f"🎯 Profit Target: ${AUTO_TAKE_PROFIT}")
print(f"🛡️ Stop Loss: ${AUTO_STOP_LOSS}")
print(f"⏱️ Trade Time Limit: {MAX_TRADE_AGE_MINUTES} minutes")
print(f"🔥 Risk Percentage: {BASE_RISK_PERCENTAGE*100}%")
'''

    # Write the new config
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("🚀 AGGRESSIVE CONFIG APPLIED!")
    print("=" * 50)
    print("📊 NEW SETTINGS FOR YOUR $20 ACCOUNT:")
    print("  💰 Trade Amount: $3.00 (was $1.00)")
    print("  🎯 Profit Target: $3.00 (was $0.75)")
    print("  🛡️ Stop Loss: $4.00 (was $1.50)")
    print("  ⏱️ Time Limit: 45 minutes (was 10)")
    print("  📈 Risk per Trade: 15% = $3.00")
    print("  🔄 Max Concurrent: 5 trades (was 3)")
    print("  🚫 Profit Protection: $15 (was $5)")
    
    print("\n🚀 THESE SETTINGS WILL TRANSFORM YOUR BOT!")
    print("✅ Much bigger position sizes")
    print("✅ Higher profit targets worth pursuing")
    print("✅ More trading opportunities")
    print("✅ Realistic growth potential")
    
    return True

if __name__ == "__main__":
    update_config_for_small_accounts()
