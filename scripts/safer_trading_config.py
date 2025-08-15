#!/usr/bin/env python3
"""
🛡️ SAFER TRADING CONFIGURATION
Ultra-conservative settings for small account protection after drawdown
"""

import shutil
from datetime import datetime

from config import *


def create_safer_config():
    """Create a backup and update config with safer settings"""

    # Create backup first
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"config_backup_before_safer_{timestamp}.py"
    shutil.copy("config.py", backup_file)
    print(f"✅ Config backed up to: {backup_file}")

    # Read current config
    with open("config.py", "r") as f:
        content = f.read()

    # Ultra-safe modifications for $42 account
    safer_settings = {
        # 🛡️ ULTRA-CONSERVATIVE POSITION SIZING
        "TRADE_AMOUNT": "0.5",  # Reduced from $3 to $0.50 (0.01 lots)

        # 🚨 TIGHTER RISK CONTROLS
        "AUTO_STOP_LOSS": "1.0",  # Reduced from $4 to $1 max loss
        "AUTO_TAKE_PROFIT": "0.75",  # Reduced from $3 to $0.75 profit target
        "MAX_LOSS_PER_TRADE": "1.0",  # $1 max loss per trade
        "MAX_PORTFOLIO_LOSS": "3.0",  # $3 max total loss (7% of $42)

        # ⏰ FASTER TRADE MANAGEMENT
        "AUTO_TIME_LIMIT": "15",  # Reduced from 45 to 15 minutes
        "MAX_TRADE_AGE_MINUTES": "15",  # Close trades faster

        # 🔢 STRICTER LIMITS
        "MAX_CONCURRENT_TRADES": "1",  # Only 1 trade at a time
        "MIN_TRADE_INTERVAL": "120",  # 2 minutes between trades

        # 🎯 SAFER SYMBOLS (avoid Crash indices)
        "DEFAULT_SYMBOL": '"Volatility 10 Index"',  # Safer than Crash 500
        "PRIMARY_TRADING_SYMBOLS": '["Volatility 10 Index", "Volatility 25 Index"]',

        # 🧠 HIGHER AI CONFIDENCE
        "AI_CONFIDENCE_THRESHOLD": "0.65",  # Increased from 0.5 to 0.65

        # 🛡️ MAXIMUM DRAWDOWN PROTECTION
        "MAX_DRAWDOWN_PERCENTAGE": "0.15",  # 15% instead of 20%

        # 💰 PROFIT PROTECTION
        "PROFIT_PROTECTION_THRESHOLD": "2.0",  # Take profits at just $2
        "PROFIT_PROTECTION_MAX_THRESHOLD": "5.0",  # Emergency stop at $5
    }

    # Apply each setting
    for setting, value in safer_settings.items():
        # Find and replace the setting
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{setting} ="):
                old_line = line
                new_line = f"{setting} = {value}"
                lines[i] = new_line
                print(f"✅ Updated: {setting} = {value}")
                break
        content = '\n'.join(lines)

    # Write safer config
    with open("config.py", "w") as f:
        f.write(content)

    print("\n🛡️ SAFER CONFIGURATION APPLIED!")
    print("📊 Key Changes:")
    print("   • Trade size: $3.00 → $0.50 (0.01 lots)")
    print("   • Stop loss: $4.00 → $1.00")
    print("   • Take profit: $3.00 → $0.75")
    print("   • Time limit: 45min → 15min")
    print("   • Symbol: Crash 500 → Volatility 10")
    print("   • AI threshold: 50% → 65%")
    print("   • Max drawdown: 20% → 15%")
    print("\n💡 These settings are optimized for your $42.47 balance")
    print("🚀 Ready to restart with safer parameters!")

if __name__ == "__main__":
    create_safer_config()
