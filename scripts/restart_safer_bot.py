#!/usr/bin/env python3
"""
🚀 SAFER BOT RESTART SCRIPT
Restart your trading bot with the new safer configuration
"""

import subprocess
import sys
import time
from datetime import datetime


def restart_safer_bot():
    """Restart the bot with safer settings"""
    print("🛡️ SAFER BOT RESTART INITIATED")
    print("=" * 40)

    # Show current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕒 Restart Time: {current_time}")

    # Show safer settings summary
    print("\n🛡️ SAFER SETTINGS ACTIVE:")
    print("   • Trade Size: $0.50 (0.01 lots)")
    print("   • Stop Loss: $1.00 max")
    print("   • Take Profit: $0.75")
    print("   • Symbol: Volatility 10 Index")
    print("   • AI Confidence: 65%")
    print("   • Trade Duration: 15 minutes")
    print("   • Max Drawdown: 15%")

    print("\n💰 Account Protection:")
    print("   • Starting Balance: ~$42.47")
    print("   • Daily Loss Limit: $8.00")
    print("   • Portfolio Loss Limit: $3.00")
    print("   • Only 1 trade at a time")

    print("\n🎯 Recovery Goals:")
    print("   • Target: +$0.75 per trade")
    print("   • Daily Goal: +$2.25 (3 trades)")
    print("   • Weekly Goal: +$11.25")

    # Countdown
    print("\n🚀 Starting safer bot in:")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)

    print("\n✅ LAUNCHING SAFER BOT!")
    print("🛡️ Maximum protection enabled")
    print("📱 Telegram notifications active")
    print("🎯 Focus on Volatility 10 Index")

    # Run the main bot
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Bot error: {e}")
        print("💡 Check your MT5 connection and try again")

if __name__ == "__main__":
    restart_safer_bot()
