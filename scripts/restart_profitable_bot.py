#!/usr/bin/env python3
"""
🚀 PROFITABLE BOT RESTART SCRIPT
Restarts the bot with optimized profitable settings
"""

import asyncio
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def display_profitable_settings():
    """Display the new profitable settings"""
    print("🚀 NEW PROFITABLE SETTINGS ACTIVATED!")
    print("=" * 60)
    print("💰 PROFIT TAKING:")
    print("   ✅ Take Profit: $0.75 (was $1.20)")
    print("   ✅ Min Profit: $0.25 (was $0.40)")
    print("   ✅ Profit Protection: $5 (was $10)")
    print("")
    print("🛡️ RISK MANAGEMENT:")
    print("   ✅ Stop Loss: $1.50 (was $1.80)")
    print("   ✅ Max Portfolio Loss: $8 (was $15)")
    print("   ✅ Trade Time Limit: 10min (was 12min)")
    print("   ✅ Max Consecutive Losses: 2 (was 100)")
    print("")
    print("📊 POSITION SIZING:")
    print("   ✅ Trade Amount: $1.00 (was $3.00)")
    print("   ✅ Max Concurrent Trades: 3 (was 6)")
    print("   ✅ Risk per Trade: 0.5% (was 1.5%)")
    print("")
    print("🎯 STRATEGY:")
    print("   ✅ RSI Buy: < 40 (was < 45)")
    print("   ✅ RSI Sell: > 60 (was > 55)")
    print("   ✅ Higher Confidence Thresholds")
    print("   ✅ Hedge Prevention: ENABLED")
    print("")

async def restart_profitable_bot():
    """Restart the bot with profitable settings"""
    try:
        print("🔄 RESTARTING BOT WITH PROFITABLE SETTINGS...")
        print("")

        # Import and restart bot
        from main import main as start_bot

        print("🚀 Starting bot with new profitable configuration...")
        print("📊 Monitor closely for the first few trades")
        print("💡 Bot will now:")
        print("   - Take profits much faster ($0.75)")
        print("   - Use tighter stop losses ($1.50)")
        print("   - Exit trades faster (10 minutes)")
        print("   - Use smaller position sizes")
        print("   - Stop after 2 consecutive losses")
        print("")
        print("🛡️ PROFIT PROTECTION ACTIVE AT $5!")
        print("")

        # Start the bot
        await start_bot()

    except KeyboardInterrupt:
        print("⏸️ Bot stopped by user")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        print("💡 Try running the bot manually with: python main.py")

def main():
    """Main entry point"""
    print("🚀 PROFITABLE BOT RESTART")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    display_profitable_settings()

    # Ask for confirmation
    response = input("🤔 Ready to restart bot with profitable settings? (y/n): ").lower().strip()

    if response in ['y', 'yes']:
        print("")
        print("✅ Starting profitable bot...")
        asyncio.run(restart_profitable_bot())
    else:
        print("❌ Restart cancelled")
        print("💡 Run this script again when ready")

if __name__ == "__main__":
    main()
