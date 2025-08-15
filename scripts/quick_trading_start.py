#!/usr/bin/env python3
"""
🚀 QUICK TRADING START - NO TELEGRAM CONFLICTS
Ultra-aggressive trading bot with immediate execution
"""

import asyncio
import logging

# Disable Telegram for immediate trading
import os

from main import DerivTradingBot

os.environ['DISABLE_TELEGRAM'] = 'true'

async def main():
    """Start trading immediately without Telegram conflicts"""
    print("🚀 STARTING ULTRA-AGGRESSIVE TRADING BOT")
    print("💰 IMMEDIATE TRADING MODE - NO TELEGRAM CONFLICTS")
    print("⚡ TRADE PARAMETERS:")
    print("   - Confidence threshold: 5%")
    print("   - Minimum price points: 1")
    print("   - Force trading after: 3 price points")
    print("   - Ultra-fast fallback: 5 price points")
    print("🔥 LET'S MAKE MONEY!")

    # Create and start bot
    bot = DerivTradingBot()

    try:
        # Skip Telegram initialization - focus on trading!
        print("⚡ Skipping Telegram - PURE TRADING MODE")

        # Start trading immediately
        await bot.start()

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.exception("Bot error")

if __name__ == "__main__":
    asyncio.run(main())
