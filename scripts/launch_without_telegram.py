#!/usr/bin/env python3
"""
Simple Bot Launcher - Start bot without Telegram conflicts
"""
import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Temporarily disable Telegram
os.environ['DISABLE_TELEGRAM'] = '1'

if __name__ == "__main__":
    print("🚀 Starting Trading Bot WITHOUT Telegram (Conflict Resolution)")
    print("=" * 60)
    print("💡 This version runs without Telegram to avoid conflicts")
    print("💡 MT5 trading and AI features are fully functional")
    print("=" * 60)

    try:
        import asyncio

        from main import main

        # Run the main bot
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n⏹️ Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error running bot: {e}")
        import traceback
        traceback.print_exc()
