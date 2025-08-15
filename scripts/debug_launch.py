#!/usr/bin/env python3
"""
Debug Bot Launch - Start bot with extensive debugging
"""
import asyncio
import os
import sys
import traceback

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Disable Telegram but enable maximum logging
os.environ['DISABLE_TELEGRAM'] = '1'

if __name__ == "__main__":
    print("🐛 DEBUG BOT LAUNCHER - Maximum Logging Mode")
    print("=" * 60)

    try:
        import logging

        # Set up maximum logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('debug.log')
            ]
        )

        from main import DerivTradingBot

        print("🔧 Creating bot instance...")
        bot = DerivTradingBot()

        print("🔧 Setting up debug mode...")
        bot.debug_mode = True

        async def debug_start():
            print("🔧 Starting bot in debug mode...")
            try:
                # Force MT5 mode
                bot.execution_mode = "MT5"
                print(f"🔧 Execution mode: {bot.execution_mode}")
                print(f"🔧 Running: {bot.running}")
                print(f"🔧 Connected: {bot.connected}")
                print(f"🔧 Using MT5: {bot.using_mt5}")

                # Start the bot
                await bot.start()

            except Exception as e:
                print(f"❌ Debug start error: {e}")
                traceback.print_exc()

        # Run the debug bot
        asyncio.run(debug_start())

    except KeyboardInterrupt:
        print("\n⏹️ Debug bot stopped by user")
    except Exception as e:
        print(f"\n❌ Debug error: {e}")
        traceback.print_exc()
