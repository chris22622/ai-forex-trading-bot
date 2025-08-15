#!/usr/bin/env python3
"""
Test script to verify the bot uses real MT5 interface during startup
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import DerivTradingBot


async def test_bot_mt5_interface():
    """Test that the bot initializes with real MT5 interface"""
    print("🧪 TESTING BOT MT5 INTERFACE INITIALIZATION")
    print("=" * 50)

    try:
        # Create bot instance
        print("📝 Creating bot instance...")
        bot = DerivTradingBot()

        # Check what interface was assigned
        print(f"🔍 Bot MT5 Interface Type: {type(bot.mt5_interface).__name__}")
        print(f"🔍 Bot using_mt5 flag: {bot.using_mt5}")
        print(f"🔍 Bot execution_mode: {bot.execution_mode}")

        if hasattr(bot, 'mt5_interface') and bot.mt5_interface:
            interface_name = type(bot.mt5_interface).__name__

            if 'Dummy' in interface_name:
                print("❌ PROBLEM: Bot is using DummyMT5Interface!")
                print("❌ This means trades will be simulated, not real!")
                return False
            else:
                print("✅ SUCCESS: Bot is using real MT5TradingInterface!")

                # Test if we can initialize it
                print("🔗 Testing MT5 connection...")
                try:
                    connection_result = await bot.mt5_interface.initialize()
                    if connection_result:
                        balance = await bot.mt5_interface.get_account_balance()
                        print(f"✅ MT5 Connected! Balance: ${balance:.2f}")
                        print("🚀 BOT WILL TRADE ON REAL MT5 ACCOUNT!")
                        return True
                    else:
                        print("❌ MT5 connection failed")
                        return False

                except Exception as conn_e:
                    print(f"❌ MT5 connection error: {conn_e}")
                    return False
        else:
            print("❌ PROBLEM: Bot has no MT5 interface!")
            return False

    except Exception as e:
        print(f"❌ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 BOT MT5 INTERFACE TEST")
    result = asyncio.run(test_bot_mt5_interface())

    if result:
        print("\n🎉 SUCCESS! Bot will trade on REAL MT5!")
        print("✅ When you run the bot, trades will appear in your MetaTrader 5 terminal")
    else:
        print("\n❌ FAILED! Bot still using simulation")
        print("🔧 Check MT5 interface assignment in bot initialization")
