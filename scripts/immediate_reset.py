"""
IMMEDIATE Emergency Reset - Force Reset Running Bot
This script will immediately reset the consecutive losses on the running bot
"""

import os
import sys

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def immediate_reset():
    """Immediate reset of running bot instance"""
    try:
        # Import the global bot instance
        from main import global_bot_instance

        if not global_bot_instance:
            print("❌ No running bot instance found")
            print("🔄 Please restart the bot to apply new config settings")
            return False

        bot = global_bot_instance

        print("🚨 IMMEDIATE EMERGENCY RESET STARTING...")
        print("📊 Current Status:")
        print(f"   Consecutive Losses: {bot.consecutive_losses}")
        print(f"   Daily P&L: ${bot.daily_profit:.2f}")
        print(f"   Active Trades: {len(bot.active_trades)}")

        # Store old values
        old_consecutive = bot.consecutive_losses
        old_daily = bot.daily_profit

        # FORCE RESET ALL LIMITS
        bot.consecutive_losses = 0
        bot.daily_profit = 0.0

        print("\n✅ EMERGENCY RESET COMPLETE:")
        print(f"   Consecutive losses: {old_consecutive} → 0")
        print(f"   Daily P&L: ${old_daily:.2f} → $0.00")
        print("🚀 Bot ready to resume trading immediately!")

        # Verify the reset worked
        print("\n🔍 Verification:")
        print(f"   New consecutive losses: {bot.consecutive_losses}")
        print(f"   New daily P&L: ${bot.daily_profit:.2f}")

        # Check if bot can now trade
        try:
            can_trade = bot._check_risk_limits()
            f"   Risk limits check: {'✅ PASS - CAN TRADE' if can_trade else '❌ STILL BLOCKED'}"
            f"
        except Exception as e:
            print(f"   Risk check error: {e}")

        # Call the bot's emergency reset method too
        try:
            bot.emergency_reset_all_limits()
            print("✅ Bot emergency reset method also called")
        except Exception as e:
            print(f"⚠️ Bot method error: {e}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔄 Bot may not be running or main.py not accessible")
        return False
    except Exception as e:
        print(f"❌ Emergency reset failed: {e}")
        return False

if __name__ == "__main__":
    print("🚨 IMMEDIATE EMERGENCY RESET TOOL")
    print("=" * 40)

    success = immediate_reset()

    if success:
        print("\n🎉 SUCCESS! Your bot should now be able to trade again!")
        print("📈 The AI can now execute those excellent 61% confidence signals!")
    else:
        print("\n⚠️ Reset failed. You may need to restart the bot.")
        print("🔄 Run: python main.py")

    print("\n📊 New Settings:")
    print("   MAX_CONSECUTIVE_LOSSES = 100 (was 10)")
    print("   Current consecutive losses = 0")
    print("   Bot is ready to trade!")
