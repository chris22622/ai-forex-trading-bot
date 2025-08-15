#!/usr/bin/env python3
"""
🚀 AGGRESSIVE TRADING ACTIVATOR 
This script immediately removes trade limits and enables ultra-aggressive trading
Run this to activate maximum profit mode!
"""

import asyncio
import os
import sys

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def activate_aggressive_trading():
    """🚀 Activate ultra-aggressive trading mode for maximum profits"""
    try:
        print("🚀 AGGRESSIVE TRADING ACTIVATOR")
        print("=" * 50)

        # Import required modules
        from main import global_bot_instance

        if not global_bot_instance:
            print("❌ No bot instance found! Make sure the main bot is running.")
            return False

        bot = global_bot_instance

        print("🔍 CURRENT BOT STATUS:")
        print(f"   MT5 Connected: {bot.mt5_connected}")
        print(f"   Balance: ${bot.current_balance:.2f}")
        print(f"   Active Trades: {len(bot.active_trades)}")
        print(f"   Active Symbols: {len(bot.active_symbols)}")
        print(f"   Consecutive Losses: {bot.consecutive_losses}")

        # Check symbol readiness
        ready_symbols = 0
        print("\n📊 SYMBOL STATUS:")
        for symbol in bot.active_symbols:
            history = bot.symbol_price_histories.get(symbol, [])
            status = "✅ READY" if len(history) >= 5 else f"⏳ {len(history)}/5"
            print(f"   {symbol}: {status}")
            if len(history) >= 5:
                ready_symbols += 1

        print(f"\n🎯 Ready to trade: {ready_symbols}/{len(bot.active_symbols)} symbols")

        # Run emergency cleanup
        print("\n🚨 RUNNING EMERGENCY CLEANUP...")
        cleanup_result = await bot.emergency_clear_trade_limits()

        if cleanup_result:
            print("\n✅ CLEANUP SUCCESSFUL:")
            print(f"   🗑️ Stale trades removed: {cleanup_result['stale_removed']}")
            print(f"   ✅ Active trades remaining: {cleanup_result['active_remaining']}")
            print(f"   🚀 Available slots: {cleanup_result['available_slots']}/10")
        else:
            print("\n⚠️ Cleanup had issues but continuing...")

        # Verify trading capabilities
        print("\n🛡️ VERIFYING TRADING CAPABILITIES:")

        # Check all the trading checks
        risk_ok = bot._check_risk_limits()
        concurrent_ok = bot._check_concurrent_trades_limit()
        timing_ok = bot._check_trade_timing()
        hedge_ok = bot._check_hedge_prevention("BUY")  # Test with BUY

        print(f"   Risk Limits: {'✅ PASS' if risk_ok else '❌ FAIL'}")
        print(f"   Concurrent Trades: {'✅ PASS' if concurrent_ok else '❌ FAIL'}")
        print(f"   Trade Timing: {'✅ PASS' if timing_ok else '❌ FAIL'}")
        print(f"   Hedge Prevention: {'✅ DISABLED (GOOD)' if hedge_ok else '❌ BLOCKED'}")

        all_checks_pass = risk_ok and concurrent_ok and timing_ok and hedge_ok

        if all_checks_pass:
            print("\n🚀 ALL SYSTEMS GO!")
            print("🎯 Bot is now configured for ULTRA-AGGRESSIVE trading!")
            print("💰 Capable of:")
            print("   • Up to 10 simultaneous trades")
            print(f"   • Trading on {len(bot.active_symbols)} symbols")
            print("   • Both BUY and SELL positions allowed")
            print("   • Same-direction stacking enabled")
            print("   • Maximum profit potential ACTIVATED!")

            # Show expected trading activity
            if ready_symbols > 0:
                print("\n⚡ EXPECT TRADING ACTIVITY WITHIN MINUTES!")
                print(f"📊 {ready_symbols} symbols ready for immediate analysis")
            else:
                print("\n⏳ Collecting price data... trades will start soon!")

            return True
        else:
            print("\n⚠️ Some checks failed - investigating...")

            if not risk_ok:
                print(f"🔧 Risk limits issue - checking consecutive losses: {bot.consecutive_losses}")
            if not concurrent_ok:
                print(f"🔧 Concurrent trades issue - current: {len(bot.active_trades)}/10")
            if not timing_ok:
                print(f"🔧 Timing issue - last trade time: {bot.last_trade_time}")

            return False

    except Exception as e:
        print(f"💥 Activation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Aggressive Trading Activation...")
    result = asyncio.run(activate_aggressive_trading())

    if result:
        print("\n🎉 SUCCESS! Ultra-aggressive trading mode ACTIVATED!")
        print("💰 Your bot is now optimized for maximum profit generation!")
    else:
        print("\n⚠️ Activation had issues. Check the bot logs for details.")

    input("\nPress Enter to exit...")
