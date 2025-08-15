#!/usr/bin/env python3
"""
Emergency Trade Script - Forces immediate trade placement
Use this when the bot is being too conservative and you need immediate action.
"""

import asyncio
import os
import sys
import time
from datetime import datetime

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def force_emergency_trade():
    """Force place an emergency trade RIGHT NOW"""
    try:
        print("🚨 EMERGENCY TRADE SCRIPT STARTING...")

        # Import required modules
        from main import global_bot_instance

        if not global_bot_instance:
            print("❌ No bot instance found! Make sure the main bot is running.")
            return False

        bot = global_bot_instance
        symbol = "Volatility 50 Index"  # Use most active symbol

        print("🔍 Bot Status Check:")
        print(f"   MT5 Connected: {bot.mt5_connected}")
        print(f"   Balance: ${bot.current_balance:.2f}")
        print(f"   Active Trades: {len(bot.active_trades)}")
        print(f"   Consecutive Losses: {bot.consecutive_losses}")

        if not bot.mt5_connected:
            print("❌ MT5 not connected! Cannot place trade.")
            return False

        if bot.current_balance < 10:
            print("❌ Insufficient balance for safe trading.")
            return False

        print(f"🚀 FORCING IMMEDIATE BUY TRADE ON {symbol}...")

        # Get current price first
        current_price = await bot._get_current_price_for_symbol(symbol)
        if not current_price:
            print(f"❌ Could not get current price for {symbol}")
            return False

        print(f"💹 Current price: {current_price}")

        # Force place a BUY trade directly through MT5
        if bot.mt5_interface:
            print("🎯 Placing emergency BUY trade...")

            result = await bot.mt5_interface.place_trade(
                action="BUY",
                symbol=symbol,
                amount=0.01  # Minimum safe lot size
            )

            if result:
                trade_id = result.get('ticket', f"emergency_{int(time.time())}")
                entry_price = result.get('price', current_price)

                # Store in bot's active trades
                bot.active_trades[str(trade_id)] = {
                    'action': "BUY",
                    'symbol': symbol,
                    'amount': 0.01,
                    'confidence': 0.9,
                    'timestamp': datetime.now().isoformat(),
                    'entry_price': entry_price,
                    'mt5_ticket': trade_id
                }

                print("✅ EMERGENCY TRADE PLACED!")
                print(f"   Ticket: {trade_id}")
                print(f"   Symbol: {symbol}")
                print("   Action: BUY")
                print("   Amount: 0.01 lots")
                print(f"   Entry Price: {entry_price}")

                # Send Telegram notification
                try:
                    await bot.notifier.telegram.send_message(f"""🚨 EMERGENCY TRADE PLACED
                    
📊 Action: BUY
🎯 Symbol: {symbol}
💰 Lot Size: 0.01
🎫 Ticket: {trade_id}
💹 Entry: {entry_price}
🕒 Time: {datetime.now().strftime('%H:%M:%S')}

🔥 This was a FORCED emergency trade to guarantee trading activity!
Bot will now manage this trade automatically.""")
                    print("📱 Telegram notification sent!")
                except Exception as e:
                    print(f"⚠️ Notification failed: {e}")

                return True
            else:
                print("❌ Emergency trade placement failed")
                return False
        else:
            print("❌ MT5 interface not available")
            return False

    except Exception as e:
        print(f"💥 Emergency trade error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def check_bot_status():
    """Check the current bot status and trading conditions"""
    try:
        from main import global_bot_instance

        if not global_bot_instance:
            print("❌ No bot instance found!")
            return

        bot = global_bot_instance

        print("🔍 DETAILED BOT STATUS:")
        print(f"   Running: {bot.running}")
        print(f"   MT5 Connected: {bot.mt5_connected}")
        print(f"   Balance: ${bot.current_balance:.2f}")
        print(f"   Active Trades: {len(bot.active_trades)}")
        print(f"   Consecutive Losses: {bot.consecutive_losses}")
        print(f"   Daily Profit: ${bot.daily_profit:.2f}")
        print(f"   Trades Today: {bot.trades_today}")

        # Check symbol price histories
        print("\n📊 SYMBOL PRICE DATA:")
        for symbol, history in bot.symbol_price_histories.items():
            print(f"   {symbol}: {len(history)} price points")
            if len(history) >= 5:
                print("      ✅ Ready for trading!")
            else:
                print(f"      ⏳ Collecting data ({len(history)}/5)...")

        # Check if we can get current prices
        print("\n💹 CURRENT PRICES:")
        for symbol in bot.active_symbols:
            try:
                price = await bot._get_current_price_for_symbol(symbol)
                print(f"   {symbol}: {price}")
            except Exception as e:
                print(f"   {symbol}: Error - {e}")

        # Check risk limits
        print("\n🛡️ RISK CHECKS:")
        risk_ok = bot._check_risk_limits()
        concurrent_ok = bot._check_concurrent_trades_limit()
        timing_ok = bot._check_trade_timing()

        print(f"   Risk Limits: {'✅ Pass' if risk_ok else '❌ Fail'}")
        print(f"   Concurrent Trades: {'✅ Pass' if concurrent_ok else '❌ Fail'}")
        print(f"   Trade Timing: {'✅ Pass' if timing_ok else '❌ Fail'}")

        if risk_ok and concurrent_ok and timing_ok:
            print("\n🚀 ALL SYSTEMS GO! Bot should be trading soon...")
        else:
            print("\n⚠️ Some checks failing - this might be why no trades are happening")

    except Exception as e:
        print(f"Status check error: {e}")

if __name__ == "__main__":
    print("🚨 EMERGENCY TRADING SCRIPT")
    print("=" * 50)

    choice = input("""
Choose action:
1. Force emergency trade NOW
2. Check bot status only
3. Exit

Enter choice (1-3): """).strip()

    if choice == "1":
        print("\n🚨 FORCING EMERGENCY TRADE...")
        result = asyncio.run(force_emergency_trade())
        if result:
            print("\n✅ Emergency trade completed successfully!")
        else:
            print("\n❌ Emergency trade failed!")
    elif choice == "2":
        print("\n🔍 CHECKING BOT STATUS...")
        asyncio.run(check_bot_status())
    else:
        print("👋 Exiting...")

    input("\nPress Enter to exit...")
