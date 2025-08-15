#!/usr/bin/env python3
"""
🚨 Advanced Risk Management Tools
Emergency tools to close losing trades and manage risk
"""

import asyncio
import os
import sys

# Add the directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def debug_current_trades():
    """Debug function to check all current trades"""
    try:
        from main import global_bot_instance

        if not global_bot_instance:
            print("❌ Bot not running")
            return

        bot = global_bot_instance
        if not bot.mt5_interface:
            print("❌ MT5 not connected")
            return

        print("🔍 CURRENT TRADES ANALYSIS")
        print("=" * 50)

        active_count = len(bot.active_trades)
        print(f"📊 Active trades in bot memory: {active_count}")

        if active_count == 0:
            print("✅ No active trades found")
            return

        total_profit = 0.0
        losing_trades = []
        old_trades = []

        for trade_id, trade in bot.active_trades.items():
            try:
                mt5_ticket = trade.get('mt5_ticket', trade_id)
                symbol = trade.get('symbol', 'Unknown')
                entry_price = trade.get('entry_price', 0)

                # Get current position info
                position_info = await bot.mt5_interface.get_position_info(mt5_ticket)
                if position_info:
                    current_profit = float(getattr(position_info, 'profit', 0.0))
                    volume = float(getattr(position_info, 'volume', 0.0))
                    current_price = float(getattr(position_info, 'price_current', 0.0))

                    total_profit += current_profit

                    # Check if losing
                    if current_profit <= -1.0:
                        losing_trades.append({
                            'ticket': mt5_ticket,
                            'symbol': symbol,
                            'profit': current_profit,
                            'volume': volume
                        })

                    # Check age
                    trade_time = trade.get('timestamp', '')
                    age_status = "Unknown age"
                    if trade_time:
                        from datetime import datetime
                        try:
                            trade_datetime = datetime.fromisoformat(trade_time)
                            age_minutes = (datetime.now() - trade_datetime).total_seconds() / 60
                            age_status = f"{age_minutes:.0f} minutes old"
                            if age_minutes > 30:
                                old_trades.append(mt5_ticket)
                        except:
                            pass

                    status = "🔴" if current_profit < 0 else "🟢" if current_profit > 0 else "⚪"
                    print(f"{status} {symbol} (#{mt5_ticket})")
                    print(f"   P&L: ${current_profit:+.2f} | Volume: {volume} | {age_status}")
                    print(f"   Entry: {entry_price} → Current: {current_price}")
                    print()
                else:
                    print(f"⚠️  {symbol} (#{mt5_ticket}) - Position not found in MT5!")
                    print()

            except Exception as e:
                print(f"❌ Error checking trade {trade_id}: {e}")

        print("=" * 50)
        print("📊 SUMMARY:")
        print(f"   Total P&L: ${total_profit:+.2f}")
        print(f"   Losing trades: {len(losing_trades)}")
        print(f"   Old trades (>30min): {len(old_trades)}")

        if losing_trades:
            print("\n🔴 LOSING TRADES:")
            for trade in losing_trades:
                print(f"   {trade['symbol']} #{trade['ticket']}: ${trade['profit']:+.2f}")

        return {
            'total_trades': active_count,
            'total_profit': total_profit,
            'losing_trades': losing_trades,
            'old_trades': old_trades
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

async def close_heavy_losers(max_loss: float = 2.0):
    """Close trades losing more than specified amount"""
    try:
        from main import global_bot_instance

        if not global_bot_instance:
            print("❌ Bot not running")
            return

        bot = global_bot_instance
        if not bot.mt5_interface:
            print("❌ MT5 not connected")
            return

        print(f"🚨 CLOSING TRADES LOSING MORE THAN ${max_loss:.2f}")
        print("=" * 50)

        closed_count = 0
        total_saved = 0.0

        for trade_id, trade in list(bot.active_trades.items()):
            try:
                mt5_ticket = trade.get('mt5_ticket', trade_id)
                symbol = trade.get('symbol', 'Unknown')

                position_info = await bot.mt5_interface.get_position_info(mt5_ticket)
                if position_info:
                    current_profit = float(getattr(position_info, 'profit', 0.0))

                    if current_profit <= -max_loss:
                        print(f"🔴 Closing {symbol} #{mt5_ticket}: ${current_profit:.2f}")

                        close_result = await bot.mt5_interface.close_position(mt5_ticket)
                        if close_result:
                            del bot.active_trades[trade_id]
                            closed_count += 1
                            total_saved += abs(current_profit)
                            print("   ✅ Closed successfully")
                        else:
                            print("   ❌ Failed to close")
                    else:
                        print(f"✅ {symbol} #{mt5_ticket}: ${current_profit:.2f} (keeping)")

            except Exception as e:
                print(f"❌ Error closing trade {trade_id}: {e}")

        print("=" * 50)
        print("🛡️ RISK MANAGEMENT COMPLETE:")
        print(f"   Trades closed: {closed_count}")
        print(f"   Potential loss saved: ${total_saved:.2f}")

        return closed_count

    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

async def close_all_losing_trades():
    """Close ALL losing trades regardless of amount"""
    try:
        from main import global_bot_instance

        if not global_bot_instance:
            print("❌ Bot not running")
            return

        bot = global_bot_instance
        if not bot.mt5_interface:
            print("❌ MT5 not connected")
            return

        print("🚨 EMERGENCY: CLOSING ALL LOSING TRADES")
        print("=" * 50)

        closed_count = 0
        total_saved = 0.0

        for trade_id, trade in list(bot.active_trades.items()):
            try:
                mt5_ticket = trade.get('mt5_ticket', trade_id)
                symbol = trade.get('symbol', 'Unknown')

                position_info = await bot.mt5_interface.get_position_info(mt5_ticket)
                if position_info:
                    current_profit = float(getattr(position_info, 'profit', 0.0))

                    if current_profit < 0:  # Any losing trade
                        print(f"🔴 Closing {symbol} #{mt5_ticket}: ${current_profit:.2f}")

                        close_result = await bot.mt5_interface.close_position(mt5_ticket)
                        if close_result:
                            del bot.active_trades[trade_id]
                            closed_count += 1
                            total_saved += abs(current_profit)
                            print("   ✅ Closed successfully")
                        else:
                            print("   ❌ Failed to close")
                    else:
                        print(f"✅ {symbol} #{mt5_ticket}: ${current_profit:.2f} (profitable - keeping)")

            except Exception as e:
                print(f"❌ Error closing trade {trade_id}: {e}")

        print("=" * 50)
        print("🚨 EMERGENCY COMPLETE:")
        print(f"   Losing trades closed: {closed_count}")
        print(f"   Total loss prevented: ${total_saved:.2f}")

        return closed_count

    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

async def close_profitable_trades():
    """Close profitable trades to lock in gains"""
    try:
        from main import global_bot_instance

        if not global_bot_instance:
            print("❌ Bot not running")
            return

        bot = global_bot_instance
        if not bot.mt5_interface:
            print("❌ MT5 not connected")
            return

        print("💰 LOCKING IN PROFITS: CLOSING PROFITABLE TRADES")
        print("=" * 50)

        closed_count = 0
        total_profit = 0.0

        for trade_id, trade in list(bot.active_trades.items()):
            try:
                mt5_ticket = trade.get('mt5_ticket', trade_id)
                symbol = trade.get('symbol', 'Unknown')

                position_info = await bot.mt5_interface.get_position_info(mt5_ticket)
                if position_info:
                    current_profit = float(getattr(position_info, 'profit', 0.0))

                    if current_profit > 0:  # Any profitable trade
                        print(f"🟢 Closing {symbol} #{mt5_ticket}: ${current_profit:.2f}")

                        close_result = await bot.mt5_interface.close_position(mt5_ticket)
                        if close_result:
                            del bot.active_trades[trade_id]
                            closed_count += 1
                            total_profit += current_profit
                            print("   ✅ Profit locked in!")
                        else:
                            print("   ❌ Failed to close")
                    else:
                        print(f"🔴 {symbol} #{mt5_ticket}: ${current_profit:.2f} (losing - keeping)")

            except Exception as e:
                print(f"❌ Error closing trade {trade_id}: {e}")

        print("=" * 50)
        print("💰 PROFIT LOCKING COMPLETE:")
        print(f"   Profitable trades closed: {closed_count}")
        print(f"   Total profit locked: ${total_profit:.2f}")

        return total_profit

    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

def main():
    """Main menu for risk management tools"""
    print("🚨 ADVANCED RISK MANAGEMENT TOOLS")
    print("=" * 40)

    while True:
        print("\nOptions:")
        print("1. 📊 Check current trades")
        print("2. 🚨 Close heavy losers (>$2.00)")
        print("3. 🔴 Close ALL losing trades")
        print("4. 💰 Close profitable trades (lock profits)")
        print("5. 🛠️  Custom loss threshold")
        print("6. 🏃 Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            print("\n" + "="*50)
            asyncio.run(debug_current_trades())
            print("="*50)

        elif choice == "2":
            print("\n⚠️  This will close trades losing more than $2.00")
            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                print("\n" + "="*50)
                asyncio.run(close_heavy_losers(2.0))
                print("="*50)

        elif choice == "3":
            print("\n🚨 WARNING: This will close ALL losing trades!")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                print("\n" + "="*50)
                asyncio.run(close_all_losing_trades())
                print("="*50)

        elif choice == "4":
            print("\n💰 This will close all profitable trades to lock in gains")
            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                print("\n" + "="*50)
                asyncio.run(close_profitable_trades())
                print("="*50)

        elif choice == "5":
            try:
                threshold = float(input("Enter loss threshold ($): "))
                print(f"\n⚠️  This will close trades losing more than ${threshold:.2f}")
                confirm = input("Continue? (yes/no): ").strip().lower()
                if confirm in ['yes', 'y']:
                    print("\n" + "="*50)
                    asyncio.run(close_heavy_losers(threshold))
                    print("="*50)
            except ValueError:
                print("❌ Invalid amount entered")

        elif choice == "6":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
