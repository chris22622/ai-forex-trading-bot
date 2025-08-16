"""
Enhanced Emergency Reset Script for Trading Bot
Comprehensive reset tool with multiple options for various stuck states
"""

import asyncio
import sys

from main import global_bot_instance


async def emergency_reset_basic():
    """Basic emergency reset - reset consecutive losses only"""
    print("🚨 BASIC EMERGENCY RESET STARTING...")

    if not global_bot_instance:
        print("❌ Bot not running - cannot reset")
        return False

    bot = global_bot_instance

    # Reset consecutive losses
    old_consecutive = bot.consecutive_losses
    bot.consecutive_losses = 0

    print("✅ BASIC RESET COMPLETE:")
    print(f"   Consecutive losses: {old_consecutive} → 0")
    print("🚀 Bot ready to resume trading!")

    # Check risk limits
    risk_status = bot._check_risk_limits()
    print(f"🔍 Risk limits check: {'✅ PASS' if risk_status else '❌ FAIL'}")

    return True

async def emergency_reset_full():
    """Full emergency reset - reset all limits"""
    print("🚨 FULL EMERGENCY RESET STARTING...")

    if not global_bot_instance:
        print("❌ Bot not running - cannot reset")
        return False

    bot = global_bot_instance

    # Store old values
    old_consecutive = bot.consecutive_losses
    old_daily = bot.daily_profit

    # Reset all limits
    bot.consecutive_losses = 0
    bot.daily_profit = 0.0

    print("✅ FULL RESET COMPLETE:")
    print(f"   Consecutive losses: {old_consecutive} → 0")
    print(f"   Daily P&L: ${old_daily:.2f} → $0.00")
    print("🚀 Bot ready to resume trading with fresh limits!")

    # Check risk limits
    risk_status = bot._check_risk_limits()
    print(f"🔍 Risk limits check: {'✅ PASS' if risk_status else '❌ FAIL'}")

    return True

async def emergency_force_close_profitable():
    """Emergency force close all profitable trades"""
    print("🚨 EMERGENCY PROFITABLE TRADE CLOSURE STARTING...")

    if not global_bot_instance:
        print("❌ Bot not running - cannot close trades")
        return False

    bot = global_bot_instance

    if not bot.active_trades:
        print("ℹ️ No active trades to close")
        return True

    closed_count = await bot.force_close_all_profitable_trades()
    print(f"✅ PROFITABLE TRADE CLOSURE COMPLETE: {closed_count} trades closed")

    return True

async def check_bot_status():
    """Check current bot status and diagnostics"""
    print("📊 BOT STATUS CHECK...")

    if not global_bot_instance:
        print("❌ Bot not running")
        return False

    bot = global_bot_instance

    print(f"""
📊 Current Bot Status:
🔸 Running: {'✅ YES' if bot.running else '❌ NO'}
🔸 MT5 Connected: {'✅ YES' if bot.mt5_connected else '❌ NO'}
🔸 Balance: ${bot.current_balance:.2f}
🔸 Active Trades: {len(bot.active_trades)}
🔸 Consecutive Losses: {bot.consecutive_losses}/100
🔸 Daily P&L: ${bot.daily_profit:+.2f}
🔸 Risk Limits: {'✅ PASS' if bot._check_risk_limits() else '❌ BLOCKED'}

📈 Session Stats:
🔸 Total Trades: {bot.session_stats['total_trades']}
🔸 Wins: {bot.session_stats['wins']}
🔸 Losses: {bot.session_stats['losses']}
🔸 Win Rate: {bot.get_win_rate():.1f}%
🔸 Total P&L: ${bot.session_stats['total_profit']:+.2f}
""")

    # Check active trades details
    if bot.active_trades:
        print("📋 Active Trades:")
        for trade_id, trade in bot.active_trades.items():
            f"   🔸 {trade_id}"
            f" {trade.get('action', 'Unknown')} {trade.get('symbol', 'Unknown')}"

    return True

async def interactive_reset():
    """Interactive reset menu"""
    print("""
🚨 ENHANCED EMERGENCY RESET MENU
==================================

Available Options:
1. Basic Reset (consecutive losses only)
2. Full Reset (all limits)
3. Force Close Profitable Trades
4. Check Bot Status
5. Exit

Choose an option (1-5):""")

    try:
        choice = input().strip()

        if choice == "1":
            await emergency_reset_basic()
        elif choice == "2":
            await emergency_reset_full()
        elif choice == "3":
            await emergency_force_close_profitable()
        elif choice == "4":
            await check_bot_status()
        elif choice == "5":
            print("👋 Exiting emergency reset tool")
            return
        else:
            print("❌ Invalid choice")
            return

    except KeyboardInterrupt:
        print("\n👋 Emergency reset interrupted")
    except Exception as e:
        print(f"❌ Emergency reset error: {e}")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "basic":
            asyncio.run(emergency_reset_basic())
        elif command == "full":
            asyncio.run(emergency_reset_full())
        elif command == "close":
            asyncio.run(emergency_force_close_profitable())
        elif command == "status":
            asyncio.run(check_bot_status())
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: basic, full, close, status")
    else:
        # Interactive mode
        asyncio.run(interactive_reset())

if __name__ == "__main__":
    main()
