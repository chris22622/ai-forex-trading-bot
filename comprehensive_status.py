#!/usr/bin/env python3
"""
📊 COMPREHENSIVE TRADING STATUS DASHBOARD
Quick overview of all bot systems and current trading status
"""

import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def comprehensive_status_check():
    """Complete status check of all trading systems"""
    print("📊 COMPREHENSIVE TRADING STATUS DASHBOARD")
    print("=" * 60)
    print(f"🕒 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # 1. Check Bot Instance Status
    print("🤖 BOT INSTANCE STATUS:")
    print("-" * 30)
    try:
        from main import global_bot_instance

        if global_bot_instance:
            bot = global_bot_instance
            running = getattr(bot, 'running', False)
            current_balance = getattr(bot, 'current_balance', 0)
            starting_balance = getattr(bot, 'starting_balance', 0)
            pnl = current_balance - starting_balance
            active_trades = len(getattr(bot, 'active_trades', {}))
            consecutive_losses = getattr(bot, 'consecutive_losses', 0)

            print("✅ Bot Instance: FOUND")
            print(f"🔄 Status: {'RUNNING' if running else 'STOPPED'}")
            print(f"💰 Current Balance: ${current_balance:.2f}")
            print(f"📊 Starting Balance: ${starting_balance:.2f}")
            print(f"📈 Session P&L: ${pnl:+.2f}")
            print(f"🎯 Active Trades: {active_trades}")
            print(f"🔴 Consecutive Losses: {consecutive_losses}")

            # Profit Protection Status
            protection_enabled = getattr(bot, 'profit_protection_enabled', False)
            protection_triggered = getattr(bot, 'profit_protection_triggered', False)
            protection_threshold = getattr(bot, 'profit_protection_threshold', 10)

            print(f"🛡️ Profit Protection: {'ENABLED' if protection_enabled else 'DISABLED'}")
            print(f"🛡️ Protection Triggered: {'YES' if protection_triggered else 'NO'}")
            print(f"🛡️ Protection Threshold: ${protection_threshold:.2f}")

        else:
            print("❌ Bot Instance: NOT FOUND")
            print("💡 Bot is not currently running")

    except Exception as e:
        print(f"❌ Error checking bot: {e}")

    print("")

    # 2. Check Configuration
    print("⚙️ CURRENT CONFIGURATION:")
    print("-" * 30)
    try:
        from config import (
            AUTO_STOP_LOSS,
            AUTO_TAKE_PROFIT,
            MAX_CONCURRENT_TRADES,
            MAX_CONSECUTIVE_LOSSES,
            MAX_DAILY_LOSS,
            PROFIT_PROTECTION_THRESHOLD,
            TRADE_AMOUNT,
        )

        print(f"💰 Take Profit Target: ${AUTO_TAKE_PROFIT}")
        print(f"🛡️ Stop Loss Limit: ${AUTO_STOP_LOSS}")
        print(f"💵 Trade Amount: ${TRADE_AMOUNT}")
        print(f"🔢 Max Concurrent Trades: {MAX_CONCURRENT_TRADES}")
        print(f"🛡️ Profit Protection: ${PROFIT_PROTECTION_THRESHOLD}")
        print(f"🔴 Max Consecutive Losses: {MAX_CONSECUTIVE_LOSSES}")
        print(f"📉 Max Daily Loss: ${MAX_DAILY_LOSS}")

    except ImportError as e:
        print(f"⚠️ Configuration error: {e}")

    print("")

    # 3. Check MT5 Status
    print("📈 MT5 CONNECTION STATUS:")
    print("-" * 30)
    try:
        import MetaTrader5 as mt5

        if mt5.initialize():
            account_info = mt5.account_info()
            if account_info:
                print("✅ MT5 Status: CONNECTED")
                print(f"🆔 Account: {account_info.login}")
                print(f"🌐 Server: {account_info.server}")
                print(f"💰 MT5 Balance: ${account_info.balance:.2f}")
                print(f"💵 MT5 Equity: ${account_info.equity:.2f}")
                print(f"🔄 Trading Allowed: {'YES' if account_info.trade_allowed else 'NO'}")
            else:
                print("❌ MT5 Status: NO ACCOUNT INFO")
            mt5.shutdown()
        else:
            print("❌ MT5 Status: FAILED TO INITIALIZE")

    except ImportError:
        print("❌ MT5 Library: NOT INSTALLED")
    except Exception as e:
        print(f"❌ MT5 Error: {e}")

    print("")

    # 4. Performance Analysis
    print("📊 PERFORMANCE ANALYSIS:")
    print("-" * 30)
    try:
        if 'global_bot_instance' in locals() and global_bot_instance:
            bot = global_bot_instance
            pnl = getattr(bot, 'current_balance', 0) - getattr(bot, 'starting_balance', 0)

            if pnl >= 5:
                print("🎉 STATUS: EXCELLENT - Target profit reached!")
                print("💡 RECOMMENDATION: Consider taking profits")
            elif pnl >= 2:
                print("✅ STATUS: GOOD - Profitable session")
                print("💡 RECOMMENDATION: Monitor for profit target")
            elif pnl >= -5:
                print("📊 STATUS: NEUTRAL - Normal trading")
                print("💡 RECOMMENDATION: Continue monitoring")
            elif pnl >= -15:
                print("⚠️ STATUS: CAUTION - Moderate losses")
                print("💡 RECOMMENDATION: Consider reducing position sizes")
            else:
                print("🚨 STATUS: DANGER - Significant losses")
                print("💡 RECOMMENDATION: STOP TRADING and review strategy")
        else:
            print("ℹ️ No active session data available")

    except Exception as e:
        print(f"❌ Performance analysis error: {e}")

    print("")

    # 5. Quick Actions
    print("🚀 QUICK ACTIONS:")
    print("-" * 30)
    print("📊 Check Status: python comprehensive_status.py")
    print("🚨 Emergency Stop: python emergency_damage_control.py")
    print("🚀 Start Bot: python restart_profitable_bot.py")
    print("🤖 Run Bot: python main.py")

    print("")
    print("=" * 60)

if __name__ == "__main__":
    comprehensive_status_check()
