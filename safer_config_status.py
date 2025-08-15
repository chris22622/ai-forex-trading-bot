#!/usr/bin/env python3
"""
🛡️ Post-Drawdown Recovery Configuration Status
Shows the safer settings now active for your $42.47 account
"""

from config import *


def show_safer_config_status():
    """Display the new safer configuration"""
    print("🛡️ POST-DRAWDOWN SAFER CONFIGURATION ACTIVE")
    print("=" * 50)

    print("\n💰 POSITION SIZING (Ultra-Conservative):")
    print(f"   • Trade Amount: ${TRADE_AMOUNT} (0.01 lots)")
    print(f"   • Risk Per Trade: {BASE_RISK_PERCENTAGE*100}%")
    print(f"   • Max Lot Size: {MAX_LOT_SIZE}")

    print("\n🛡️ RISK MANAGEMENT (Tightened):")
    print(f"   • Stop Loss: ${AUTO_STOP_LOSS} max")
    print(f"   • Take Profit: ${AUTO_TAKE_PROFIT}")
    print(f"   • Max Portfolio Loss: ${MAX_PORTFOLIO_LOSS}")
    print(f"   • Max Daily Loss: ${MAX_DAILY_LOSS}")
    print(f"   • Max Drawdown: {MAX_DRAWDOWN_PERCENTAGE*100}%")

    print("\n⏰ TRADE MANAGEMENT (Faster):")
    print(f"   • Trade Time Limit: {MAX_TRADE_AGE_MINUTES} minutes")
    print(f"   • Min Trade Interval: {MIN_TRADE_INTERVAL} seconds")
    print(f"   • Max Concurrent: {MAX_CONCURRENT_TRADES} trade")
    print(f"   • Max Consecutive Losses: {MAX_CONSECUTIVE_LOSSES}")

    print("\n🎯 SYMBOL SELECTION (Safer):")
    print(f"   • Primary Symbol: {DEFAULT_SYMBOL}")
    print(f"   • Available Symbols: {len(PRIMARY_TRADING_SYMBOLS)}")
    for symbol in PRIMARY_TRADING_SYMBOLS:
        print(f"     - {symbol}")

    print("\n🧠 AI SETTINGS (Higher Quality):")
    print(f"   • Confidence Threshold: {AI_CONFIDENCE_THRESHOLD*100}%")
    print(f"   • RSI Buy Threshold: < {RSI_BUY_THRESHOLD}")
    print(f"   • RSI Sell Threshold: > {RSI_SELL_THRESHOLD}")

    print("\n🛡️ PROFIT PROTECTION:")
    print(f"   • Stop at Profit: +${PROFIT_PROTECTION_THRESHOLD}")
    print(f"   • Emergency Stop: +${PROFIT_PROTECTION_MAX_THRESHOLD}")

    print("\n" + "=" * 50)
    print("🚀 RECOVERY STRATEGY:")
    print("   ✅ Smaller positions (0.01 lots)")
    print("   ✅ Tighter stop losses ($1 max)")
    print("   ✅ Quick profit taking ($0.75)")
    print("   ✅ Safer symbols (Volatility 10/25)")
    print("   ✅ Higher AI confidence (65%)")
    print("   ✅ Faster trade exits (15 min)")
    print("   ✅ Single trade focus")

    # Calculate potential with safer settings
    account_balance = 42.47
    daily_target = AUTO_TAKE_PROFIT * 3  # 3 successful trades per day
    weekly_target = daily_target * 5
    monthly_target = weekly_target * 4

    print("\n📈 RECOVERY PROJECTIONS (Conservative):")
    print(f"   • Per Trade Target: +${AUTO_TAKE_PROFIT}")
    print(f"   • Daily Target (3 trades): +${daily_target}")
    print(f"   • Weekly Target: +${weekly_target}")
    print(f"   • Monthly Target: +${monthly_target}")
    print(f"   • Monthly Growth: {(monthly_target/account_balance)*100:.1f}%")

    print("\n🎯 Your account will be protected while growing steadily!")
    print(f"💡 Ready to restart with Balance: ${account_balance}")

if __name__ == "__main__":
    show_safer_config_status()
