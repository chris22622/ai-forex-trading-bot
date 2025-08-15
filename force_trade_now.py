"""
🚀 EMERGENCY FORCE TRADING SCRIPT
Force immediate trading by manipulating the running bot
"""
import os
import sys
import time
from datetime import datetime

import MetaTrader5 as mt5

# Add current directory to path to import main modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_quick_signal():
    """Get immediate trading signal based on current price movement"""
    if not mt5.initialize():
        return None

    symbol = "Volatility 50 Index"

    # Get current tick
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return None

    current_price = tick.bid

    # Get last few ticks for quick trend
    rates = mt5.copy_ticks_from(symbol, datetime.now(), 10, mt5.COPY_TICKS_ALL)
    if rates is None or len(rates) < 3:
        return "BUY"  # Default signal if no data

    # Simple momentum check
    recent_prices = [r.bid for r in rates[-3:]]
    if recent_prices[-1] > recent_prices[0]:
        return "BUY"
    else:
        return "SELL"

def force_immediate_trade():
    """Force an immediate trade using quick signal"""
    print("🚀 FORCING IMMEDIATE TRADE...")

    if not mt5.initialize():
        print("❌ Could not initialize MT5")
        return False

    # Get account info
    account_info = mt5.account_info()
    if not account_info:
        print("❌ Could not get account info")
        return False

    balance = account_info.balance
    print(f"💰 Account Balance: ${balance}")

    signal = get_quick_signal()
    if not signal:
        print("❌ Could not get trading signal")
        return False

    print(f"📈 Signal: {signal}")

    # Calculate lot size (2% risk)
    lot_size = max(0.1, round(balance * 0.02 / 100, 2))
    symbol = "Volatility 50 Index"

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"❌ Could not get price for {symbol}")
        return False

    current_price = tick.ask if signal == "BUY" else tick.bid
    print(f"💲 Current Price: {current_price}")

    # Place order
    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - 0.5  # 50 points SL
        tp = price + 1.0  # 100 points TP
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + 0.5  # 50 points SL
        tp = price - 1.0  # 100 points TP

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "EMERGENCY_FORCE_TRADE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    print("📋 Order Details:")
    print(f"   Symbol: {symbol}")
    print(f"   Type: {signal}")
    print(f"   Volume: {lot_size}")
    print(f"   Price: {price}")
    print(f"   SL: {sl}")
    print(f"   TP: {tp}")

    # Send order
    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order failed: {result.retcode} - {result.comment}")
        return False

    print("✅ ORDER PLACED SUCCESSFULLY!")
    print(f"   Order #: {result.order}")
    print(f"   Deal #: {result.deal}")
    print(f"   Volume: {result.volume}")
    print(f"   Price: {result.price}")

    return True

def check_current_positions():
    """Check if there are any current positions"""
    if not mt5.initialize():
        return []

    positions = mt5.positions_get()
    if positions:
        print(f"📊 Current Positions: {len(positions)}")
        for pos in positions:
            print(f"   {pos.symbol}: {pos.type_str} {pos.volume} @ {pos.price_open}")
    else:
        print("📊 No current positions")

    return positions

if __name__ == "__main__":
    print("🚀 EMERGENCY FORCE TRADING SCRIPT")
    print("=" * 50)

    # Check current positions first
    check_current_positions()

    # Force immediate trade
    success = force_immediate_trade()

    if success:
        print("\n✅ EMERGENCY TRADE PLACED!")
        print("📱 Check your MT5 terminal for the new position")

        # Wait a moment and check positions again
        time.sleep(2)
        check_current_positions()
    else:
        print("\n❌ Failed to place emergency trade")

    input("\nPress Enter to exit...")
