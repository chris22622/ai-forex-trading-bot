#!/usr/bin/env python3
"""
Quick MT5 Trading Bot Launcher
Bypasses complex initialization for immediate trading
"""

import random

import MetaTrader5 as mt5


def quick_mt5_check():
    """Quick MT5 connection check"""
    print("🔍 Quick MT5 Connection Check...")

    if not mt5.initialize():
        print("❌ MT5 not initialized")
        return False

    # Login
    login = 31899532
    password = "Goon22622$"
    server = "Deriv-Demo"

    if not mt5.login(login, password=password, server=server):
        print(f"❌ Login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    account_info = mt5.account_info()
    if account_info is None:
        print("❌ No account info")
        mt5.shutdown()
        return False

    print(f"✅ Connected: Account {account_info.login}")
    print(f"💰 Balance: ${account_info.balance:.2f}")

    return True

def quick_trade_test():
    """Place a quick test trade"""
    symbol = "Volatility 75 Index"
    lot = 0.01

    # Get current price
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ {symbol} not found")
        return False

    price = mt5.symbol_info_tick(symbol)
    if price is None:
        print(f"❌ No price data for {symbol}")
        return False

    print(f"📊 Current price: {price.bid}")

    # Simple trade
    action = random.choice(["BUY", "SELL"])
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price_to_use = price.ask if action == "BUY" else price.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price_to_use,
        "deviation": 20,
        "magic": 234000,
        "comment": "Quick test trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,  # This works for Deriv
    }

    print(f"🔄 Placing {action} trade...")
    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Trade failed: {result.retcode} - {result.comment}")
        return False

    print(f"✅ Trade successful! Ticket: {result.order}")
    return True

def main():
    """Main quick test function"""
    print("🚀 QUICK MT5 TRADER")
    print("=" * 50)

    if not quick_mt5_check():
        print("❌ MT5 connection failed")
        return

    print("\n🎯 Testing trade execution...")
    if quick_trade_test():
        print("✅ Quick trade test successful!")
    else:
        print("❌ Quick trade test failed")

    mt5.shutdown()
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()
