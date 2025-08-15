"""
Quick test to check if Deriv synthetic indices are active 24/7 or have market hours
"""

import time
from datetime import datetime

import MetaTrader5 as mt5


def check_market_status():
    """Check if markets are open for synthetic indices"""

    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return

    # Try to connect to live account
    if not mt5.login(62233085, password="Goon22622$", server="DerivVU-Server"):
        print(f"❌ Login failed: {mt5.last_error()}")
        mt5.shutdown()
        return

    print("✅ Connected to MT5")

    # Test multiple volatility symbols
    test_symbols = [
        "Volatility 10 Index",
        "Volatility 10 (1s) Index",
        "Volatility 25 Index",
        "Volatility 25 (1s) Index",
        "Volatility 50 Index",
        "Step Index"
    ]

    for symbol in test_symbols:
        print(f"\n🔍 Testing: {symbol}")

        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print("   ❌ Symbol not found")
            continue

        # Check market status
        print(f"   📊 Trade mode: {symbol_info.trade_mode}")
        print(f"   🕒 Market open: {symbol_info.session_deals > 0 or symbol_info.trade_mode == 4}")

        # Select symbol and try to get price
        if mt5.symbol_select(symbol, True):
            print("   ✅ Symbol selected")

            # Wait and get tick
            time.sleep(0.5)
            tick = mt5.symbol_info_tick(symbol)

            if tick:
                print("   💰 Prices available:")
                print(f"      Bid: {tick.bid}")
                print(f"      Ask: {tick.ask}")
                print(f"      Last: {tick.last}")
                print(f"      Time: {datetime.fromtimestamp(tick.time)}")

                if tick.bid > 0 or tick.ask > 0 or tick.last > 0:
                    print("   ✅ ACTIVE - Valid prices available!")
                else:
                    print("   ⚠️  All prices are zero")
            else:
                print("   ❌ No tick data available")
        else:
            print("   ❌ Cannot select symbol")

    # Check current time and server time
    server_time = mt5.symbol_info_tick("Volatility 10 Index")
    if server_time:
        print(f"\n🕒 Server time: {datetime.fromtimestamp(server_time.time)}")
    print(f"🕒 Local time: {datetime.now()}")

    mt5.shutdown()

if __name__ == "__main__":
    check_market_status()
