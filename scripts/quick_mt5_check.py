#!/usr/bin/env python3
"""Quick MT5 symbol checker"""

import MetaTrader5 as mt5

# Initialize MT5
if mt5.initialize():
    print("✅ MT5 Connected!")

    # Get account info
    account = mt5.account_info()
    if account:
        print(f"💰 Balance: ${account.balance:.2f}")
        print(f"🏦 Server: {account.server}")
        print(f"👤 Login: {account.login}")

    # Get all symbols
    symbols = mt5.symbols_get()
    print(f"\n📊 Total symbols: {len(symbols) if symbols else 0}")

    # Find volatility-related symbols
    if symbols:
        vol_symbols = []
        for symbol in symbols:
            name = symbol.name
            if any(keyword in name for keyword in ['Volatility', 'R_', 'Vol', 'Crash', 'Boom']):
                vol_symbols.append(name)

        print(f"\n🎯 Found {len(vol_symbols)} volatility/synthetic symbols:")
        for sym in vol_symbols[:15]:  # Show first 15
            print(f"   - {sym}")

    # Test specific symbols
    test_symbols = ["Volatility 75 Index", "Volatility 100 Index", "R_75", "R_100"]
    print("\n🔍 Testing specific symbols:")
    for sym in test_symbols:
        tick = mt5.symbol_info_tick(sym)
        if tick:
            print(f"   ✅ {sym}: {tick.ask:.5f}")
        else:
            print(f"   ❌ {sym}: Not available")

    mt5.shutdown()
else:
    print("❌ MT5 connection failed")
    error = mt5.last_error()
    print(f"Error: {error}")
