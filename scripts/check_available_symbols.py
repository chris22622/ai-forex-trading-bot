"""
Quick script to check what Volatility Index symbols are available in MT5
"""


import MetaTrader5 as mt5


def check_symbols():
    """Check available symbols"""
    print("🔍 Checking available MT5 symbols...")

    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return False

    print("✅ MT5 initialized successfully")

    # Test common Volatility Index symbol names
    test_symbols = [
        "Volatility 75 Index",
        "Volatility 100 Index",
        "Volatility75",
        "Volatility100",
        "VIX75",
        "VIX100",
        "R_75",
        "R_100",
        "CRASH1000",
        "BOOM1000"
    ]

    print("\n📊 Testing symbol availability:")
    available_symbols = []

    for symbol in test_symbols:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is not None:
            print(f"✅ {symbol}: Available")
            print(f"   - Volume min: {getattr(symbol_info, 'volume_min', 'N/A')}")
            print(f"   - Volume max: {getattr(symbol_info, 'volume_max', 'N/A')}")
            print(f"   - Volume step: {getattr(symbol_info, 'volume_step', 'N/A')}")
            print(f"   - Point: {getattr(symbol_info, 'point', 'N/A')}")
            print(f"   - Digits: {getattr(symbol_info, 'digits', 'N/A')}")
            available_symbols.append(symbol)
        else:
            print(f"❌ {symbol}: Not available")

    print(f"\n📈 Found {len(available_symbols)} available volatility symbols:")
    for symbol in available_symbols:
        print(f"   • {symbol}")

    # Get all symbols from market watch
    print("\n🔍 Checking Market Watch symbols...")
    all_symbols = mt5.symbols_get()
    if all_symbols:
        volatility_symbols = [s.name for s in all_symbols if 'volat' in s.name.lower() or 'crash' in s.name.lower() or 'boom' in s.name.lower() or 'r_' in s.name.lower()]
        print(f"📊 Found {len(volatility_symbols)} volatility-related symbols in Market Watch:")
        for symbol in volatility_symbols[:10]:  # Show first 10
            print(f"   • {symbol}")
        if len(volatility_symbols) > 10:
            print(f"   ... and {len(volatility_symbols) - 10} more")

    mt5.shutdown()
    return True

if __name__ == "__main__":
    check_symbols()
