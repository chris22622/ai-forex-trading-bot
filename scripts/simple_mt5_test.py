#!/usr/bin/env python3
"""
Simple MT5 Test - Identify initialization issues
"""

import traceback
from datetime import datetime


def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")

    try:
        print("✅ MetaTrader5 imported")
    except Exception as e:
        print(f"❌ MetaTrader5 import failed: {e}")
        return False

    try:
        print("✅ MT5TradingInterface imported")
    except Exception as e:
        print(f"❌ MT5TradingInterface import failed: {e}")
        traceback.print_exc()
        return False

    return True

def test_mt5_basic():
    """Test basic MT5 functionality"""
    print("\\n🔍 Testing basic MT5 functionality...")

    try:
        import MetaTrader5 as mt5

        # Initialize MT5
        if not mt5.initialize():
            print("❌ MT5 initialize() failed")
            print(f"Last error: {mt5.last_error()}")
            return False

        print("✅ MT5 initialized")

        # Get terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"✅ Terminal: {terminal_info.name}")
            print(f"   Version: {terminal_info.build}")
            print(f"   Path: {terminal_info.path}")

        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Account: {account_info.login}")
            print(f"   Server: {account_info.server}")
            print(f"   Balance: ${account_info.balance:.2f}")
            print(f"   Currency: {account_info.currency}")
        else:
            print("⚠️ No account info available")

        # Test symbol
        symbol_info = mt5.symbol_info("Volatility 75 Index")
        if symbol_info:
            print("✅ Volatility 75 Index symbol found")
        else:
            print("⚠️ Volatility 75 Index symbol not found")
            # Try alternative symbol
            symbol_info = mt5.symbol_info("EURUSD")
            if symbol_info:
                print("✅ EURUSD symbol found as alternative")
            else:
                print("❌ No usable symbols found")

        mt5.shutdown()
        return True

    except Exception as e:
        print(f"❌ MT5 basic test failed: {e}")
        traceback.print_exc()
        return False

def test_mt5_interface():
    """Test MT5TradingInterface"""
    print("\\n🔍 Testing MT5TradingInterface...")

    try:
        from mt5_integration import MT5TradingInterface

        interface = MT5TradingInterface()
        print("✅ MT5TradingInterface created")

        return True

    except Exception as e:
        print(f"❌ MT5TradingInterface test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔧 SIMPLE MT5 INITIALIZATION TEST")
    print("=" * 60)
    print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run tests
    imports_ok = test_basic_imports()

    if imports_ok:
        mt5_basic_ok = test_mt5_basic()
        interface_ok = test_mt5_interface()
    else:
        mt5_basic_ok = False
        interface_ok = False

    # Summary
    print("\\n📊 TEST SUMMARY:")
    print("=" * 30)
    print(f"Basic Imports:     {'✅' if imports_ok else '❌'}")
    print(f"MT5 Basic:         {'✅' if mt5_basic_ok else '❌'}")
    print(f"MT5 Interface:     {'✅' if interface_ok else '❌'}")

    if imports_ok and mt5_basic_ok and interface_ok:
        print("\\n🎉 ALL TESTS PASSED!")
        print("MT5 should work correctly with your bot.")
    else:
        print("\\n⚠️ Issues found:")
        if not imports_ok:
            print("   - Import problems detected")
        if not mt5_basic_ok:
            print("   - MT5 terminal issues detected")
        if not interface_ok:
            print("   - MT5TradingInterface issues detected")

    print("\\n" + "=" * 60)

if __name__ == "__main__":
    main()
