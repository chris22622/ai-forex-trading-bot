"""
Quick MT5 Connection Test with Your Credentials
"""
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import time

    import MetaTrader5 as mt5

    from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

    print("🔄 Testing MT5 connection with your credentials...")
    print(f"Login: {MT5_LOGIN}")
    print(f"Server: {MT5_SERVER}")
    print(f"Password: {'*' * len(MT5_PASSWORD)}")

    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        print(f"Error: {mt5.last_error()}")
        sys.exit(1)

    print("✅ MT5 initialized successfully")

    # Attempt login
    print(f"🔄 Attempting login to {MT5_SERVER}...")

    authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)

    if authorized:
        print("✅ LOGIN SUCCESSFUL!")

        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"💰 Account Balance: ${account_info.balance:.2f}")
            print(f"💳 Account Currency: {account_info.currency}")
            print(f"🏦 Account Name: {account_info.name}")
            print(f"📊 Account Leverage: 1:{account_info.leverage}")
            print(f"🌐 Server: {account_info.server}")
            print(f"🏢 Company: {account_info.company}")

        # Test symbol availability
        print("\n📊 Testing symbol availability...")
        symbols = ["Volatility 75 Index", "Volatility 100 Index", "EURUSD", "GBPUSD"]

        for symbol in symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                print(f"✅ {symbol} - Available (Spread: {symbol_info.spread})")
            else:
                print(f"❌ {symbol} - Not available")

        # Get current tick for Volatility 75 Index
        tick = mt5.symbol_info_tick("Volatility 75 Index")
        if tick:
            print(f"\n📈 Current Volatility 75 Index price: {tick.bid:.5f}")

        print("\n🎉 MT5 CONNECTION TEST PASSED!")
        print("✅ Your bot is ready to trade on demo account!")

    else:
        print("❌ LOGIN FAILED!")
        error = mt5.last_error()
        print(f"Error code: {error[0]}")
        print(f"Error message: {error[1]}")

        if error[0] == 10004:
            print("🔧 This usually means invalid credentials. Please check:")
            print("   - Login number is correct")
            print("   - Password is correct")
            print("   - Server name is correct")
        elif error[0] == 10007:
            print("🔧 Server connection issue. Please check:")
            print("   - Internet connection")
            print("   - Server name spelling")

    # Cleanup
    mt5.shutdown()

except ImportError:
    print("❌ MetaTrader5 library not found!")
    print("Run: pip install MetaTrader5")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
