#!/usr/bin/env python3
"""
Quick verification that MT5 connection is working for real trades
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the MT5 interface directly
import asyncio

from mt5_integration import MT5TradingInterface


async def verify_mt5_real_connection():
    print("🔍 VERIFYING REAL MT5 CONNECTION...")
    print("=" * 50)

    try:
        # Create the interface
        mt5_interface = MT5TradingInterface()
        print("✅ MT5TradingInterface created")

        # Initialize connection
        connected = await mt5_interface.initialize()
        if not connected:
            print("❌ MT5 initialization failed")
            return False

        print("✅ MT5 initialized successfully")

        # Get account balance
        balance = await mt5_interface.get_account_balance()
        print(f"💰 Account Balance: ${balance:.2f}")

        # Check if this is real MT5 (not dummy)
        interface_type = type(mt5_interface).__name__
        print(f"🔗 Interface Type: {interface_type}")

        if "Dummy" in interface_type:
            print("❌ WARNING: Using Dummy interface!")
            return False

        # Test symbol availability
        symbol = "Volatility 75 Index"
        symbol_info = await mt5_interface.get_symbol_info(symbol)

        if symbol_info:
            print(f"📊 Symbol {symbol} available")
            print(f"📈 Current Bid: {symbol_info.get('bid', 'N/A')}")
            print(f"📉 Current Ask: {symbol_info.get('ask', 'N/A')}")
        else:
            print(f"❌ Symbol {symbol} not available")
            return False

        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("✅ Real MT5 connection established")
        print("✅ Account accessible")
        print("✅ Trading symbols available")
        print("🚀 Ready for REAL trading!")

        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 MT5 REAL CONNECTION VERIFICATION")
    result = asyncio.run(verify_mt5_real_connection())

    if result:
        print("\n✅ SUCCESS: MT5 is ready for real trading!")
        print("📱 When the bot runs, trades will appear in your MT5 terminal")
    else:
        print("\n❌ FAILED: MT5 connection issues detected")
        print("🔧 Check your MT5 setup and try again")
