#!/usr/bin/env python3
"""Direct MT5 test without importing main.py"""

import asyncio

async def test_direct_mt5():
    print("🔧 Testing MT5 directly...")
    
    # Test 1: Import MetaTrader5
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5 imported")
        
        # Test connection
        if mt5.initialize():
            print("✅ MT5 terminal connected")
            balance = mt5.account_info().balance if mt5.account_info() else 0
            print(f"Account Balance: ${balance:.2f}")
            mt5.shutdown()
        else:
            error = mt5.last_error()
            print(f"❌ MT5 connection failed: {error}")
    except Exception as e:
        print(f"❌ MetaTrader5 error: {e}")
    
    # Test 2: Import mt5_integration
    try:
        from mt5_integration import MT5TradingInterface
        print("✅ mt5_integration imported")
        
        interface = MT5TradingInterface()
        print("✅ MT5TradingInterface created")
        
        # Test initialization
        result = await interface.initialize()
        print(f"Interface Initialize: {result}")
        
        if result:
            balance = await interface.get_account_balance()
            print(f"Interface Balance: ${balance:.2f}")
            print("🚀 MT5 integration is WORKING!")
            return True
        else:
            print("❌ MT5 interface failed to initialize")
            return False
            
    except Exception as e:
        print(f"❌ mt5_integration error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_direct_mt5())
    if result:
        print("\n✅ CONCLUSION: MT5 is working correctly!")
        print("🔧 The issue is in main.py import logic")
    else:
        print("\n❌ CONCLUSION: MT5 has connection issues")
