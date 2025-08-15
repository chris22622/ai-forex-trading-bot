#!/usr/bin/env python3
"""
Test script to verify all trading bug fixes
"""

import sys
import asyncio
from datetime import datetime

async def test_mt5_connection():
    """Test MT5 connection and price retrieval"""
    try:
        print("🔧 Testing MT5 connection...")
        
        # Test import
        try:
            from mt5_integration import MT5TradingInterface
            print("✅ MT5TradingInterface imported successfully")
        except Exception as e:
            print(f"❌ Import failed: {e}")
            return False
        
        # Test interface creation
        try:
            interface = MT5TradingInterface()
            print("✅ MT5TradingInterface created successfully")
        except Exception as e:
            print(f"❌ Interface creation failed: {e}")
            return False
        
        # Test price retrieval
        try:
            price = await interface.get_current_price("Volatility 75 Index")
            if price is None:
                print("⚠️ Price returned None (expected if MT5 not connected)")
            elif price > 1.0:
                print(f"✅ Valid price retrieved: {price}")
            else:
                print(f"❌ Invalid price: {price}")
                return False
        except Exception as e:
            print(f"❌ Price retrieval failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_bot_creation():
    """Test bot creation without crashes"""
    try:
        print("🔧 Testing bot creation...")
        
        # This should not crash
        from main import DerivTradingBot
        bot = DerivTradingBot()
        
        print("✅ Bot created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Bot creation failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🔍 TESTING ALL TRADING BUG FIXES")
    print("=" * 40)
    
    tests = [
        ("MT5 Connection", test_mt5_connection),
        ("Bot Creation", test_bot_creation),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🧪 Running {name} test...")
        try:
            result = await test_func()
            results.append((name, result))
            print(f"{'✅' if result else '❌'} {name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {name}: CRASHED - {e}")
            results.append((name, False))
    
    print("\n📊 TEST RESULTS:")
    print("=" * 20)
    for name, result in results:
        print(f"{'✅' if result else '❌'} {name}")
    
    all_passed = all(result for _, result in results)
    print(f"\n{'🎉 ALL TESTS PASSED!' if all_passed else '⚠️ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main())
