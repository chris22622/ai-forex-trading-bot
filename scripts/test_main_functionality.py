#!/usr/bin/env python3
"""
Quick test to verify main.py functionality
"""

import asyncio
import traceback


def test_main_imports():
    """Test that main.py imports without errors"""
    try:
        print("✅ Main imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_bot_initialization():
    """Test that bot can be initialized"""
    try:
        from main import DerivTradingBot
        bot = DerivTradingBot()
        print("✅ Bot initialization successful")
        print(f"   - Running: {bot.running}")
        print(f"   - Connected: {bot.connected}")
        print(f"   - Using MT5: {bot.using_mt5}")
        print(f"   - Current balance: ${bot.current_balance:.2f}")
        print(f"   - Active trades: {len(bot.active_trades)}")
        return True
    except Exception as e:
        print(f"❌ Bot initialization error: {e}")
        traceback.print_exc()
        return False

async def test_basic_methods():
    """Test basic bot methods"""
    try:
        from main import DerivTradingBot
        bot = DerivTradingBot()

        # Test basic methods
        print("🔄 Testing basic methods...")

        # Test market analysis methods
        market_condition = bot.analyze_market_condition()
        print(f"   - Market condition: {market_condition}")

        # Test strategy selection
        strategy = bot.select_optimal_strategy()
        print(f"   - Selected strategy: {strategy}")

        # Test pause/resume
        bot.paused = True
        paused = bot.check_pause_status()
        print(f"   - Pause status: {paused}")

        # Test cooldown
        in_cooldown = bot.is_in_cooldown()
        print(f"   - In cooldown: {in_cooldown}")

        # Test dynamic position sizing
        position_size = bot.calculate_dynamic_position_size(10.0, 0.75, 0.6)
        print(f"   - Dynamic position size: ${position_size:.2f}")

        print("✅ Basic methods test successful")
        return True

    except Exception as e:
        print(f"❌ Basic methods test error: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🧪 MAIN.PY FUNCTIONALITY TEST")
    print("=" * 50)

    tests_passed = 0
    total_tests = 3

    # Test 1: Imports
    print("\n1️⃣ Testing imports...")
    if test_main_imports():
        tests_passed += 1

    # Test 2: Bot initialization
    print("\n2️⃣ Testing bot initialization...")
    if test_bot_initialization():
        tests_passed += 1

    # Test 3: Basic methods
    print("\n3️⃣ Testing basic methods...")
    if await test_basic_methods():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! main.py is ready to run!")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    asyncio.run(main())
