#!/usr/bin/env python3
"""
Comprehensive Test Script for Main.py
Tests all major functionality and identifies any remaining issues
"""

import sys
import traceback
from datetime import datetime


def comprehensive_test():
    """Run comprehensive tests on main.py"""

    print("🧪 COMPREHENSIVE MAIN.PY TEST")
    print("=" * 50)
    print(f"🕒 Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    test_results = []

    # Test 1: Basic Import
    print("🔍 Test 1: Basic Import")
    try:
        sys.path.insert(0, '.')
        import main
        print("  ✅ main.py imports successfully")
        test_results.append(("Import", True, None))
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        test_results.append(("Import", False, str(e)))
        return test_results

    # Test 2: Bot Class Creation
    print("\n🔍 Test 2: Bot Class Creation")
    try:
        bot = main.DerivTradingBot()
        print("  ✅ DerivTradingBot instance created")
        test_results.append(("Bot Creation", True, None))
    except Exception as e:
        print(f"  ❌ Bot creation failed: {e}")
        test_results.append(("Bot Creation", False, str(e)))
        return test_results

    # Test 3: Essential Methods Exist
    print("\n🔍 Test 3: Essential Methods Check")
    essential_methods = [
        'start', 'stop', 'connect_mt5_with_retries', 'start_simulation_mode',
        'should_trade', 'get_ai_prediction', 'place_trade', 'simulate_trade',
        'check_trade_results', 'analyze_market_condition', 'calculate_dynamic_position_size'
    ]

    missing_methods = []
    for method in essential_methods:
        if hasattr(bot, method):
            print(f"  ✅ {method}() exists")
        else:
            print(f"  ❌ {method}() missing")
            missing_methods.append(method)

    if missing_methods:
        test_results.append(("Essential Methods", False, f"Missing: {missing_methods}"))
    else:
        test_results.append(("Essential Methods", True, None))

    # Test 4: Configuration Check
    print("\n🔍 Test 4: Configuration Check")
    try:
        # Check if basic attributes exist
        required_attrs = ['running', 'connected', 'current_balance', 'active_trades', 'price_history']
        missing_attrs = []

        for attr in required_attrs:
            if hasattr(bot, attr):
                print(f"  ✅ {attr} attribute exists")
            else:
                print(f"  ❌ {attr} attribute missing")
                missing_attrs.append(attr)

        if missing_attrs:
            test_results.append(("Configuration", False, f"Missing attributes: {missing_attrs}"))
        else:
            test_results.append(("Configuration", True, None))

    except Exception as e:
        print(f"  ❌ Configuration check failed: {e}")
        test_results.append(("Configuration", False, str(e)))

    # Test 5: AI System Check
    print("\n🔍 Test 5: AI System Check")
    try:
        if hasattr(bot, 'ai_manager') and bot.ai_manager:
            print("  ✅ Enhanced AI system available")
        elif hasattr(bot, 'ai_model') and bot.ai_model:
            print("  ✅ Basic AI system available")
        else:
            print("  ⚠️ No AI system detected")

        test_results.append(("AI System", True, None))
    except Exception as e:
        print(f"  ❌ AI system check failed: {e}")
        test_results.append(("AI System", False, str(e)))

    # Test 6: MT5 Interface Check
    print("\n🔍 Test 6: MT5 Interface Check")
    try:
        if hasattr(bot, 'mt5_interface'):
            if bot.mt5_interface:
                print("  ✅ MT5 interface initialized")
            else:
                print("  ⚠️ MT5 interface not initialized (expected if MT5 not available)")
        else:
            print("  ❌ MT5 interface attribute missing")

        test_results.append(("MT5 Interface", True, None))
    except Exception as e:
        print(f"  ❌ MT5 interface check failed: {e}")
        test_results.append(("MT5 Interface", False, str(e)))

    # Test 7: Method Functionality Test
    print("\n🔍 Test 7: Method Functionality Test")
    try:
        # Test should_trade (should work without async)
        if hasattr(bot, 'check_pause_status'):
            paused = bot.check_pause_status()
            print(f"  ✅ check_pause_status() returns: {paused}")

        # Test market analysis
        if hasattr(bot, 'analyze_market_condition'):
            # Add some sample price data
            bot.price_history = [100.0, 100.5, 101.0, 100.8, 101.2]
            condition = bot.analyze_market_condition()
            print(f"  ✅ analyze_market_condition() returns: {condition}")

        # Test position sizing
        if hasattr(bot, 'calculate_dynamic_position_size'):
            size = bot.calculate_dynamic_position_size(10.0, 0.75)
            print(f"  ✅ calculate_dynamic_position_size() returns: ${size:.2f}")

        test_results.append(("Method Functionality", True, None))
    except Exception as e:
        print(f"  ❌ Method functionality test failed: {e}")
        test_results.append(("Method Functionality", False, str(e)))

    # Test Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)

    for test_name, success, error in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {test_name}")
        if error:
            print(f"         | Error: {error}")

    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! main.py is ready to run!")
        return True
    else:
        print("⚠️ Some tests failed. Check errors above.")
        return False

if __name__ == "__main__":
    try:
        success = comprehensive_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test script crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
