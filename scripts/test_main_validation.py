#!/usr/bin/env python3
"""
Quick validation test for main.py to ensure no red errors
"""

import sys
import traceback


def test_main_validation():
    """Test that main.py can be imported and basic functionality works"""
    print("🔍 Testing main.py for red errors...")

    try:
        # Test import
        print("  ✓ Testing import...")
        import main
        print("  ✅ Import successful")

        # Test DerivTradingBot class instantiation
        print("  ✓ Testing DerivTradingBot class...")
        bot = main.DerivTradingBot()
        print("  ✅ DerivTradingBot instantiation successful")

        # Test basic methods exist
        print("  ✓ Testing method existence...")
        methods_to_test = [
            'start', 'stop', 'get_effective_trade_amount',
            'get_effective_symbol', 'check_pause_status',
            'analyze_market_condition', 'calculate_dynamic_position_size'
        ]

        for method_name in methods_to_test:
            if hasattr(bot, method_name):
                print(f"    ✅ Method '{method_name}' exists")
            else:
                print(f"    ❌ Method '{method_name}' missing")
                return False

        # Test basic attributes exist
        print("  ✓ Testing attributes...")
        attributes_to_test = [
            'running', 'connected', 'price_history',
            'session_stats', 'active_trades'
        ]

        for attr_name in attributes_to_test:
            if hasattr(bot, attr_name):
                print(f"    ✅ Attribute '{attr_name}' exists")
            else:
                print(f"    ❌ Attribute '{attr_name}' missing")
                return False

        print("\n🎉 ALL TESTS PASSED! main.py has no red errors.")
        return True

    except Exception as e:
        print(f"\n❌ ERROR FOUND: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_main_validation()
    sys.exit(0 if success else 1)
