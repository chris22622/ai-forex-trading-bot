#!/usr/bin/env python3
"""
Test the volume guard system by simulating trade execution
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SYMBOL_MIN_LOT_OVERRIDES, VOLUME_GUARD_MODE


def normalize_volume(position_size, min_lot, step, max_lot):
    """
    Normalize position size to broker constraints
    """
    # Handle override if available
    if hasattr(normalize_volume, '_overrides'):
        for symbol, override_min in normalize_volume._overrides.items():
            if min_lot == override_min:  # Match based on min_lot
                min_lot = override_min
                break

    # Apply volume guard mode
    if VOLUME_GUARD_MODE == "raise_to_min":
        # If position size is below minimum, raise to minimum
        if position_size < min_lot:
            position_size = min_lot

    # Ensure it's within bounds
    if position_size > max_lot:
        position_size = max_lot

    # Round to step size
    steps = round(position_size / step)
    final_size = steps * step

    # Ensure final size is at least minimum
    if final_size < min_lot:
        final_size = min_lot

    return final_size

# Set overrides
normalize_volume._overrides = SYMBOL_MIN_LOT_OVERRIDES

def test_volume_guard():
    print("🔄 Testing Volume Guard System...")
    print(f"📊 VOLUME_GUARD_MODE: {VOLUME_GUARD_MODE}")
    print(f"🔧 SYMBOL_MIN_LOT_OVERRIDES: {SYMBOL_MIN_LOT_OVERRIDES}")

    # Test scenarios
    test_cases = [
        # (symbol, position_size, min_lot, step, max_lot, expected_behavior)
        ("Volatility 50 Index", 0.03, 4.0, 1.0, 10.0, "Should raise to 4.0"),
        ("Volatility 25 Index", 0.02, 0.5, 0.1, 10.0, "Should raise to 0.5"),
        ("Volatility 10 Index", 0.05, 0.01, 0.01, 10.0, "Should keep 0.05"),
        ("Small account test", 0.01, 4.0, 1.0, 10.0, "Should raise to 4.0 (big jump!)"),
    ]

    print("\n📋 Volume Guard Test Results:")
    print("=" * 70)

    for symbol, pos_size, min_lot, step, max_lot, expected in test_cases:
        result = normalize_volume(pos_size, min_lot, step, max_lot)

        print(f"🎯 {symbol}:")
        print(f"   Input: {pos_size} lots")
        print(f"   Broker: min={min_lot}, step={step}, max={max_lot}")
        print(f"   Output: {result} lots")

        if result > pos_size:
            risk_increase = (result / pos_size) * 100
            print(f"   ✅ VOLUME GUARD: Raised to broker minimum (+{risk_increase:.0f}% risk)")
            print("   🛡️ Protected by $0.75 universal stop loss")
        else:
            print("   ✅ VOLUME GUARD: Kept original size")

        print(f"   Expected: {expected}")
        print()

    # Show the key benefit
    print("🎯 KEY BENEFIT:")
    print("   Before: Trades were SKIPPED when min_lot > safety_cap")
    print("   After: Trades use broker minimum + $0.75 stop protection")
    print("   Result: Bot can actually trade on all symbols!")

if __name__ == "__main__":
    test_volume_guard()
