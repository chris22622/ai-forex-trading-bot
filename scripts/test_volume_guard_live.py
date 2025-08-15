#!/usr/bin/env python3
"""
Test the volume guard system with live MT5 connection
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5

from config import *
from mt5_integration import connect_to_mt5


def test_volume_guard():
    print("🔄 Testing Volume Guard System with Live MT5...")

    # Initialize MT5
    if not connect_to_mt5():
        print("❌ Failed to connect to MT5")
        return

    # Test the normalize_volume function directly
    print("\n📊 Testing normalize_volume function:")

    # Import the function from main.py
    exec(open('main.py').read(), globals())

    # Test with actual symbol data
    test_symbols = ['Volatility 50 Index', 'Volatility 25 Index', 'Volatility 10 Index']

    for symbol in test_symbols:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            step = symbol_info.volume_step

            print(f"\n🎯 {symbol}:")
            print(f"   Broker Min: {min_lot}, Max: {max_lot}, Step: {step}")

            # Test with small position size (should trigger volume guard)
            test_size = 0.02
            normalized = normalize_volume(test_size, min_lot, step, max_lot)
            print(f"   Input: {test_size} → Output: {normalized}")

            # Show volume guard behavior
            if normalized > test_size:
                print(f"   ✅ VOLUME GUARD: Raised {test_size} to broker minimum {normalized}")
            else:
                print(f"   ✅ VOLUME GUARD: Kept original size {normalized}")
        else:
            print(f"❌ Symbol {symbol} not found")

    # Test with SYMBOL_MIN_LOT_OVERRIDES
    print(f"\n🔧 Testing SYMBOL_MIN_LOT_OVERRIDES: {SYMBOL_MIN_LOT_OVERRIDES}")
    for symbol, override_min in SYMBOL_MIN_LOT_OVERRIDES.items():
        print(f"   {symbol}: Override minimum = {override_min}")

        # Test normalize_volume with override
        test_size = 0.01
        normalized = normalize_volume(test_size, override_min, 0.1, 10.0)
        print(f"   Input: {test_size} → Output: {normalized} (using override)")

if __name__ == "__main__":
    test_volume_guard()
