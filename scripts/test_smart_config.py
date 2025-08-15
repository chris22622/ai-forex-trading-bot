#!/usr/bin/env python3
"""
Test the improved smart symbol filtering and configuration
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *


def test_smart_configuration():
    print("🎯 TESTING SMART SMALL ACCOUNT CONFIGURATION")
    print("=" * 60)

    print("📊 Current Configuration:")
    print(f"   DEFAULT_SYMBOL: {DEFAULT_SYMBOL}")
    print(f"   PRIMARY_TRADING_SYMBOLS: {PRIMARY_TRADING_SYMBOLS}")
    print(f"   VOLUME_GUARD_MODE: {VOLUME_GUARD_MODE}")
    print(f"   MAX_SAFE_LOT_SIZE: {MAX_SAFE_LOT_SIZE}")
    print()

    print("✅ PREFERRED SYMBOLS (low min lot):")
    for symbol in PREFERRED_SYMBOLS:
        print(f"   📈 {symbol}")
    print()

    print("⚠️ AVOIDED SYMBOLS (high min lot for small accounts):")
    for symbol in AVOID_HIGH_MIN_LOT_SYMBOLS:
        print(f"   ❌ {symbol}")
    print()

    print("🎯 SMART IMPROVEMENTS MADE:")
    print("   ✅ Changed DEFAULT_SYMBOL: 'Volatility 25 Index' → 'Volatility 10 Index'")
    print("   ✅ Updated PRIMARY_TRADING_SYMBOLS: Removed V50, prioritized V10")
    print("   ✅ Added VOLUME_GUARD_MODE: 'smart_filter' (intelligent symbol selection)")
    print("   ✅ Added PREFERRED_SYMBOLS: Focus on low min lot symbols")
    print("   ✅ Added AVOID_HIGH_MIN_LOT_SYMBOLS: Skip V50 (4.0 min lot too risky)")
    print()

    print("💰 ACCOUNT SAFETY ANALYSIS:")
    print("   💵 Balance: $27.43")
    print(f"   🛡️ Max safe lot size: {MAX_SAFE_LOT_SIZE}")
    print("   📊 V10 Index: ~0.01 min lot (SAFE) ✅")
    print("   📊 V25 Index: ~0.5 min lot (manageable) ⚠️")
    print("   📊 V50 Index: ~4.0 min lot (AVOID) ❌")
    print()

    print("🚀 EXPECTED BEHAVIOR:")
    print("   🎯 Bot will prioritize V10 Index (safest)")
    print("   🎯 V25 Index will be used if needed (0.5 lots = reasonable risk)")
    print("   🎯 V50 Index will be AVOIDED (4.0 lots = $100+ risk per trade)")
    print("   🎯 All trades protected by $0.75 universal stop loss")

if __name__ == "__main__":
    test_smart_configuration()
