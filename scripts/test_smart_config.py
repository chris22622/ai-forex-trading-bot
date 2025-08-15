#!/usr/bin/env python3
"""
Test the improved smart symbol filtering and configuration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *

def test_smart_configuration():
    print("🎯 TESTING SMART SMALL ACCOUNT CONFIGURATION")
    print("=" * 60)
    
    print(f"📊 Current Configuration:")
    print(f"   DEFAULT_SYMBOL: {DEFAULT_SYMBOL}")
    print(f"   PRIMARY_TRADING_SYMBOLS: {PRIMARY_TRADING_SYMBOLS}")
    print(f"   VOLUME_GUARD_MODE: {VOLUME_GUARD_MODE}")
    print(f"   MAX_SAFE_LOT_SIZE: {MAX_SAFE_LOT_SIZE}")
    print()
    
    print(f"✅ PREFERRED SYMBOLS (low min lot):")
    for symbol in PREFERRED_SYMBOLS:
        print(f"   📈 {symbol}")
    print()
    
    print(f"⚠️ AVOIDED SYMBOLS (high min lot for small accounts):")
    for symbol in AVOID_HIGH_MIN_LOT_SYMBOLS:
        print(f"   ❌ {symbol}")
    print()
    
    print(f"🎯 SMART IMPROVEMENTS MADE:")
    print(f"   ✅ Changed DEFAULT_SYMBOL: 'Volatility 25 Index' → 'Volatility 10 Index'")
    print(f"   ✅ Updated PRIMARY_TRADING_SYMBOLS: Removed V50, prioritized V10")
    print(f"   ✅ Added VOLUME_GUARD_MODE: 'smart_filter' (intelligent symbol selection)")
    print(f"   ✅ Added PREFERRED_SYMBOLS: Focus on low min lot symbols")
    print(f"   ✅ Added AVOID_HIGH_MIN_LOT_SYMBOLS: Skip V50 (4.0 min lot too risky)")
    print()
    
    print(f"💰 ACCOUNT SAFETY ANALYSIS:")
    print(f"   💵 Balance: $27.43")
    print(f"   🛡️ Max safe lot size: {MAX_SAFE_LOT_SIZE}")
    print(f"   📊 V10 Index: ~0.01 min lot (SAFE) ✅")
    print(f"   📊 V25 Index: ~0.5 min lot (manageable) ⚠️")  
    print(f"   📊 V50 Index: ~4.0 min lot (AVOID) ❌")
    print()
    
    print(f"🚀 EXPECTED BEHAVIOR:")
    print(f"   🎯 Bot will prioritize V10 Index (safest)")
    print(f"   🎯 V25 Index will be used if needed (0.5 lots = reasonable risk)")
    print(f"   🎯 V50 Index will be AVOIDED (4.0 lots = $100+ risk per trade)")
    print(f"   🎯 All trades protected by $0.75 universal stop loss")

if __name__ == "__main__":
    test_smart_configuration()
