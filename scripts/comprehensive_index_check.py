"""
Comprehensive Deriv Index Availability Checker
Checks all Deriv synthetic indices for active pricing and trading availability
"""

import MetaTrader5 as mt5
from datetime import datetime
import time

def comprehensive_index_check():
    """Check all available Deriv synthetic indices"""
    
    # Complete list of Deriv synthetic indices
    all_deriv_indices = [
        # Volatility Indices (Standard)
        "Volatility 10 Index",
        "Volatility 25 Index", 
        "Volatility 50 Index",
        "Volatility 75 Index",
        "Volatility 100 Index",
        
        # Volatility Indices (1-second)
        "Volatility 10 (1s) Index",
        "Volatility 15 (1s) Index",
        "Volatility 25 (1s) Index",
        "Volatility 30 (1s) Index",
        "Volatility 50 (1s) Index", 
        "Volatility 75 (1s) Index",
        "Volatility 90 (1s) Index",
        "Volatility 100 (1s) Index",
        "Volatility 150 (1s) Index",
        "Volatility 250 (1s) Index",
        
        # Boom and Crash Indices
        "Boom 300 Index",
        "Boom 500 Index",
        "Boom 600 Index",
        "Boom 900 Index",
        "Boom 1000 Index",
        "Crash 300 Index",
        "Crash 500 Index", 
        "Crash 900 Index",
        "Crash 1000 Index",
        
        # Step and Jump Indices
        "Step Index",
        "Jump 10 Index",
        "Jump 25 Index",
        "Jump 50 Index",
        "Jump 75 Index",
        "Jump 100 Index",
        
        # Volatility over Boom/Crash
        "Vol over Boom 400",
        "Vol over Boom 750",
        "Vol over Crash 400",
        "Vol over Crash 550",
        "Vol over Crash 750",
        
        # Range Break Indices
        "Range Break 100 Index",
        "Range Break 200 Index",
    ]
    
    print("=" * 80)
    print("🔍 COMPREHENSIVE DERIV SYNTHETIC INDICES STATUS CHECK")
    print("=" * 80)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    # Login
    login_result = mt5.login(62233085, password="Goon22622$", server="DerivVU-Server")
    if not login_result:
        print(f"❌ Login failed: {mt5.last_error()}")
        mt5.shutdown()
        return
    
    print("✅ MT5 connected successfully")
    print(f"📊 Account: 62233085, Balance: ${mt5.account_info().balance:.2f}")
    print("-" * 80)
    
    active_indices = []
    inactive_indices = []
    error_indices = []
    
    for i, symbol in enumerate(all_deriv_indices, 1):
        try:
            print(f"[{i:2d}/{len(all_deriv_indices)}] Testing: {symbol:<35}", end=" ")
            
            # Check if symbol exists
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print("❌ NOT FOUND")
                error_indices.append(symbol)
                continue
            
            # Try to select symbol
            if not mt5.symbol_select(symbol, True):
                print("❌ CANNOT SELECT")
                error_indices.append(symbol)
                continue
            
            # Wait for symbol to load
            time.sleep(0.2)
            
            # Get tick data
            tick = mt5.symbol_info_tick(symbol)
            
            if tick and tick.bid > 0 and tick.ask > 0:
                spread = tick.ask - tick.bid
                print(f"✅ ACTIVE   Bid: {tick.bid:>10.2f} Ask: {tick.ask:>10.2f} Spread: {spread:>6.2f}")
                active_indices.append({
                    'symbol': symbol,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'spread': spread,
                    'time': datetime.fromtimestamp(tick.time)
                })
            else:
                print("❌ NO PRICES")
                inactive_indices.append(symbol)
                
        except Exception as e:
            print(f"⚠️ ERROR: {str(e)[:30]}")
            error_indices.append(symbol)
    
    # Results Summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ ACTIVE INDICES ({len(active_indices)}):")
    if active_indices:
        # Sort by symbol type for better organization
        volatility_indices = [idx for idx in active_indices if 'Volatility' in idx['symbol']]
        boom_crash_indices = [idx for idx in active_indices if 'Boom' in idx['symbol'] or 'Crash' in idx['symbol']]
        other_indices = [idx for idx in active_indices if idx not in volatility_indices + boom_crash_indices]
        
        if volatility_indices:
            print("\n  📈 VOLATILITY INDICES:")
            for idx in volatility_indices:
                print(f"     - {idx['symbol']:<35} Bid: {idx['bid']:>10.2f} Spread: {idx['spread']:>6.2f}")
        
        if boom_crash_indices:
            print("\n  💥 BOOM/CRASH INDICES:")
            for idx in boom_crash_indices:
                print(f"     - {idx['symbol']:<35} Bid: {idx['bid']:>10.2f} Spread: {idx['spread']:>6.2f}")
        
        if other_indices:
            print("\n  🎯 OTHER INDICES:")
            for idx in other_indices:
                print(f"     - {idx['symbol']:<35} Bid: {idx['bid']:>10.2f} Spread: {idx['spread']:>6.2f}")
    
    print(f"\n❌ INACTIVE INDICES ({len(inactive_indices)}):")
    for symbol in inactive_indices:
        print(f"     - {symbol}")
    
    print(f"\n⚠️ ERROR/NOT FOUND ({len(error_indices)}):")
    for symbol in error_indices:
        print(f"     - {symbol}")
    
    # Recommendations
    print(f"\n🎯 TRADING RECOMMENDATIONS:")
    print(f"   📊 Total Active: {len(active_indices)} indices available for trading")
    
    if len(active_indices) >= 5:
        print(f"   ✅ Excellent coverage - Multiple indices available")
        print(f"   💡 Recommended: Enable multi-symbol trading with top 3-5 indices")
    elif len(active_indices) >= 3:
        print(f"   ✅ Good coverage - Several indices available") 
        print(f"   💡 Recommended: Enable multi-symbol trading with all active indices")
    elif len(active_indices) >= 1:
        print(f"   ⚠️ Limited coverage - Few indices available")
        print(f"   💡 Recommended: Single symbol trading with most active index")
    else:
        print(f"   ❌ No active indices found - Check market hours or connection")
    
    # Top 5 recommendations by lowest spread
    if active_indices:
        top_5 = sorted(active_indices, key=lambda x: x['spread'])[:5]
        print(f"\n🏆 TOP 5 RECOMMENDED (by spread):")
        for i, idx in enumerate(top_5, 1):
            print(f"   {i}. {idx['symbol']:<35} Spread: {idx['spread']:>6.2f}")
    
    mt5.shutdown()
    return active_indices, inactive_indices, error_indices

if __name__ == "__main__":
    comprehensive_index_check()
