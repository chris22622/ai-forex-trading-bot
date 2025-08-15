#!/usr/bin/env python3
"""
Simple MT5 Symbol Test - Check what symbols are actually available
"""

import MetaTrader5 as mt5
import asyncio

async def test_mt5_symbols():
    """Test what symbols are actually available in MT5"""
    print("🔧 Testing MT5 symbol availability...")
    
    try:
        # Initialize MT5
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return
        
        print("✅ MT5 initialized successfully")
        
        # Test various symbol names for Deriv
        test_symbols = [
            "Volatility 75 Index",
            "CRASH75",
            "BOOM75", 
            "R_75",
            "CR75",
            "VOL75",
            "Volatility75",
            "volatility_75_index",
            "CR_75",
            "Crash 75 Index",
            "Boom 75 Index"
        ]
        
        print("\n📊 Testing symbol availability:")
        print("=" * 40)
        
        available_symbols = []
        
        for symbol in test_symbols:
            # Try to select symbol
            selected = mt5.symbol_select(symbol, True)
            
            if selected:
                # Try to get symbol info
                info = mt5.symbol_info(symbol)
                tick = mt5.symbol_info_tick(symbol)
                
                if info and tick:
                    print(f"✅ {symbol}: Available - Price: {tick.bid}")
                    available_symbols.append(symbol)
                else:
                    print(f"⚠️ {symbol}: Selected but no data")
            else:
                print(f"❌ {symbol}: Not available")
        
        print(f"\n🎯 Found {len(available_symbols)} working symbols:")
        for symbol in available_symbols:
            print(f"  - {symbol}")
        
        # Try to get all available symbols
        print("\n🔍 Getting all available symbols...")
        try:
            symbols = mt5.symbols_get()
            if symbols:
                print(f"📊 Total symbols available: {len(symbols)}")
                
                # Look for volatility/crash/boom symbols
                relevant_symbols = []
                for sym in symbols:
                    name = sym.name.lower()
                    if any(keyword in name for keyword in ['volatility', 'crash', 'boom', 'vol', 'cr']):
                        relevant_symbols.append(sym.name)
                
                if relevant_symbols:
                    print(f"🎯 Relevant symbols found ({len(relevant_symbols)}):")
                    for sym in relevant_symbols[:10]:  # Show first 10
                        print(f"  - {sym}")
                else:
                    print("⚠️ No volatility/crash/boom symbols found")
                    
                    # Show first 10 symbols for reference
                    print("📋 First 10 available symbols:")
                    for sym in symbols[:10]:
                        print(f"  - {sym.name}")
            else:
                print("❌ No symbols returned")
                
        except Exception as e:
            print(f"❌ Error getting symbols: {e}")
        
        mt5.shutdown()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mt5_symbols())
