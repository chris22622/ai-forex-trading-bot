#!/usr/bin/env python3
"""
Quick Test Script to Force a Demo Trade
This will bypass AI analysis and place a simple demo trade to test the MT5 connection
"""

import sys
import os
import asyncio
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from mt5_integration import MT5Integration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_force_trade():
    """Force a test trade to verify MT5 connection works"""
    print("🧪 TESTING: Force Demo Trade on MT5")
    
    try:
        # Initialize MT5
        mt5 = MT5Integration()
        await mt5.initialize()
        
        # Get available symbols
        symbols = mt5.deriv_symbols
        if not symbols:
            print("❌ No symbols available!")
            return
            
        test_symbol = symbols[0]  # Use first available symbol
        print(f"🎯 Testing with symbol: {test_symbol}")
        
        # Get current price
        price_info = await mt5.get_current_price(test_symbol)
        if not price_info:
            print(f"❌ Could not get price for {test_symbol}")
            return
            
        current_price = price_info['bid']
        print(f"💰 Current price: {current_price}")
        
        # Force a BUY trade with minimal amount
        print("🚀 FORCING BUY TRADE...")
        result = await mt5.place_trade(
            action='BUY',
            symbol=test_symbol,
            amount=0.01,  # Minimal amount
            price=current_price
        )
        
        if result and result.get('success'):
            print(f"✅ SUCCESS! Trade placed: {result}")
            
            # Check if trade is actually in MT5
            positions = await mt5.get_positions()
            if positions:
                print(f"📊 Active positions: {len(positions)}")
                for pos in positions:
                    print(f"   - {pos}")
            else:
                print("⚠️ No positions found in MT5")
                
        else:
            print(f"❌ FAILED! Trade result: {result}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_force_trade())
