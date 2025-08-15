#!/usr/bin/env python3
"""
Direct test using the main bot's place_mt5_trade method
"""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt5_integration import MT5Interface

async def test_simplified_trade():
    """Test if the main bot's fixed trade method works"""
    print("🔥 TESTING SIMPLIFIED TRADE METHOD")
    print("=" * 50)
    
    # Create MT5 interface like main bot does
    mt5_interface = MT5Interface()
    
    try:
        # Initialize MT5 like the main bot
        if not await mt5_interface.initialize():
            print("❌ Failed to initialize MT5")
            return
            
        print("✅ MT5 initialized successfully")
        
        # Test the exact same trade method the main bot uses
        print("📈 Testing BUY trade...")
        result = await mt5_interface.place_trade(
            symbol="Volatility 10 Index",
            action="BUY", 
            volume=0.5,
            comment="Direct test trade"
        )
        
        if result:
            print("✅ BUY TRADE SUCCESSFUL!")
            print(f"📊 Trade result: {result}")
        else:
            print("❌ BUY TRADE FAILED!")
            
        # Test SELL trade
        await asyncio.sleep(2)
        print("📉 Testing SELL trade...")
        result = await mt5_interface.place_trade(
            symbol="Volatility 10 Index",
            action="SELL",
            volume=0.5, 
            comment="Direct test trade 2"
        )
        
        if result:
            print("✅ SELL TRADE SUCCESSFUL!")
            print(f"📊 Trade result: {result}")
        else:
            print("❌ SELL TRADE FAILED!")
            
    except Exception as e:
        print(f"❌ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎯 Direct test completed!")

if __name__ == "__main__":
    asyncio.run(test_simplified_trade())
