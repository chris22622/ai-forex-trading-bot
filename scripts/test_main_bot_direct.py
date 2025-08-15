#!/usr/bin/env python3
"""
Test if main bot can place a trade by manually calling place_mt5_trade
"""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_main_bot_direct():
    """Test main bot place_mt5_trade method directly"""
    print("🔥 TESTING MAIN BOT PLACE_MT5_TRADE METHOD")
    print("=" * 50)
    
    try:
        # Import main bot
        from main import DerivTradingBot
        
        # Create bot instance
        bot = DerivTradingBot()
        
        # Initialize MT5 interface
        if bot.mt5_interface:
            success = await bot.mt5_interface.initialize()
            if success:
                print("✅ MT5 Interface initialized successfully")
                
                # Try to call the place_mt5_trade method directly
                print("📈 Calling place_mt5_trade directly...")
                result = await bot.place_mt5_trade("BUY")
                
                if result:
                    print("✅ MAIN BOT TRADE SUCCESSFUL!")
                    print(f"📊 Result: {result}")
                else:
                    print("❌ Main bot trade failed")
            else:
                print("❌ MT5 initialization failed")
        else:
            print("❌ No MT5 interface available")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_main_bot_direct())
