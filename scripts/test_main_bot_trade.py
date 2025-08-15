#!/usr/bin/env python3
"""
Test script to verify main bot can execute trades like simple_order_test.py
"""

import asyncio
import sys
import os
import time

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TradingBot
from config import Config
from mt5_integration import MT5Interface

async def test_main_bot_trade():
    """Test if main bot can execute a trade"""
    print("🔥 TESTING MAIN BOT TRADE EXECUTION")
    print("=" * 50)
    
    # Create config
    config = Config()
    
    # Create bot instance
    bot = TradingBot()
    
    try:
        # Initialize the bot's systems
        print("📋 Initializing bot systems...")
        await bot.setup()
        
        # Give it a moment to settle
        await asyncio.sleep(2)
        
        # Manual trade trigger - Buy signal
        print("📈 TRIGGERING BUY SIGNAL")
        result = await bot.place_mt5_trade("BUY", 0.5, "Manual test trade")
        
        if result:
            print("✅ TRADE EXECUTED SUCCESSFULLY!")
            print(f"📊 Trade result: {result}")
        else:
            print("❌ TRADE FAILED!")
            
        # Wait a moment then try a sell signal
        await asyncio.sleep(3)
        
        print("📉 TRIGGERING SELL SIGNAL") 
        result = await bot.place_mt5_trade("SELL", 0.5, "Manual test trade 2")
        
        if result:
            print("✅ SECOND TRADE EXECUTED SUCCESSFULLY!")
            print(f"📊 Trade result: {result}")
        else:
            print("❌ SECOND TRADE FAILED!")
            
    except Exception as e:
        print(f"❌ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            await bot.cleanup()
        except:
            pass
    
    print("🎯 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_main_bot_trade())
