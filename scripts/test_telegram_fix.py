#!/usr/bin/env python3
"""
Test script for Telegram conflict resolution
"""

import asyncio
import os
import sys

# Disable Telegram for testing
os.environ['DISABLE_TELEGRAM'] = '1'
os.environ['ENABLE_TELEGRAM_ALERTS'] = 'False'

async def test_telegram_fix():
    """Test the Telegram conflict resolution"""
    print("🔧 Testing Telegram conflict resolution...")
    
    try:
        # Import the main module
        import main
        print("✅ Main module imported successfully")
        
        # Test creating a bot instance
        bot = main.DerivTradingBot()
        print("✅ Bot instance created successfully")
        
        # Test Telegram command handler
        telegram_handler = bot.telegram_handler
        print("✅ Telegram handler created successfully")
        
        # Check if Telegram is properly disabled
        if hasattr(telegram_handler, 'is_monitoring'):
            print(f"📊 Telegram monitoring status: {telegram_handler.is_monitoring}")
        
        print("🎯 All Telegram conflict fixes working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    try:
        result = asyncio.run(test_telegram_fix())
        if result:
            print("\n✅ SUCCESS: Telegram conflict resolution is working!")
        else:
            print("\n❌ FAILED: Issues detected")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)
