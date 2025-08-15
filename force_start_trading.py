#!/usr/bin/env python3
"""
Emergency script to force-start trading immediately
Run this script to bypass the slow price collection phase
"""

import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def force_start_trading():
    """Force the bot to start trading immediately"""
    try:
        # Import the global bot instance
        from main import global_bot_instance
        
        if global_bot_instance is None:
            print("❌ No bot instance found. Start the bot first with: python main.py")
            return False
        
        bot = global_bot_instance
        print("🤖 Found active bot instance")
        
        # Check current status
        print(f"📊 Current Status:")
        print(f"   Running: {bot.running}")
        print(f"   MT5 Connected: {bot.mt5_connected}")
        print(f"   Price History: {len(bot.price_history)} points")
        print(f"   Active Symbols: {len(bot.active_symbols)}")
        print(f"   Consecutive Losses: {bot.consecutive_losses}")
        
        # Apply emergency fixes
        print("\n🚨 APPLYING EMERGENCY FIXES...")
        
        # 1. Force start trading now
        result = bot.force_start_trading_now()
        if 'error' in result:
            print(f"❌ Force start failed: {result['error']}")
            return False
        
        print("✅ Force start completed!")
        print(f"   Symbol: {result['symbol']}")
        print(f"   Price Points: {result['price_points']}")
        print(f"   Ready to Trade: {result['ready_to_trade']}")
        
        # 2. Reset any blocking limits
        bot.emergency_reset_all_limits()
        print("✅ All limits reset!")
        
        # 3. Force enable trading flags
        if hasattr(bot, '_trading_paused'):
            bot._trading_paused = False
        
        print("\n🚀 EMERGENCY STARTUP COMPLETE!")
        print("📈 Bot should start placing trades in the next 30 seconds!")
        print("📱 Watch for trading notifications on Telegram")
        
        return True
        
    except ImportError:
        print("❌ Could not import bot instance. Make sure main.py is running.")
        return False
    except Exception as e:
        print(f"❌ Emergency startup failed: {e}")
        return False

if __name__ == "__main__":
    print("🚨 EMERGENCY TRADING STARTUP")
    print("=" * 50)
    
    success = force_start_trading()
    
    if success:
        print("\n✅ SUCCESS: Bot forced to start trading!")
        print("💡 Monitor the main terminal for trading activity")
    else:
        print("\n❌ FAILED: Could not force start trading")
        print("💡 Make sure the main bot is running first")
    
    print("\nPress Enter to exit...")
    input()
