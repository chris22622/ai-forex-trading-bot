#!/usr/bin/env python3
"""
Emergency Trading Startup Script
Forces the bot to start trading immediately by bypassing the slow data collection phase
"""

import asyncio
import sys
import os
from typing import Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def force_immediate_trading():
    """Force the bot to start trading immediately"""
    try:
        # Import the global bot instance
        from main import global_bot_instance
        
        if global_bot_instance is None:
            print("❌ ERROR: No bot instance found!")
            print("💡 Solution: Start the bot first with: python main.py")
            return False
        
        bot = global_bot_instance
        print("🤖 Found active bot instance!")
        
        # Check current status
        print(f"\n📊 Current Status:")
        print(f"   Running: {bot.running}")
        print(f"   MT5 Connected: {bot.mt5_connected}")
        print(f"   Price History: {len(bot.price_history)} points")
        print(f"   Active Symbols: {len(bot.active_symbols)} ({bot.active_symbols})")
        print(f"   Consecutive Losses: {bot.consecutive_losses}")
        print(f"   Symbol Histories: {len(bot.symbol_price_histories)} symbols")
        
        # Show individual symbol data
        for symbol in bot.active_symbols:
            symbol_history = bot.symbol_price_histories.get(symbol, [])
            print(f"   {symbol}: {len(symbol_history)} price points")
        
        print("\n🚨 APPLYING EMERGENCY FIXES...")
        
        # 1. Force start trading function
        print("🔧 Step 1: Force populating price data...")
        result = bot.force_start_trading_now()
        
        if 'error' in result:
            print(f"❌ Force start failed: {result['error']}")
            return False
        
        print("✅ Force start completed!")
        print(f"   Symbol: {result.get('symbol', 'N/A')}")
        print(f"   Price Points: {result.get('price_points', 0)}")
        print(f"   Ready to Trade: {result.get('ready_to_trade', False)}")
        
        # 2. Reset all blocking limits
        print("\n🔧 Step 2: Resetting all limits...")
        reset_success = bot.emergency_reset_all_limits()
        if reset_success:
            print("✅ All limits reset!")
        else:
            print("⚠️ Limit reset had issues")
        
        # 3. Apply auto-fix for any remaining issues
        print("\n🔧 Step 3: Auto-fixing any blocking limits...")
        bot.check_and_auto_fix_limits()
        print("✅ Auto-fix applied!")
        
        # 4. Force a manual trading check
        print("\n🔧 Step 4: Testing AI prediction system...")
        try:
            best_symbol = "Volatility 50 Index"
            test_price = 125.05
            
            # Force check trading opportunity
            await bot._check_trading_opportunity_for_symbol(test_price, best_symbol)
            print("✅ Trading system test completed!")
            
        except Exception as test_error:
            print(f"⚠️ Trading system test error: {test_error}")
        
        print("\n🚀 EMERGENCY STARTUP COMPLETE!")
        print("=" * 60)
        print("📈 Bot should start placing trades in the next 15-30 seconds!")
        print("📱 Watch for AI prediction logs and Telegram notifications")
        print("🎯 Monitor the main terminal for trading activity")
        print("=" * 60)
        
        return True
        
    except ImportError as import_error:
        print(f"❌ Import Error: {import_error}")
        print("💡 Make sure main.py is running in another terminal")
        return False
    except Exception as e:
        print(f"❌ Emergency startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def monitor_bot_status():
    """Monitor bot status for a few minutes to see if trading starts"""
    try:
        from main import global_bot_instance
        
        if not global_bot_instance:
            print("❌ Bot not found for monitoring")
            return
        
        bot = global_bot_instance
        print("\n🔍 MONITORING BOT STATUS...")
        print("Press Ctrl+C to stop monitoring")
        
        for i in range(60):  # Monitor for 1 minute
            try:
                # Check if any AI predictions are being made
                if hasattr(bot, '_loop_counter'):
                    loop_count = bot._loop_counter
                else:
                    loop_count = "Unknown"
                
                # Check symbol data
                symbol_status = []
                for symbol in bot.active_symbols:
                    history_len = len(bot.symbol_price_histories.get(symbol, []))
                    symbol_status.append(f"{symbol}:{history_len}pts")
                
                status_line = f"Loop:{loop_count} | Symbols:[{', '.join(symbol_status)}] | Active trades: {len(bot.active_trades)}"
                print(f"\r🔄 {status_line}", end="", flush=True)
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\n\n👋 Monitoring stopped by user")
                break
            except Exception as monitor_error:
                print(f"\n⚠️ Monitoring error: {monitor_error}")
                break
        
        print(f"\n\n📊 Final Status:")
        print(f"   Active Trades: {len(bot.active_trades)}")
        print(f"   Trades Today: {bot.trades_today}")
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")

def main():
    """Main emergency startup function"""
    print("🚨 EMERGENCY TRADING STARTUP")
    print("=" * 60)
    print("This script forces the bot to start trading immediately")
    print("by bypassing the slow price data collection phase.")
    print("=" * 60)
    
    # Run the emergency startup
    success = asyncio.run(force_immediate_trading())
    
    if success:
        print("\n✅ SUCCESS: Bot forced to start trading!")
        
        # Ask if user wants to monitor
        try:
            monitor = input("\n❓ Monitor bot status for 1 minute? (y/n): ").lower().strip()
            if monitor in ['y', 'yes', '']:
                asyncio.run(monitor_bot_status())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print("\n❌ FAILED: Could not force start trading")
        print("💡 Make sure the main bot is running with: python main.py")
    
    print("\n🔚 Emergency startup script completed.")

if __name__ == "__main__":
    main()
