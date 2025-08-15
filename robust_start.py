#!/usr/bin/env python3
"""
Robust Startup Script for Deriv Trading Bot
Handles MT5 connection issues with automatic retry and fallback options
"""

import asyncio
import sys
import time
import traceback
from datetime import datetime

# Add current directory to path
sys.path.insert(0, '.')

async def robust_bot_startup():
    """Start the bot with robust error handling and retry logic"""
    
    print("🚀 ROBUST TRADING BOT STARTUP")
    print("=" * 50)
    print(f"🕒 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    max_startup_attempts = 5
    current_attempt = 0
    
    while current_attempt < max_startup_attempts:
        current_attempt += 1
        
        try:
            print(f"🔄 Startup attempt {current_attempt}/{max_startup_attempts}")
            
            # Import the main bot module
            print("📦 Importing bot modules...")
            from main import DerivTradingBot, logger
            
            # Create bot instance
            print("🤖 Creating bot instance...")
            bot = DerivTradingBot()
            
            # Apply demo mode settings
            bot.force_demo_mode = True
            bot.dry_run_mode = True
            
            print("✅ Bot instance created successfully")
            print()
            
            # Attempt to start the bot
            print("🚀 Starting trading bot...")
            await bot.start()
            
            # If we get here, the bot started successfully
            print("✅ Bot startup completed successfully!")
            return
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("🔍 Check if all required modules are installed")
            break
            
        except Exception as e:
            print(f"❌ Startup attempt {current_attempt} failed: {e}")
            
            # Show the full error for debugging
            if current_attempt == 1:
                print("\n🐛 Full error details:")
                traceback.print_exc()
                print()
            
            if current_attempt < max_startup_attempts:
                wait_time = current_attempt * 10  # Progressive wait: 10s, 20s, 30s, 40s
                print(f"⏱️ Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                print("❌ All startup attempts failed")
                print()
                print("🛠️ TROUBLESHOOTING GUIDE:")
                print("1. Check MT5 Terminal is running")
                print("2. Verify MT5 login credentials")
                print("3. Check network connection")
                print("4. Try restarting MT5 Terminal")
                print("5. Check logs directory for detailed errors")
                break
    
    print()
    print("🔚 Startup process completed")

if __name__ == "__main__":
    try:
        asyncio.run(robust_bot_startup())
    except KeyboardInterrupt:
        print("\n⏹️ Startup interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
