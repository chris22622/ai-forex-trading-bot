#!/usr/bin/env python3
"""
Emergency Reset Script for Trading Bot
Resets consecutive losses and resumes trading immediately
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import global_bot_instance

async def emergency_reset():
    """Emergency reset for consecutive losses"""
    print("🚨 EMERGENCY RESET INITIATED")
    print("=" * 50)
    
    if not global_bot_instance:
        print("❌ Bot not running - please start the bot first")
        return False
    
    bot = global_bot_instance
    
    # Get current status
    old_count = bot.consecutive_losses
    current_balance = bot.current_balance
    active_trades = len(bot.active_trades)
    
    print(f"📊 CURRENT STATUS:")
    print(f"   💰 Balance: ${current_balance:.2f}")
    print(f"   📈 Active Trades: {active_trades}")
    print(f"   📉 Consecutive Losses: {old_count}")
    print()
    
    # Reset consecutive losses
    bot.consecutive_losses = 0
    
    print(f"✅ RESET COMPLETE:")
    print(f"   📉 Consecutive Losses: {old_count} → 0")
    print(f"   🎯 New Limit: 25 losses")
    print(f"   🚀 Status: READY TO TRADE")
    print()
    
    # Check if bot can trade now
    if bot._check_risk_limits():
        print("🟢 RISK LIMITS: PASSED ✅")
        print("🚀 BOT IS NOW READY TO TRADE!")
        print("💡 Watching for 61% confidence AI signals...")
    else:
        print("🔴 RISK LIMITS: STILL BLOCKED ❌")
        print("⚠️ Check other risk factors (balance, daily loss)")
    
    print("=" * 50)
    return True

def main():
    """Main entry point"""
    try:
        # Run the emergency reset
        result = asyncio.run(emergency_reset())
        
        if result:
            print("🎉 Emergency reset successful!")
            print("🔥 Your bot should now resume trading with 61% confidence signals!")
        else:
            print("❌ Emergency reset failed!")
            
    except Exception as e:
        print(f"❌ Error during emergency reset: {e}")

if __name__ == "__main__":
    main()
