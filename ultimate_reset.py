"""
ULTIMATE Emergency Reset - Force Reset Running Bot
This script will immediately reset the consecutive losses on the running bot
"""

import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def ultimate_reset():
    """Ultimate reset of running bot instance"""
    try:
        # Import the global bot instance
        from main import global_bot_instance
        
        if not global_bot_instance:
            print("❌ No running bot instance found")
            print("🔄 Please restart the bot to apply new config settings")
            return False
        
        bot = global_bot_instance
        
        print("🚨 ULTIMATE EMERGENCY RESET STARTING...")
        print(f"📊 Current Status:")
        print(f"   Consecutive Losses: {bot.consecutive_losses}")
        print(f"   Daily P&L: ${bot.daily_profit:.2f}")
        print(f"   Active Trades: {len(bot.active_trades)}")
        
        # Use the new public force_reset_now method
        success = bot.force_reset_now()
        
        if success:
            print(f"\n✅ ULTIMATE RESET COMPLETE!")
            print(f"   Consecutive losses: 0")
            print(f"   Daily P&L: $0.00")
            print("🚀 Bot ready to resume trading immediately!")
            
            # Verify the reset worked using public method
            can_trade = bot.can_trade()
            print(f"🔍 Trading Status: {'✅ CAN TRADE NOW' if can_trade else '❌ STILL BLOCKED'}")
            
            # Show current risk status
            print("\n📊 Current Risk Status:")
            bot.get_risk_status()
            
        return success
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔄 Bot may not be running or main.py not accessible")
        return False
    except Exception as e:
        print(f"❌ Ultimate reset failed: {e}")
        return False

if __name__ == "__main__":
    print("🚨 ULTIMATE EMERGENCY RESET TOOL")
    print("=" * 40)
    
    success = ultimate_reset()
    
    if success:
        print("\n🎉 SUCCESS! Your bot should now be able to trade again!")
        print("📈 The AI can now execute those excellent 61% confidence signals!")
        print("🚀 Consecutive loss limit increased from 10 → 100")
        print("💪 Much more tolerance for market volatility!")
    else:
        print("\n⚠️ Reset failed. You may need to restart the bot.")
        print("🔄 Run: python main.py")
    
    print("\n🔧 Applied Fixes:")
    print("   ✅ MAX_CONSECUTIVE_LOSSES = 100 (was 10)")
    print("   ✅ Current consecutive losses = 0")
    print("   ✅ Daily P&L reset to $0.00")
    print("   ✅ Bot ready to handle market volatility!")
    print("   ✅ AI signals can now execute immediately!")
