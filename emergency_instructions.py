"""
Direct emergency script to force trading
This modifies the main.py file to add a direct call for emergency startup
"""

# Simple script to add emergency startup call to main.py
import os

script_content = '''
# EMERGENCY STARTUP - Add this to the end of main.py temporarily
if __name__ == "__main__":
    import asyncio
    import time
    
    async def emergency_startup():
        """Emergency startup with immediate trading"""
        bot = TradingBot()
        
        # Set global instance
        global global_bot_instance
        global_bot_instance = bot
        
        # Start bot
        startup_task = asyncio.create_task(bot.start())
        
        # Wait for startup
        await asyncio.sleep(10)
        
        # Force immediate trading
        try:
            print("🚨 APPLYING EMERGENCY FORCE START...")
            result = bot.force_start_trading_now()
            print(f"✅ Emergency result: {result}")
            
            # Reset limits
            bot.emergency_reset_all_limits()
            print("✅ All limits reset!")
            
        except Exception as e:
            print(f"Emergency error: {e}")
        
        # Continue with normal operation
        await startup_task
    
    # Run with emergency startup
    print("🚨 EMERGENCY MODE: Starting bot with forced trading...")
    asyncio.run(emergency_startup())
'''

print("🔧 Emergency Script Created")
print("=" * 50)
print("To force immediate trading, replace the current main.py startup with:")
print(script_content)
print("=" * 50)
print("Or run the existing bot and it should start trading soon with the reduced")
print("price requirement (10 points instead of 20)!")
