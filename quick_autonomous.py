#!/usr/bin/env python3
"""
Quick Autonomous Launch - Forces the main bot into autonomous simulation mode
"""

import asyncio
import logging
import os
from datetime import datetime

# Force simulation mode settings
os.environ['FORCE_SIMULATION'] = 'true'
os.environ['AUTONOMOUS_MODE'] = 'true'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - AUTONOMOUS - %(message)s'
)

async def main():
    """Launch main bot in autonomous mode"""
    print("🤖 LAUNCHING AUTONOMOUS TRADING BOT")
    print("=" * 50)
    print("✅ Mode: Fully Autonomous Simulation")
    print("✅ Input Required: NONE")
    print("✅ Runs 24/7 automatically")
    print("=" * 50)
    
    try:
        # Import and configure the main bot
        from main import main as main_bot
        
        # Override config for autonomous operation
        import config
        config.PAPER_TRADING = True
        config.DEMO_MODE = True
        config.MT5_REAL_TRADING = False
        
        print("🚀 Starting autonomous trading...")
        await main_bot()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔄 Bot will restart automatically...")
        await asyncio.sleep(5)
        await main()  # Restart

if __name__ == "__main__":
    asyncio.run(main())
