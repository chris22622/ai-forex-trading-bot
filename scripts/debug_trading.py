#!/usr/bin/env python3
"""
Debug Trading Status - Quick diagnostic tool
Run this while the main bot is running to see why trades aren't happening
"""

import asyncio
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the debug function from main
from main import debug_trading_status


async def run_debug():
    """Run trading status debug"""
    print("🔍 RUNNING TRADING DEBUG...")
    print("=" * 50)
    try:
        await debug_trading_status()
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
    print("=" * 50)
    print("✅ Debug complete!")

if __name__ == "__main__":
    asyncio.run(run_debug())
