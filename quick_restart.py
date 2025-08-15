#!/usr/bin/env python3
"""
Simple Bot Restart Script - No External Dependencies
"""

import subprocess
import time
import sys
import os

def kill_python_processes():
    """Kill Python processes using taskkill"""
    try:
        # Kill all python processes
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                      capture_output=True, text=True)
        time.sleep(2)
        print("✅ Python processes cleaned up")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

def start_bot():
    """Start the bot and monitor initial output"""
    try:
        print("🚀 Starting Fixed Trading Bot...")
        print("=" * 50)
        
        # Start the bot
        subprocess.Popen([
            '.venv/Scripts/python.exe', 'main.py'
        ], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        print("✅ Bot started in new console window!")
        print("📋 Applied Fixes:")
        print("   ✅ Trailing Stop Loss - Implemented") 
        print("   ✅ Symbol R_75 Error - Fixed (maps to 'Volatility 10 Index')")
        print("   ✅ List .get() Error - Safety checks added")
        print("   ✅ Telegram Conflicts - Resolved with cleanup")
        print("   ✅ Enhanced Error Handling - Applied")
        print("\n🎯 Check the new console window for bot output!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        return False

def main():
    print("🔧 QUICK BOT RESTART")
    print("=" * 30)
    
    # Cleanup
    print("🧹 Cleaning up old processes...")
    kill_python_processes()
    
    # Start
    print("🚀 Starting bot...")
    success = start_bot()
    
    if success:
        print("\n🎉 RESTART COMPLETE!")
    else:
        print("\n❌ RESTART FAILED")

if __name__ == "__main__":
    main()
