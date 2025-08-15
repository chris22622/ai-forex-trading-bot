#!/usr/bin/env python3
"""
🚀 AGGRESSIVE BOT RESTART FOR SMALL ACCOUNTS
Restart the bot with all aggressive settings for small account growth
"""

import subprocess
import time
import os

def restart_aggressive_bot():
    """Restart bot with aggressive small account settings"""
    
    print("🚀 AGGRESSIVE BOT RESTART FOR SMALL ACCOUNTS")
    print("=" * 60)
    
    # Kill any existing bot processes first
    print("🔴 Stopping any existing bot processes...")
    try:
        subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                      capture_output=True, check=False)
        time.sleep(2)
    except:
        pass
    
    print("\n📊 YOUR NEW AGGRESSIVE SETTINGS:")
    print("=" * 40)
    print("💰 Account Balance: ~$20")
    print("📈 Risk per Trade: 15% = $3.00")
    print("🎯 Profit Target: $3.00 (not $0.75!)")
    print("🛡️ Stop Loss: $4.00 (not $1.50!)")
    print("📊 Lot Size: 0.05 (not 0.01!)")
    print("⏱️ Trade Duration: 45 minutes (not 10!)")
    print("🔄 Max Concurrent: 5 trades")
    print("🚫 Profit Protection: $15 total")
    
    print("\n🔥 WHY THIS WILL WORK NOW:")
    print("=" * 40)
    print("✅ BIGGER POSITIONS = Meaningful profits")
    print("✅ LONGER TIME = Volatility can develop")
    print("✅ HIGHER TARGETS = Worth the risk")
    print("✅ MORE TRADES = More opportunities")
    print("✅ AGGRESSIVE RISK = Account can actually grow")
    
    print("\n🚀 Starting aggressive bot...")
    
    # Start the bot
    cmd = [".venv\\Scripts\\python.exe", "main.py"]
    
    print(f"🎯 Command: {' '.join(cmd)}")
    print("🔄 Starting bot process...")
    
    # Start the process in the background
    process = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    print("✅ Bot started with aggressive small account settings!")
    print("\n📱 MONITOR YOUR TELEGRAM FOR UPDATES")
    print("🎯 Expected behavior:")
    print("  - Larger position notifications ($3+ trades)")
    print("  - Higher profit targets ($3+ exits)")
    print("  - More frequent trading")
    print("  - Meaningful profit notifications")
    
    print("\n🔍 Showing first 30 seconds of output...")
    print("-" * 60)
    
    # Show initial output for 30 seconds
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            output = process.stdout.readline()
            if output:
                print(output.strip())
            elif process.poll() is not None:
                break
        except KeyboardInterrupt:
            print("\n👋 Stopping output display...")
            break
        except:
            break
    
    if process.poll() is None:
        print("\n✅ Bot is running with aggressive settings!")
        print("🔥 Your $20 account should now see REAL growth!")
        print("📱 Check Telegram for trade notifications")
        print("\n💡 To stop the bot, run: kill_bot_processes.py")
    else:
        print("\n❌ Bot process ended. Check for errors above.")
    
    return process.poll() is None

if __name__ == "__main__":
    restart_aggressive_bot()
