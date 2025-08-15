#!/usr/bin/env python3
"""
Kill Bot Processes - Clean up any running bot instances
"""
import psutil
import sys
import time

def kill_bot_processes():
    """Kill any running bot processes"""
    killed_count = 0
    
    print("🔍 Searching for running bot processes...")
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a Python process running our bot
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('main.py' in str(cmd) for cmd in cmdline):
                    print(f"🎯 Found bot process: PID {proc.info['pid']}")
                    print(f"   Command: {' '.join(cmdline) if cmdline else 'N/A'}")
                    
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
                        print(f"✅ Terminated process {proc.info['pid']}")
                        killed_count += 1
                    except psutil.TimeoutExpired:
                        print(f"⚠️ Process {proc.info['pid']} didn't terminate gracefully, forcing kill...")
                        proc.kill()
                        killed_count += 1
                    except psutil.AccessDenied:
                        print(f"❌ Access denied for process {proc.info['pid']}")
                    except psutil.NoSuchProcess:
                        print(f"⚠️ Process {proc.info['pid']} already terminated")
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e:
            print(f"❌ Error checking process: {e}")
    
    if killed_count > 0:
        print(f"✅ Killed {killed_count} bot process(es)")
        print("⏳ Waiting 3 seconds for cleanup...")
        time.sleep(3)
    else:
        print("✅ No running bot processes found")
    
    return killed_count

if __name__ == "__main__":
    print("🛑 Bot Process Killer")
    print("=" * 50)
    
    killed = kill_bot_processes()
    
    if killed > 0:
        print("\n🔄 All bot processes terminated. You can now start a fresh instance.")
    else:
        print("\n✅ System is clean. Ready to start bot.")
    
    print("\n💡 To start the bot, run: python main.py")
