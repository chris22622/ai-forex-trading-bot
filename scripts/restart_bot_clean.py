#!/usr/bin/env python3
"""
Clean Bot Restart Script
Kills all bot processes and starts fresh
"""

import os
import subprocess
import time


def kill_all_python_processes():
    """Kill all Python processes"""
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe'],
                      capture_output=True, text=True)
        time.sleep(2)
    except Exception as e:
        print(f"Error killing processes: {e}")

def start_bot():
    """Start the bot fresh"""
    try:
        # Change to bot directory
        os.chdir(r'c:\Users\Chris\scalping_deriv_bot')

        # Start bot
        subprocess.Popen([
            r'.\.venv\Scripts\python.exe',
            'main.py'
        ])

        print("✅ Bot restarted successfully!")

    except Exception as e:
        print(f"❌ Error starting bot: {e}")

if __name__ == "__main__":
    print("🔄 Cleaning and restarting bot...")
    kill_all_python_processes()
    start_bot()
