#!/usr/bin/env python3
"""
Quick restart script to apply MT5 fixes
"""

import os
import subprocess
import time


def restart_bot():
    print("🔄 Restarting bot to apply MT5 connection fixes...")

    # Kill any existing Python processes
    try:
        subprocess.run(["taskkill", "/F", "/IM", "python.exe"],
                      capture_output=True, text=True)
        print("✅ Stopped existing bot processes")
    except:
        print("⚠️ No existing processes to stop")

    # Wait a moment
    time.sleep(3)

    # Start the bot again
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        python_exe = os.path.join(script_dir, ".venv", "Scripts", "python.exe")
        main_py = os.path.join(script_dir, "main.py")

        print(f"🚀 Starting bot with: {python_exe} {main_py}")

        # Start in new process
        subprocess.Popen([python_exe, main_py],
                        cwd=script_dir,
                        creationflags=subprocess.CREATE_NEW_CONSOLE)

        print("✅ Bot restarted with MT5 connection monitoring and enhanced order handling!")

    except Exception as e:
        print(f"❌ Failed to restart bot: {e}")

if __name__ == "__main__":
    restart_bot()
