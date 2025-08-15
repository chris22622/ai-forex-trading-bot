#!/usr/bin/env python3
"""
Enhanced Trading Bot Launcher with CLI Support
Quick launch wrapper that preserves original functionality while adding CLI options
"""

import subprocess
import sys


def main():
    print("🚀 ENHANCED TRADING BOT LAUNCHER")
    print("=" * 50)

    # Check if any arguments were passed
    if len(sys.argv) > 1:
        # Pass all arguments to main.py
        cmd = [sys.executable, "main.py"] + sys.argv[1:]
        print(f"🔄 Running: {' '.join(cmd)}")
        print("=" * 50)

        try:
            # Run with arguments
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print("🛑 Interrupted by user")
            sys.exit(0)
    else:
        # No arguments - show interactive menu
        show_interactive_menu()

def show_interactive_menu():
    """Show interactive menu for users who don't use CLI"""

    print("📋 SELECT LAUNCH MODE:")
    print()
    print("1. 🧪 Demo Mode (Safe Testing)")
    print("2. 🔴 Live Mode (Real Trading)")
    print("3. 📝 Paper Trading (Simulation)")
    print("4. 🔍 Dry Run (Analysis Only)")
    print("5. 📊 Check Status")
    print("6. 💰 Check Balance")
    print("7. 🔧 Custom Settings")
    print("8. 📱 Show CLI Examples")
    print("9. ❌ Exit")
    print()

    try:
        choice = input("Enter your choice (1-9): ").strip()

        if choice == "1":
            # Demo mode
            cmd = [sys.executable, "main.py", "--demo", "--telegram"]
            print("🧪 Starting in Demo Mode with Telegram...")

        elif choice == "2":
            # Live mode with confirmation
            confirm = input("⚠️  Are you sure you want LIVE trading? (yes/no): ").strip().lower()
            if confirm == "yes":
                cmd = [sys.executable, "main.py", "--live", "--telegram"]
                print("🔴 Starting LIVE trading...")
            else:
                print("❌ Live trading cancelled")
                return

        elif choice == "3":
            # Paper trading
            cmd = [sys.executable, "main.py", "--paper", "--telegram"]
            print("📝 Starting Paper Trading mode...")

        elif choice == "4":
            # Dry run
            cmd = [sys.executable, "main.py", "--dry-run", "--telegram"]
            print("🔍 Starting Dry Run mode...")

        elif choice == "5":
            # Check status
            cmd = [sys.executable, "main.py", "--status"]
            print("📊 Checking bot status...")

        elif choice == "6":
            # Check balance
            cmd = [sys.executable, "main.py", "--balance"]
            print("💰 Checking account balance...")

        elif choice == "7":
            # Custom settings
            show_custom_settings()
            return

        elif choice == "8":
            # Show CLI examples
            return

        elif choice == "9":
            print("👋 Goodbye!")
            return

        else:
            print("❌ Invalid choice")
            return

        print("=" * 50)
        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\n🛑 Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

def show_custom_settings():
    """Show custom settings menu"""

    print("\n🔧 CUSTOM SETTINGS:")
    print("=" * 30)

    # Get basic settings
    mode = input("Trading mode (demo/live/paper): ").strip().lower()
    amount = input("Trade amount (default 10): ").strip() or "10"
    symbol = input("Symbol (default R_100): ").strip() or "R_100"
    execution = input("Execution (MT5/WEBSOCKET/AUTO): ").strip().upper() or "AUTO"

    # Build command
    cmd = [sys.executable, "main.py"]

    if mode == "demo":
        cmd.append("--demo")
    elif mode == "live":
        cmd.append("--live")
    elif mode == "paper":
        cmd.append("--paper")

    cmd.extend([
        "--amount", amount,
        "--symbol", symbol,
        "--execution", execution,
        "--telegram"
    ])

    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("=" * 50)

    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
