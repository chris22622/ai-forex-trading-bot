"""
Unified Trading Bot Launcher
Choose between WebSocket API or MetaTrader 5 execution
"""

import asyncio
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

def print_welcome():
    """Print welcome screen"""
    print("🎯" * 25)
    print("🤖 DERIV TRADING BOT - UNIFIED LAUNCHER")
    print("🎯" * 25)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_execution_options():
    """Print available execution methods"""
    print("🔗 CHOOSE EXECUTION METHOD:")
    print("=" * 40)
    print()

    print("1. 🌐 WEBSOCKET API (Original)")
    print("   • Direct connection to Deriv API")
    print("   • Requires working internet & API tokens")
    print("   • May have connection issues")
    print()

    print("2. 📊 METATRADER 5 (Recommended)")
    print("   • Uses MT5 terminal for execution")
    print("   • More stable connection")
    print("   • Visual trading interface")
    print("   • Requires MT5 with Deriv account")
    print()

    print("3. 🧪 OFFLINE DEMO")
    print("   • Simulated trading without real connections")
    print("   • For testing bot logic")
    print("   • No real market data")
    print()

async def launch_websocket_bot():
    """Launch the original WebSocket bot"""
    try:
        print("\n🌐 LAUNCHING WEBSOCKET BOT")
        print("=" * 35)
        print("⚠️  This requires working API connection...")
        print()

        # Import and run original bot
        from main import main as websocket_main
        await websocket_main()

    except Exception as e:
        print(f"❌ WebSocket bot error: {e}")
        print("\n💡 TIP: Try MetaTrader 5 option if connection fails")

async def launch_mt5_bot():
    """Launch the MT5 bot"""
    try:
        print("\n📊 LAUNCHING MT5 BOT")
        print("=" * 30)

        # Check if MT5 is available
        try:
            import MetaTrader5 as mt5  # type: ignore
            if not mt5.initialize():  # type: ignore
                print("❌ Cannot connect to MetaTrader 5")
                print("\n🔧 SETUP REQUIRED:")
                print("1. Install MetaTrader 5")
                print("2. Login to your Deriv account")
                print("3. Enable automated trading")
                print("4. Restart MT5 and try again")
                return
            mt5.shutdown()  # type: ignore

        except ImportError:
            print("❌ MetaTrader5 library not installed")
            print("   Run: pip install MetaTrader5")
            return

        # Import and run MT5 bot
        from main_mt5 import main as mt5_main
        await mt5_main()

    except Exception as e:
        print(f"❌ MT5 bot error: {e}")
        import traceback
        traceback.print_exc()

async def launch_offline_demo():
    """Launch offline demo"""
    try:
        print("\n🧪 LAUNCHING OFFLINE DEMO")
        print("=" * 32)

        # Import and run offline demo
        from ultimate_offline_demo import main as demo_main
        await demo_main()

    except Exception as e:
        print(f"❌ Demo error: {e}")

def print_quick_setup_mt5():
    """Print quick MT5 setup guide"""
    print("\n📋 QUICK MT5 SETUP:")
    print("-" * 20)
    print("1. Download MT5: https://www.metatrader5.com/")
    print("2. Install and run MT5")
    print("3. Login with your Deriv credentials")
    print("4. Tools → Options → Expert Advisors → Allow automated trading")
    print("5. Restart MT5")
    print("6. Run this launcher again")
    print()

async def main():
    """Main launcher function"""
    print_welcome()
    print_execution_options()

    while True:
        try:
            print("❓ SELECT OPTION:")
            choice = input("Enter your choice (1-3, or 'q' to quit): ").strip().lower()

            if choice in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break

            elif choice == "1":
                await launch_websocket_bot()
                break

            elif choice == "2":
                await launch_mt5_bot()
                break

            elif choice == "3":
                await launch_offline_demo()
                break

            elif choice == "help":
                print_quick_setup_mt5()
                continue

            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 'q'")
                print("   Type 'help' for MT5 setup guide")
                continue

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Launcher error: {e}")
