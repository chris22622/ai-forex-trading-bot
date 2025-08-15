"""
MetaTrader 5 Bot Launcher and Setup Guide
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

def check_mt5_requirements() -> bool:
    """Check if all MT5 requirements are met"""
    print("🔧 CHECKING MT5 REQUIREMENTS")
    print("-" * 40)

    requirements_met = True

    # Check if MetaTrader5 library is installed
    try:
        import MetaTrader5 as mt5  # type: ignore
        print("✅ MetaTrader5 Python library installed")
    except ImportError:
        print("❌ MetaTrader5 library not installed")
        print("   Run: pip install MetaTrader5")
        requirements_met = False

    # Check if MT5 terminal is running (this will be tested during connection)
    print("⏳ MT5 terminal status will be checked during connection...")

    print()
    return requirements_met

def print_mt5_setup_guide() -> None:
    """Print detailed setup guide for MT5 trading"""
    print("📋 METATRADER 5 SETUP GUIDE")
    print("=" * 50)
    print()

    print("1. 📥 INSTALL METATRADER 5:")
    print("   - Download from: https://www.metatrader5.com/")
    print("   - Install and run MT5 terminal")
    print()

    print("2. 🔗 CONNECT DERIV ACCOUNT:")
    print("   - In MT5: File → Login to Trade Account")
    print("   - Use your Deriv MT5 login credentials")
    print("   - Server: Should auto-detect Deriv server")
    print()

    print("3. ⚙️ ENABLE AUTOMATED TRADING:")
    print("   - In MT5: Tools → Options → Expert Advisors")
    print("   - Check 'Allow automated trading'")
    print("   - Check 'Allow DLL imports'")
    print("   - Restart MT5 after changes")
    print()

    print("4. 📊 VERIFY DERIV SYMBOLS:")
    print("   - Look for symbols like:")
    print("     • Volatility 100 Index")
    print("     • Volatility 75 Index")
    print("     • Boom 1000 Index")
    print("     • Crash 1000 Index")
    print("   - If not visible, contact Deriv support")
    print()

    print("5. 🚀 RUN THE BOT:")
    print("   - Ensure MT5 is running and logged in")
    print("   - Run: python main_mt5.py")
    print("   - The bot will auto-connect to your MT5 terminal")
    print()

    print("⚠️  IMPORTANT NOTES:")
    print("   - Keep MT5 running while the bot trades")
    print("   - Monitor your account balance and positions")
    print("   - Start with small amounts for testing")
    print("   - The bot will use your actual MT5 account")
    print()

async def test_mt5_connection() -> bool:
    """Test MetaTrader 5 connection"""
    try:
        print("🔄 TESTING MT5 CONNECTION")
        print("-" * 30)

        import MetaTrader5 as mt5  # type: ignore

        # Initialize MT5
        if not mt5.initialize():  # type: ignore
            error = mt5.last_error()  # type: ignore
            print(f"❌ MT5 initialization failed: {error}")
            print()
            print("🔧 TROUBLESHOOTING:")
            print("1. Make sure MetaTrader 5 is running")
            print("2. Ensure you're logged into your Deriv account")
            print("3. Check if MT5 allows automated trading")
            print("4. Try restarting MT5 terminal")
            return False

        print("✅ MT5 initialized successfully")

        # Get account info
        account_info = mt5.account_info()  # type: ignore
        if account_info is None:
            print("❌ Failed to get account info")
            print("   Make sure you're logged into your trading account")
            mt5.shutdown()  # type: ignore
            return False

        print(f"✅ Connected to account: {account_info.login}")  # type: ignore
        print(f"📊 Server: {account_info.server}")  # type: ignore
        print(f"💰 Balance: ${account_info.balance:.2f}")  # type: ignore
        print(f"💎 Equity: ${account_info.equity:.2f}")  # type: ignore
        print(f"💱 Currency: {account_info.currency}")  # type: ignore

        # Check for Deriv symbols
        print("\n📊 CHECKING DERIV SYMBOLS:")
        symbols = mt5.symbols_get()  # type: ignore
        deriv_symbols: List[str] = []

        if symbols:
            for symbol in symbols:  # type: ignore
                if any(keyword in symbol.name.lower() for keyword in   # type: ignore
                      ['volatility', 'boom', 'crash', 'step', 'jump']):
                    deriv_symbols.append(symbol.name)  # type: ignore

        if deriv_symbols:
            print(f"✅ Found {len(deriv_symbols)} Deriv symbols:")
            for symbol in deriv_symbols[:5]:  # Show first 5
                print(f"   - {symbol}")
            if len(deriv_symbols) > 5:
                print(f"   ... and {len(deriv_symbols) - 5} more")
        else:
            print("⚠️  No Deriv symbols found")
            print("   Contact Deriv support to enable synthetic indices")

        # Shutdown test connection
        mt5.shutdown()  # type: ignore
        print("\n✅ MT5 connection test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ MT5 connection test failed: {e}")
        return False

async def launch_mt5_bot() -> None:
    """Launch the MT5 trading bot"""
    try:
        print("\n🚀 LAUNCHING MT5 TRADING BOT")
        print("=" * 35)

        # Import and run the MT5 bot
        from main_mt5 import main as mt5_main
        await mt5_main()

    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Bot launch error: {e}")
        import traceback
        traceback.print_exc()

async def main() -> None:
    """Main launcher function"""
    print("🎯" * 20)
    print("🤖 DERIV MT5 TRADING BOT LAUNCHER")
    print("🎯" * 20)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check requirements
    if not check_mt5_requirements():
        print("\n❌ Requirements not met. Please install missing components.")
        return

    # Show setup guide
    print_mt5_setup_guide()

    # Ask user what they want to do
    print("🎯 WHAT WOULD YOU LIKE TO DO?")
    print("1. Test MT5 connection")
    print("2. Launch trading bot")
    print("3. Show setup guide again")
    print("4. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                await test_mt5_connection()
                print("\n" + "="*50)
                continue

            elif choice == "2":
                print("\n⚠️  WARNING: This will start live trading!")
                print("Make sure you've tested the connection first.")
                confirm = input("Continue? (yes/no): ").strip().lower()

                if confirm in ['yes', 'y']:
                    await launch_mt5_bot()
                    break
                else:
                    print("❌ Bot launch cancelled")
                    continue

            elif choice == "3":
                print()
                print_mt5_setup_guide()
                continue

            elif choice == "4":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
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
