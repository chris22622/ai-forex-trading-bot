#!/usr/bin/env python3
"""
MT5 Initialization Diagnostics and Fix Tool
This script helps diagnose and fix MT5 initialization issues
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime

def print_header():
    print("=" * 80)
    print("🔧 MT5 INITIALIZATION DIAGNOSTICS & FIX TOOL")
    print("=" * 80)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_mt5_package():
    """Check if MetaTrader5 package is installed"""
    print("📦 CHECKING METATRADER5 PACKAGE...")
    try:
        import MetaTrader5 as mt5
        print("✅ MetaTrader5 package is installed")
        print(f"   Version: {mt5.__version__ if hasattr(mt5, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        print("❌ MetaTrader5 package not found")
        print(f"   Error: {e}")
        print("💡 Fix: pip install MetaTrader5")
        return False
    except Exception as e:
        print(f"❌ Error importing MetaTrader5: {e}")
        return False

def check_mt5_integration():
    """Check if mt5_integration.py exists"""
    print("\\n🔗 CHECKING MT5 INTEGRATION MODULE...")
    try:
        from mt5_integration import MT5TradingInterface
        print("✅ mt5_integration.py found and importable")
        return True
    except ImportError as e:
        print("❌ mt5_integration.py not found or not importable")
        print(f"   Error: {e}")
        print("💡 Fix: Make sure mt5_integration.py exists in the project directory")
        return False
    except Exception as e:
        print(f"❌ Error importing mt5_integration: {e}")
        return False

def check_mt5_terminal():
    """Check if MT5 terminal is running"""
    print("\\n🖥️ CHECKING MT5 TERMINAL...")
    try:
        import MetaTrader5 as mt5
        
        # Try to initialize
        if mt5.initialize():
            print("✅ MT5 terminal is running and accessible")
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                print(f"   Account: {account_info.login}")
                print(f"   Server: {account_info.server}")
                print(f"   Balance: ${account_info.balance:.2f}")
                print(f"   Currency: {account_info.currency}")
            
            # Check symbols
            symbols = mt5.symbols_get()
            if symbols:
                print(f"   Available symbols: {len(symbols)}")
            
            mt5.shutdown()
            return True
        else:
            print("❌ Cannot initialize MT5 terminal")
            print("💡 Fix: Make sure MetaTrader 5 is running and logged in")
            return False
            
    except Exception as e:
        print(f"❌ Error checking MT5 terminal: {e}")
        print("💡 Fix: Install/restart MetaTrader 5 application")
        return False

async def test_mt5_interface():
    """Test the MT5TradingInterface"""
    print("\\n🧪 TESTING MT5 TRADING INTERFACE...")
    try:
        from mt5_integration import MT5TradingInterface
        
        interface = MT5TradingInterface()
        print("✅ MT5TradingInterface created successfully")
        
        # Test initialization
        success = await interface.initialize()
        if success:
            print("✅ MT5TradingInterface initialized successfully")
            
            # Test getting balance
            try:
                balance = await interface.get_account_balance()
                if balance is not None:
                    print(f"✅ Account balance retrieved: ${balance:.2f}")
                else:
                    print("⚠️ Account balance is None")
            except Exception as e:
                print(f"❌ Error getting balance: {e}")
            
            # Test getting price
            try:
                price = await interface.get_current_price("Volatility 75 Index")
                if price:
                    print(f"✅ Price data retrieved: {price}")
                else:
                    print("⚠️ No price data available")
            except Exception as e:
                print(f"❌ Error getting price: {e}")
            
            return True
        else:
            print("❌ MT5TradingInterface initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing MT5TradingInterface: {e}")
        traceback.print_exc()
        return False

def provide_solutions():
    """Provide comprehensive solutions"""
    print("\\n🛠️ COMMON SOLUTIONS:")
    print("=" * 50)
    print("1. Install MetaTrader5 package:")
    print("   pip install --upgrade MetaTrader5")
    print()
    print("2. Make sure MetaTrader 5 terminal is running:")
    print("   - Download from: https://www.metatrader5.com/")
    print("   - Login with valid demo/live account")
    print("   - Keep terminal open while running bot")
    print()
    print("3. Check mt5_integration.py exists:")
    print("   - File should be in the same directory as main.py")
    print("   - Contains MT5TradingInterface class")
    print()
    print("4. Verify Python environment:")
    print("   - Use correct virtual environment")
    print("   - Python 3.8+ recommended")
    print()
    print("5. If still having issues:")
    print("   - Restart MetaTrader 5 terminal")
    print("   - Restart your Python script")
    print("   - Check Windows firewall/antivirus")

async def main():
    """Main diagnostic function"""
    print_header()
    
    # Run all checks
    mt5_package_ok = check_mt5_package()
    mt5_integration_ok = check_mt5_integration()
    mt5_terminal_ok = check_mt5_terminal()
    
    if mt5_package_ok and mt5_integration_ok:
        mt5_interface_ok = await test_mt5_interface()
    else:
        mt5_interface_ok = False
    
    # Summary
    print("\\n📊 DIAGNOSTIC SUMMARY:")
    print("=" * 50)
    print(f"MetaTrader5 Package: {'✅ OK' if mt5_package_ok else '❌ FAILED'}")
    print(f"MT5 Integration:     {'✅ OK' if mt5_integration_ok else '❌ FAILED'}")
    print(f"MT5 Terminal:        {'✅ OK' if mt5_terminal_ok else '❌ FAILED'}")
    print(f"MT5 Interface:       {'✅ OK' if mt5_interface_ok else '❌ FAILED'}")
    
    all_ok = mt5_package_ok and mt5_integration_ok and mt5_terminal_ok and mt5_interface_ok
    
    if all_ok:
        print("\\n🎉 ALL CHECKS PASSED! MT5 should work correctly.")
    else:
        print("\\n⚠️ Issues found. See solutions below.")
        provide_solutions()
    
    print("\\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
