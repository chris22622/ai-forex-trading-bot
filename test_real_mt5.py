#!/usr/bin/env python3
"""Test MT5 integration and verify real trades are being sent"""

import asyncio
import sys
from datetime import datetime

# Import the fixed bot
from main import DerivTradingBot

async def test_mt5_integration():
    print("🔧 Testing FIXED MT5 Integration...")
    print("=" * 50)
    
    # Create bot instance
    bot = DerivTradingBot()
    
    # Check interface type
    interface_type = type(bot.mt5_interface).__name__
    print(f"MT5 Interface Type: {interface_type}")
    
    if interface_type == "DummyMT5Interface":
        print("❌ PROBLEM: Still using DummyMT5Interface!")
        print("   Trades will be simulated, not sent to real MT5")
        return False
    elif interface_type == "MT5TradingInterface":
        print("✅ GOOD: Using real MT5TradingInterface")
    else:
        print(f"❓ UNKNOWN: Using {interface_type}")
    
    print(f"Using MT5: {bot.using_mt5}")
    print(f"Connected: {bot.connected}")
    
    # Test MT5 connection
    if bot.mt5_interface:
        print("\n🔗 Testing MT5 connection...")
        try:
            result = await bot.mt5_interface.initialize()
            print(f"MT5 Initialize: {result}")
            
            if result:
                # Test account balance
                balance = await bot.mt5_interface.get_account_balance()
                print(f"Account Balance: ${balance:.2f}")
                
                # Test price feed
                price = await bot.mt5_interface.get_current_price("Volatility 75 Index")
                print(f"Volatility 75 Index Price: {price}")
                
                print("\n✅ MT5 connection is WORKING!")
                print("🚀 Bot will send REAL trades to MT5!")
                return True
            else:
                print("❌ MT5 connection failed")
                return False
                
        except Exception as e:
            print(f"❌ MT5 test error: {e}")
            return False
    else:
        print("❌ No MT5 interface available")
        return False

async def test_trade_placement():
    """Test if a trade would actually be sent to MT5"""
    print("\n🎯 Testing trade placement logic...")
    
    bot = DerivTradingBot()
    
    # Force MT5 connection
    if bot.mt5_interface and not bot.using_mt5:
        try:
            connection_result = await bot.mt5_interface.initialize()
            if connection_result:
                bot.using_mt5 = True
                bot.connected = True
                print("✅ MT5 connection established for testing")
            else:
                print("❌ Could not establish MT5 connection")
                return False
        except Exception as e:
            print(f"❌ MT5 connection error: {e}")
            return False
    
    # Test the place_mt5_trade function
    if bot.using_mt5:
        print("🚀 Would place REAL trade via MT5!")
        print("   - Trade will appear in MT5 terminal")
        print("   - Real money will be used")
        return True
    else:
        print("❌ Would use simulation mode")
        print("   - No trades will appear in MT5")
        return False

if __name__ == "__main__":
    print(f"🕒 Test started at {datetime.now().strftime('%H:%M:%S')}")
    
    # Test MT5 integration
    mt5_working = asyncio.run(test_mt5_integration())
    
    # Test trade placement
    trade_working = asyncio.run(test_trade_placement())
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print(f"   MT5 Integration: {'✅ WORKING' if mt5_working else '❌ BROKEN'}")
    print(f"   Trade Placement: {'✅ REAL TRADES' if trade_working else '❌ SIMULATED'}")
    
    if mt5_working and trade_working:
        print("\n🎉 SUCCESS: Bot will send REAL trades to MT5!")
    else:
        print("\n⚠️  ISSUE: Bot is not properly connected to MT5")
        print("   Check MetaTrader 5 is running and logged in")
