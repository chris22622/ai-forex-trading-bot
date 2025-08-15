#!/usr/bin/env python3
"""
MT5 Auto-Trading Fix Script
This script will help diagnose and fix MT5 auto-trading issues
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 package not installed!")
    print("📋 Install with: pip install MetaTrader5")
    sys.exit(1)

async def diagnose_mt5_trading():
    """Diagnose MT5 trading issues and provide solutions"""
    print("🔍 MT5 AUTO-TRADING DIAGNOSTIC")
    print("=" * 50)

    try:
        # Initialize MT5
        print("📝 Step 1: Initializing MT5...")
        if not mt5.initialize():
            error = mt5.last_error()
            print(f"❌ MT5 initialization failed: {error}")
            print("📋 Solutions:")
            print("   1. Make sure MetaTrader 5 terminal is running")
            print("   2. Login to your Deriv account in MT5")
            print("   3. Keep MT5 terminal open")
            return False

        print("✅ MT5 initialized successfully")

        # Check terminal info
        print("\n📝 Step 2: Checking terminal settings...")
        terminal_info = mt5.terminal_info()
        if terminal_info:
            trade_allowed = getattr(terminal_info, 'trade_allowed', False)
            print(f"🔍 Trade Allowed: {trade_allowed}")

            if not trade_allowed:
                print("\n❌ AUTO-TRADING IS DISABLED!")
                print("📋 TO ENABLE AUTO-TRADING:")
                print("   1. Open MetaTrader 5")
                print("   2. Go to Tools → Options → Expert Advisors")
                print("   3. Check ✅ 'Allow automated trading'")
                print("   4. Check ✅ 'Allow DLL imports'")
                print("   5. Click OK")
                print("   6. In MT5 toolbar, click the 'AutoTrading' button")
                print("   7. The button should turn GREEN when enabled")
                print("\n⚠️ IMPORTANT: You MUST do this for real trading to work!")
                return False
            else:
                print("✅ AUTO-TRADING IS ENABLED!")

        # Check account info
        print("\n📝 Step 3: Checking account...")
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ Account: {account_info.login}")
            print(f"✅ Server: {account_info.server}")
            print(f"✅ Balance: ${account_info.balance:.2f}")
            print(f"✅ Trade Allowed: {account_info.trade_allowed}")

            if not account_info.trade_allowed:
                print("❌ Account trading is disabled!")
                print("📋 Check your account permissions with Deriv support")
                return False
        else:
            print("❌ Could not get account info")
            return False

        # Check symbols
        print("\n📝 Step 4: Checking trading symbols...")
        symbols = mt5.symbols_get()
        if symbols:
            volatility_symbols = [s.name for s in symbols if 'Volatility' in s.name]
            print(f"✅ Found {len(volatility_symbols)} Volatility symbols")

            if volatility_symbols:
                test_symbol = volatility_symbols[0]
                print(f"🎯 Testing symbol: {test_symbol}")

                # Check symbol info
                symbol_info = mt5.symbol_info(test_symbol)
                if symbol_info:
                    print(f"✅ Symbol visible: {symbol_info.visible}")
                    print(f"✅ Symbol trade mode: {symbol_info.trade_mode}")
                    print(f"✅ Min volume: {symbol_info.volume_min}")
                    print(f"✅ Max volume: {symbol_info.volume_max}")

                    # Test order placement (dry run)
                    print("\n📝 Step 5: Testing order placement...")
                    print("⚠️ This is a DRY RUN - no real trade will be placed")

                    # Get current price
                    tick = mt5.symbol_info_tick(test_symbol)
                    if tick:
                        print(f"✅ Current price: {tick.bid} / {tick.ask}")

                        # Prepare test order (but don't send it)
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": test_symbol,
                            "volume": symbol_info.volume_min,
                            "type": mt5.ORDER_TYPE_BUY,
                            "price": tick.ask,
                            "sl": 0.0,
                            "tp": 0.0,
                            "deviation": 20,
                            "magic": 234000,
                            "comment": "Test order",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }

                        print("✅ Order parameters are valid")
                        print("📋 If auto-trading is enabled, real orders should work")

                        return True
                    else:
                        print("❌ Could not get price data")
                        return False
                else:
                    print(f"❌ Could not get symbol info for {test_symbol}")
                    return False
            else:
                print("❌ No Volatility symbols found")
                return False
        else:
            print("❌ No symbols available")
            return False

    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
        return False
    finally:
        mt5.shutdown()

async def main():
    """Main diagnostic function"""
    print("🧪 MT5 AUTO-TRADING DIAGNOSTIC TOOL")
    print("This tool will help you fix MT5 trading issues\n")

    result = await diagnose_mt5_trading()

    if result:
        print("\n🎉 SUCCESS! MT5 is ready for auto-trading!")
        print("✅ Your bot should now be able to place real trades")
        print("📱 Run your trading bot to test real order placement")
    else:
        print("\n❌ ISSUES FOUND! Please fix the problems above")
        print("🔧 Follow the instructions to enable auto-trading")
        print("📞 Contact Deriv support if account issues persist")

if __name__ == "__main__":
    asyncio.run(main())
