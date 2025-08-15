#!/usr/bin/env python3
"""
MT5 Order Send Fix - Enhanced Order Placement
"""

import asyncio
import time

import MetaTrader5 as mt5


class MT5OrderFix:
    """Enhanced MT5 order handling with better error detection"""

    def __init__(self):
        self.magic_number = 234000
        self.slippage = 20

    async def diagnose_and_fix_order_issue(self, symbol: str = "Volatility 75 Index"):
        """Comprehensive MT5 order issue diagnosis and fix"""
        print("🔍 MT5 ORDER SEND DIAGNOSTIC & FIX")
        print("=" * 50)

        # 1. Initialize MT5
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return None

        print("✅ MT5 initialized")

        # 2. Check terminal status
        terminal = mt5.terminal_info()
        if not terminal:
            print("❌ No terminal info available")
            mt5.shutdown()
            return None

        print(f"✅ Terminal - Connected: {terminal.connected}, Trade Allowed: {terminal.trade_allowed}")

        if not terminal.connected:
            print("❌ Terminal not connected to trade server")
            mt5.shutdown()
            return None

        if not terminal.trade_allowed:
            print("❌ Trading not allowed in terminal")
            mt5.shutdown()
            return None

        # 3. Check account
        account = mt5.account_info()
        if not account:
            print("❌ No account info available")
            mt5.shutdown()
            return None

        print(f"✅ Account - Login: {account.login}, Trade Allowed: {account.trade_allowed}, Balance: {account.balance}")

        if not account.trade_allowed:
            print("❌ Trading not allowed for account")
            mt5.shutdown()
            return None

        # 4. Check symbol
        print(f"🔍 Checking symbol: {symbol}")

        # First, try to select the symbol
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Could not select symbol: {mt5.last_error()}")
            mt5.shutdown()
            return None

        # Wait a bit for symbol to load
        time.sleep(1)

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"❌ No symbol info for {symbol}: {mt5.last_error()}")
            mt5.shutdown()
            return None

        print("✅ Symbol Info:")
        print(f"   Visible: {symbol_info.visible}")
        print(f"   Selected: {symbol_info.select}")
        print(f"   Trade Mode: {symbol_info.trade_mode}")
        print(f"   Order Mode: {symbol_info.order_mode}")
        print(f"   Filling Mode: {symbol_info.filling_mode}")
        print(f"   Min Volume: {symbol_info.volume_min}")
        print(f"   Max Volume: {symbol_info.volume_max}")
        print(f"   Volume Step: {symbol_info.volume_step}")
        print(f"   Bid: {symbol_info.bid}")
        print(f"   Ask: {symbol_info.ask}")

        # Check if symbol allows trading
        if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            print("❌ Trading disabled for this symbol")
            mt5.shutdown()
            return None

        # 5. Test order placement with proper parameters
        print("\n🚀 Testing order placement...")

        # Determine correct filling mode based on symbol
        if symbol_info.filling_mode & mt5.SYMBOL_FILLING_IOC:
            filling = mt5.ORDER_FILLING_IOC
            filling_name = "IOC"
        elif symbol_info.filling_mode & mt5.SYMBOL_FILLING_FOK:
            filling = mt5.ORDER_FILLING_FOK
            filling_name = "FOK"
        else:
            filling = mt5.ORDER_FILLING_RETURN
            filling_name = "RETURN"

        print(f"   Using filling mode: {filling_name}")

        # Clear any previous errors
        mt5.last_error()

        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": symbol_info.volume_min,  # Use minimum volume
            "type": mt5.ORDER_TYPE_SELL,
            "price": symbol_info.bid,
            "deviation": self.slippage,
            "magic": self.magic_number,
            "comment": "MT5 Fix Test",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        print(f"   Order request: {request}")

        # Try to send order
        result = mt5.order_send(request)
        error = mt5.last_error()

        print(f"   Order result: {result}")
        print(f"   Last error: {error}")

        if result is None:
            print("❌ ORDER_SEND RETURNED NONE - INVESTIGATING...")

            # Additional checks when order_send returns None

            # Check if symbol is still visible
            current_symbol_info = mt5.symbol_info(symbol)
            if current_symbol_info:
                print(f"   Symbol still visible: {current_symbol_info.visible}")

            # Check current market state
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                print(f"   Current tick - Bid: {tick.bid}, Ask: {tick.ask}, Time: {tick.time}")

            # Try with different order types and filling modes
            print("\n🔄 Trying alternative order configurations...")

            test_configs = [
                {"type": mt5.ORDER_TYPE_BUY, "price": symbol_info.ask, "filling": mt5.ORDER_FILLING_IOC},
                {"type": mt5.ORDER_TYPE_SELL, "price": symbol_info.bid, "filling": mt5.ORDER_FILLING_FOK},
                {"type": mt5.ORDER_TYPE_BUY, "price": symbol_info.ask, "filling": mt5.ORDER_FILLING_RETURN},
            ]

            for i, config in enumerate(test_configs):
                print(f"   Test {i+1}: {config}")
                test_request = request.copy()
                test_request.update(config)

                mt5.last_error()  # Clear errors
                test_result = mt5.order_send(test_request)
                test_error = mt5.last_error()

                print(f"     Result: {test_result}")
                print(f"     Error: {test_error}")

                if test_result and test_result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"   ✅ SUCCESS with config {i+1}!")
                    result = test_result
                    break

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ ORDER SUCCESSFUL!")
            print(f"   Order: {result.order}")
            print(f"   Deal: {result.deal}")
            print(f"   Volume: {result.volume}")
            print(f"   Price: {result.price}")
        elif result:
            print(f"❌ ORDER FAILED: {result.retcode} - {result.comment}")
        else:
            print("❌ ORDER_SEND CONSISTENTLY RETURNS NONE")
            print("   This indicates a fundamental MT5 configuration issue")

        # 6. Final diagnostics
        print(f"\n🔍 Final Error Check: {mt5.last_error()}")

        # Cleanup
        mt5.shutdown()
        return result

    async def fix_mt5_configuration(self):
        """Apply fixes for common MT5 issues"""
        print("🔧 Applying MT5 configuration fixes...")

        if not mt5.initialize():
            print("❌ Cannot initialize MT5")
            return False

        # Enable all symbols in Market Watch
        symbols = mt5.symbols_get()
        if symbols:
            print(f"   Found {len(symbols)} total symbols")

            # Make sure our target symbol is visible
            target_symbols = ["Volatility 75 Index", "XAUUSD", "EURUSD"]
            for symbol in target_symbols:
                if mt5.symbol_select(symbol, True):
                    print(f"   ✅ Added {symbol} to Market Watch")
                else:
                    print(f"   ❌ Could not add {symbol}")

        mt5.shutdown()
        return True

async def main():
    fixer = MT5OrderFix()

    # Apply configuration fixes
    await fixer.fix_mt5_configuration()

    # Run diagnostic
    result = await fixer.diagnose_and_fix_order_issue()

    if result:
        print("\n✅ MT5 ORDER ISSUE RESOLVED!")
    else:
        print("\n❌ MT5 ORDER ISSUE PERSISTS")
        print("Possible solutions:")
        print("1. Check MT5 terminal is connected to broker")
        print("2. Verify account has trading permissions")
        print("3. Check symbol trading hours")
        print("4. Contact broker support")

if __name__ == "__main__":
    asyncio.run(main())
