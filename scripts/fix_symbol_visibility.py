#!/usr/bin/env python3
"""
Fix MT5 Symbol Visibility Issues
This script will add trading symbols to Market Watch and test trading
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 package not installed!")
    sys.exit(1)

async def fix_symbol_visibility():
    """Fix symbol visibility issues in MT5"""
    print("🔧 FIXING MT5 SYMBOL VISIBILITY")
    print("=" * 40)

    try:
        # Initialize MT5
        if not mt5.initialize():
            error = mt5.last_error()
            print(f"❌ MT5 initialization failed: {error}")
            return False

        print("✅ MT5 initialized")

        # Get all available symbols
        symbols = mt5.symbols_get()
        if not symbols:
            print("❌ No symbols found")
            return False

        # Find Volatility symbols
        volatility_symbols = [s.name for s in symbols if 'Volatility' in s.name]
        print(f"📊 Found {len(volatility_symbols)} Volatility symbols")

        # Priority symbols to make visible
        priority_symbols = [
            'Volatility 75 Index',
            'Volatility 100 Index',
            'Volatility 50 Index',
            'Volatility 25 Index',
            'Volatility 10 Index'
        ]

        fixed_symbols = []

        # Make symbols visible and test them
        for symbol in priority_symbols:
            if symbol in volatility_symbols:
                print(f"\n🔧 Processing {symbol}...")

                # Add to Market Watch
                if mt5.symbol_select(symbol, True):
                    print(f"✅ Added {symbol} to Market Watch")

                    # Check if now visible
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info and symbol_info.visible:
                        print(f"✅ {symbol} is now visible")

                        # Test getting price
                        tick = mt5.symbol_info_tick(symbol)
                        if tick:
                            print(f"✅ Price available: {tick.bid} / {tick.ask}")
                            fixed_symbols.append(symbol)
                        else:
                            print(f"❌ No price data for {symbol}")
                    else:
                        print(f"❌ {symbol} still not visible")
                else:
                    print(f"❌ Failed to add {symbol} to Market Watch")
            else:
                print(f"⚠️ {symbol} not found in available symbols")

        if fixed_symbols:
            print(f"\n🎉 SUCCESS! Fixed {len(fixed_symbols)} symbols:")
            for symbol in fixed_symbols:
                print(f"   ✅ {symbol}")

            # Test a small order on the first working symbol
            test_symbol = fixed_symbols[0]
            print(f"\n🧪 Testing order on {test_symbol}...")

            symbol_info = mt5.symbol_info(test_symbol)
            tick = mt5.symbol_info_tick(test_symbol)

            if symbol_info and tick:
                # Prepare minimal test order
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
                    "comment": "Test_Order_Bot",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                print("⚠️ About to place a REAL test order!")
                print(f"Symbol: {test_symbol}")
                print(f"Volume: {symbol_info.volume_min}")
                print(f"Price: {tick.ask}")

                # Ask for confirmation
                confirm = input("\nPlace REAL test order? (type YES to confirm): ")
                if confirm.upper() == "YES":
                    result = mt5.order_send(request)

                    if result is None:
                        print("❌ order_send returned None - this is the problem!")
                        print("📋 Check MT5 auto-trading settings again")
                        return False
                    elif result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"🎉 SUCCESS! Test order placed: {result.order}")
                        print("✅ Real trading is working!")

                        # Close the test order immediately
                        positions = mt5.positions_get(symbol=test_symbol)
                        if positions:
                            for pos in positions:
                                if pos.magic == 234000:  # Our test order
                                    close_request = {
                                        "action": mt5.TRADE_ACTION_DEAL,
                                        "symbol": test_symbol,
                                        "volume": pos.volume,
                                        "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                                        "position": pos.ticket,
                                        "price": tick.bid if pos.type == 0 else tick.ask,
                                        "deviation": 20,
                                        "magic": 234000,
                                        "comment": "Close_Test",
                                        "type_time": mt5.ORDER_TIME_GTC,
                                        "type_filling": mt5.ORDER_FILLING_IOC,
                                    }
                                    close_result = mt5.order_send(close_request)
                                    if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                                        print("✅ Test order closed successfully")
                                    break
                        return True
                    else:
                        print(f"❌ Order failed: {result.retcode}")
                        error_desc = {
                            10004: "Requote",
                            10006: "Request rejected",
                            10007: "Request canceled",
                            10008: "Order placed",
                            10009: "Request completed",
                            10010: "Only part executed",
                            10011: "Request processing error",
                            10012: "Request canceled by timeout",
                            10013: "Invalid request",
                            10014: "Invalid volume",
                            10015: "Invalid price",
                            10016: "Invalid stops",
                            10017: "Trade disabled",
                            10018: "Market closed",
                            10019: "No money",
                            10020: "Price changed",
                            10021: "No quotes to process",
                            10022: "Invalid order expiration",
                            10023: "Order state changed",
                            10024: "Too frequent requests",
                            10025: "No changes in request",
                            10026: "Auto trading disabled",
                            10027: "Auto trading disabled by client",
                            10028: "Request locked for processing",
                            10029: "Order or position frozen",
                            10030: "Invalid order filling type"
                        }
                        error_msg = error_desc.get(result.retcode, "Unknown error")
                        print(f"Error description: {error_msg}")
                        return False
                else:
                    print("🚫 Test cancelled by user")
                    return True  # Still success - symbols are fixed

            return True
        else:
            print("\n❌ No symbols could be fixed")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    asyncio.run(fix_symbol_visibility())
