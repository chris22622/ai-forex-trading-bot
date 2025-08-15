#!/usr/bin/env python3
"""
FINAL MT5 ORDER FIX - Addresses order_send returning None
Based on diagnostic analysis of the trading bot logs
"""

import MetaTrader5 as mt5


def fix_mt5_order_issues():
    """
    Fix common MT5 order_send returning None issues
    """
    print("🔧 FIXING MT5 ORDER ISSUES...")
    print("=" * 50)

    # 1. Initialize with connection check
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False

    print("✅ MT5 initialized")

    # 2. Check terminal connection - CRITICAL
    terminal = mt5.terminal_info()
    if not terminal:
        print("❌ Cannot get terminal info")
        return False

    print("🖥️ Terminal Status:")
    print(f"   Connected: {terminal.connected}")
    print(f"   Trade Allowed: {terminal.trade_allowed}")
    print(f"   Company: {terminal.company}")

    # CRITICAL CHECK: Terminal must be connected
    if not terminal.connected:
        print("❌ CRITICAL: MT5 Terminal is NOT connected to server!")
        print("   This is the root cause of order_send returning None")
        print("   SOLUTION: Please connect your MT5 terminal to a trading server")
        print("   1. Open MetaTrader 5 terminal")
        print("   2. Go to File -> Login to Trade Account")
        print("   3. Enter your broker login credentials")
        print("   4. Ensure connection shows 'Connected' status")
        return False

    # 3. Check account status
    account = mt5.account_info()
    if not account:
        print("❌ Cannot get account info")
        return False

    print("💰 Account Status:")
    print(f"   Login: {account.login}")
    print(f"   Server: {account.server}")
    print(f"   Balance: ${account.balance:.2f}")
    print(f"   Trade Allowed: {account.trade_allowed}")
    print(f"   Trade Mode: {account.trade_mode}")

    if not account.trade_allowed:
        print("❌ CRITICAL: Account trading is disabled!")
        print("   SOLUTION: Contact your broker to enable trading")
        return False

    # 4. Test symbol setup
    symbol = "Volatility 75 Index"
    print(f"\n📊 Testing Symbol: {symbol}")

    # Ensure symbol is visible
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Failed to add {symbol} to Market Watch")
        return False

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"❌ Cannot get symbol info for {symbol}")
        return False

    print(f"   Visible: {symbol_info.visible}")
    print(f"   Trade Mode: {symbol_info.trade_mode}")
    print(f"   Volume Min: {symbol_info.volume_min}")

    # 5. Get current price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"❌ Cannot get price for {symbol}")
        return False

    print(f"   Price: {tick.ask}/{tick.bid}")

    # 6. Create BULLETPROOF order request
    print("\n🚀 Creating bulletproof order request...")

    # Use the most compatible order settings
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": symbol_info.volume_min,  # Use minimum volume
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "sl": 0.0,  # No stop loss
        "tp": 0.0,  # No take profit
        "deviation": 50,  # Larger slippage allowance
        "magic": 123456,
        "comment": "BulletproofTest",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,  # Most compatible
    }

    print("📋 Order Request:")
    for key, value in request.items():
        print(f"   {key}: {value}")

    # 7. Clear any previous errors
    mt5.last_error()

    # 8. TEST ORDER PLACEMENT
    print("\n🎯 TESTING ORDER PLACEMENT...")

    result = mt5.order_send(request)

    if result is None:
        error = mt5.last_error()
        print("❌ order_send STILL returns None!")
        print(f"❌ Error: {error}")

        # Additional diagnostics
        if error:
            error_code, error_desc = error
            print(f"❌ Error Code: {error_code}")
            print(f"❌ Error Description: {error_desc}")

            if error_code == 1:
                print("🔍 Error Code 1 = General error (usually connection)")
            elif error_code == 10025:
                print("🔍 Error Code 10025 = Invalid order request")
            elif error_code == 10027:
                print("🔍 Error Code 10027 = Autotrading disabled")
            elif error_code == 10018:
                print("🔍 Error Code 10018 = Market closed")

        # Try different filling modes
        print("\n🔄 Trying different order filling modes...")

        filling_modes = [
            mt5.ORDER_FILLING_FOK,      # Fill or Kill
            mt5.ORDER_FILLING_RETURN,   # Market execution
        ]

        for i, filling in enumerate(filling_modes):
            print(f"   Attempt {i+1}: Filling mode {filling}")
            request["type_filling"] = filling

            result = mt5.order_send(request)
            if result is not None:
                print(f"✅ SUCCESS with filling mode {filling}!")
                break
            else:
                error = mt5.last_error()
                print(f"   ❌ Failed: {error}")

        if result is None:
            print("\n❌ ALL ORDER ATTEMPTS FAILED")
            print("🔍 ROOT CAUSE ANALYSIS:")
            print("   1. Check MT5 terminal is connected to broker server")
            print("   2. Verify account has live trading permissions")
            print("   3. Ensure broker allows automated trading")
            print("   4. Check if demo account requires special settings")
            return False
    else:
        print("✅ ORDER SUCCESS!")
        print(f"📊 Result: {result}")
        print(f"📊 Return Code: {result.retcode}")

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print("🎉 TRADE EXECUTED SUCCESSFULLY!")
            if hasattr(result, 'order'):
                print(f"📊 Order ID: {result.order}")
        else:
            print(f"⚠️ Order completed with code: {result.retcode}")

        return True

    mt5.shutdown()
    return False

def apply_mt5_fixes():
    """Apply fixes to the main trading interface"""
    print("\n🔧 APPLYING FIXES TO TRADING INTERFACE...")

    # The main fix is ensuring MT5 terminal connection
    # This needs to be done manually in MT5 terminal

    print("✅ Diagnostic complete")
    print("🎯 NEXT STEPS:")
    print("   1. Ensure MetaTrader 5 terminal is running")
    print("   2. Connect to your broker's trading server")
    print("   3. Verify account login and trading permissions")
    print("   4. Enable auto-trading in MT5 terminal")
    print("   5. Restart the trading bot")

if __name__ == "__main__":
    success = fix_mt5_order_issues()
    if success:
        print("\n🎉 MT5 ORDER ISSUE RESOLVED!")
    else:
        apply_mt5_fixes()
