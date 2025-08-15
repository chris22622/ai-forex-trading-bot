#!/usr/bin/env python3
"""
MT5 Connection and Trading Authorization Diagnostic
"""
import MetaTrader5 as mt5

def diagnose_mt5_trading():
    print("🔍 MT5 Trading Authorization Diagnostic")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return
    
    print("✅ MT5 initialized successfully")
    
    # 1. Terminal Info - CRITICAL CHECK
    terminal = mt5.terminal_info()
    if terminal:
        print(f"\n🖥️ TERMINAL STATUS:")
        print(f"   Connected: {terminal.connected}")
        print(f"   Trade Allowed: {terminal.trade_allowed}")
        print(f"   Name: {terminal.name}")
        print(f"   Company: {terminal.company}")
        print(f"   Path: {terminal.path}")
        print(f"   Data Path: {terminal.data_path}")
        print(f"   Common Data Path: {terminal.commondata_path}")
        print(f"   Language: {terminal.language}")
        print(f"   Build: {terminal.build}")
        print(f"   DLL Allowed: {terminal.dlls_allowed}")
        print(f"   Trade Context Busy: {terminal.trade_context_busy}")
    else:
        print("❌ Cannot get terminal info")
    
    # 2. Account Info - CRITICAL CHECK
    account = mt5.account_info()
    if account:
        print(f"\n💰 ACCOUNT STATUS:")
        print(f"   Login: {account.login}")
        print(f"   Server: {account.server}")
        print(f"   Name: {account.name}")
        print(f"   Company: {account.company}")
        print(f"   Balance: ${account.balance:.2f}")
        print(f"   Credit: ${account.credit:.2f}")
        print(f"   Profit: ${account.profit:.2f}")
        print(f"   Equity: ${account.equity:.2f}")
        print(f"   Margin: ${account.margin:.2f}")
        print(f"   Free Margin: ${account.margin_free:.2f}")
        print(f"   Currency: {account.currency}")
        print(f"   Trade Allowed: {account.trade_allowed}")
        print(f"   Limit Orders: {account.limit_orders}")
        print(f"   Margin SO Mode: {account.margin_so_mode}")
        print(f"   Trade Mode: {account.trade_mode}")
        print(f"   Trade Expert: {account.trade_expert}")
    else:
        print("❌ Cannot get account info")
    
    # 3. Connection Status Check
    print(f"\n🔌 CONNECTION STATUS:")
    connected = mt5.terminal_info().connected if mt5.terminal_info() else False
    print(f"   MT5 Terminal Connected: {connected}")
    
    if not connected:
        print("❌ CRITICAL: MT5 terminal is NOT connected to trading server!")
        print("   This is why order_send returns None")
        print("   Solution: Check MT5 terminal connection and login")
    
    # 4. Test Symbol Access
    symbol = "Volatility 75 Index"
    print(f"\n📊 SYMBOL TEST: {symbol}")
    
    # Add to Market Watch
    symbol_added = mt5.symbol_select(symbol, True)
    print(f"   Added to Market Watch: {symbol_added}")
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        print(f"   Visible: {symbol_info.visible}")
        print(f"   Trade Mode: {symbol_info.trade_mode}")
        print(f"   Currency Base: {symbol_info.currency_base}")
        print(f"   Currency Profit: {symbol_info.currency_profit}")
        print(f"   Volume Min: {symbol_info.volume_min}")
        print(f"   Volume Max: {symbol_info.volume_max}")
        print(f"   Volume Step: {symbol_info.volume_step}")
        print(f"   Digits: {symbol_info.digits}")
        print(f"   Point: {symbol_info.point}")
        print(f"   Spread: {symbol_info.spread}")
    
    # Get tick
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"   Current Bid: {tick.bid}")
        print(f"   Current Ask: {tick.ask}")
        print(f"   Last Price: {tick.last}")
        print(f"   Volume: {tick.volume}")
        print(f"   Time: {tick.time}")
    else:
        print("   ❌ No tick data available")
    
    # 5. Trading Hours Check
    print(f"\n⏰ TRADING HOURS CHECK:")
    import datetime
    now = datetime.datetime.now()
    print(f"   Current Time: {now}")
    print(f"   Current Day: {now.strftime('%A')}")
    
    # 6. Last Error Check
    error = mt5.last_error()
    print(f"\n❌ LAST ERROR: {error}")
    
    # 7. Minimal Order Test
    print(f"\n🚀 MINIMAL ORDER TEST:")
    print("   Testing if order_send returns None...")
    
    if symbol_info and tick:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": symbol_info.volume_min,
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "deviation": 20,
            "magic": 999999,
            "comment": "ConnTest",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        print(f"   Request prepared: {request}")
        
        # Clear errors
        mt5.last_error()
        
        # Try order
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            print(f"   ❌ order_send returned None (ERROR: {error})")
            
            if error and error[0] == 1:
                print("   🔍 Error Code 1: General error - likely connection issue")
            elif not connected:
                print("   🔍 Root Cause: MT5 terminal not connected to server")
            else:
                print(f"   🔍 Unexpected error: {error}")
        else:
            print(f"   ✅ order_send returned result: {result}")
            print(f"   📊 Return Code: {result.retcode}")
    
    print(f"\n📋 DIAGNOSIS SUMMARY:")
    if not connected:
        print("❌ PRIMARY ISSUE: MT5 Terminal is NOT connected to trading server")
        print("   - order_send will always return None when disconnected")
        print("   - Fix: Check MT5 terminal login and server connection")
    elif not account.trade_allowed:
        print("❌ ACCOUNT ISSUE: Trading not allowed on this account")
    elif not terminal.trade_allowed:
        print("❌ TERMINAL ISSUE: Auto-trading disabled in terminal")
    else:
        print("✅ All basic checks passed - investigate order request structure")
    
    mt5.shutdown()

if __name__ == "__main__":
    diagnose_mt5_trading()
