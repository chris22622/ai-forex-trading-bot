#!/usr/bin/env python3
"""
Direct MT5 Order Placement Test - Sync version
"""
import MetaTrader5 as mt5
import time

def test_mt5_orders():
    print("🚀 Testing MT5 Direct Order Placement...")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return
    
    print("✅ MT5 initialized")
    
    # Account info
    account = mt5.account_info()
    print(f"💰 Balance: ${account.balance:.2f}, Trade Allowed: {account.trade_allowed}")
    
    # Symbol setup
    symbol = "Volatility 75 Index"
    
    # Make symbol visible
    print(f"🔧 Ensuring {symbol} is visible...")
    if mt5.symbol_select(symbol, True):
        print(f"✅ {symbol} added to Market Watch")
    else:
        print(f"❌ Failed to add {symbol}: {mt5.last_error()}")
        return
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"❌ Symbol info failed: {mt5.last_error()}")
        return
        
    print(f"📊 Symbol Info:")
    print(f"   Visible: {symbol_info.visible}")
    print(f"   Trade Mode: {symbol_info.trade_mode}")
    print(f"   Volume Min: {symbol_info.volume_min}")
    print(f"   Filling Modes: {symbol_info.filling_mode}")
    
    # Get price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"❌ No tick data: {mt5.last_error()}")
        return
        
    print(f"💰 Current Price: {tick.ask} / {tick.bid}")
    
    # Clear previous errors
    mt5.last_error()
    
    # Create order request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": symbol_info.volume_min,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "sl": 0.0,
        "tp": 0.0,
        "deviation": 20,
        "magic": 123456,
        "comment": f"DirectTest_{int(time.time())}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print(f"📋 Order Request:")
    for k, v in request.items():
        print(f"   {k}: {v}")
    
    # Send order
    print(f"🚀 Sending order...")
    result = mt5.order_send(request)
    
    # Check result
    if result is None:
        error = mt5.last_error()
        print(f"❌ order_send returned None!")
        print(f"❌ Last Error: {error}")
        
        # More diagnostics
        terminal = mt5.terminal_info()
        if terminal:
            print(f"🔍 Terminal Connected: {terminal.connected}")
            print(f"🔍 Terminal Trade Allowed: {terminal.trade_allowed}")
            
        # Check symbol again
        symbol_check = mt5.symbol_info(symbol)
        if symbol_check:
            print(f"🔍 Symbol Still Visible: {symbol_check.visible}")
            print(f"🔍 Symbol Trade Mode: {symbol_check.trade_mode}")
        
    else:
        print(f"✅ Got result object!")
        print(f"📊 Return Code: {result.retcode}")
        if hasattr(result, 'order'):
            print(f"📊 Order ID: {result.order}")
        if hasattr(result, 'comment'):
            print(f"📊 Comment: {result.comment}")
            
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"🎉 SUCCESS! Order placed successfully!")
        else:
            print(f"⚠️ Order failed with code: {result.retcode}")
    
    mt5.shutdown()
    print("✅ Test completed")

if __name__ == "__main__":
    test_mt5_orders()
