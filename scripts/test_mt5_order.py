#!/usr/bin/env python3
"""
Quick MT5 Order Test - Diagnose the real issue
"""

import MetaTrader5 as mt5
import time

def test_mt5_order():
    print("🔍 MT5 Order Execution Test")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return
    
    print("✅ MT5 Initialized")
    
    # Check account info
    account = mt5.account_info()
    if account:
        print(f"Account: {account.login}")
        print(f"Balance: ${account.balance}")
        print(f"Trade Mode: {account.trade_mode}")
        print(f"Trade Allowed: {account.trade_allowed}")
    
    # Check terminal info
    terminal = mt5.terminal_info()
    if terminal:
        print(f"Terminal Connected: {terminal.connected}")
        print(f"Trade Allowed: {terminal.trade_allowed}")
        print(f"Expert Advisors Enabled: {terminal.trade_allowed}")
    
    # Get symbol info
    symbol = "Volatility 10 Index"
    symbol_info = mt5.symbol_info(symbol)
    
    if not symbol_info:
        print(f"❌ Symbol {symbol} not found")
        mt5.shutdown()
        return
    
    print(f"\nSymbol: {symbol}")
    print(f"Visible: {symbol_info.visible}")
    print(f"Trade Mode: {symbol_info.trade_mode}")
    print(f"Min Volume: {symbol_info.volume_min}")
    print(f"Max Volume: {symbol_info.volume_max}")
    print(f"Volume Step: {symbol_info.volume_step}")
    
    # Make symbol visible if needed
    if not symbol_info.visible:
        print(f"🔧 Adding {symbol} to Market Watch...")
        if mt5.symbol_select(symbol, True):
            print("✅ Symbol added to Market Watch")
        else:
            print("❌ Failed to add symbol to Market Watch")
            mt5.shutdown()
            return
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print("❌ Cannot get current price")
        mt5.shutdown()
        return
    
    print(f"Current Price: {tick.bid}/{tick.ask}")
    
    # Prepare order request
    lot_size = symbol_info.volume_min  # Use minimum volume
    price = tick.ask  # For BUY order
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": 0.0,
        "tp": 0.0,
        "deviation": 10,
        "magic": 234000,
        "comment": "Test order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print(f"\nTest Order Details:")
    print(f"Action: BUY")
    print(f"Volume: {lot_size}")
    print(f"Price: {price}")
    print(f"Magic: {request['magic']}")
    
    # Check order before sending
    result = mt5.order_check(request)
    if result is None:
        print("❌ Order check returned None")
        error = mt5.last_error()
        print(f"Last Error: {error}")
    else:
        print(f"Order Check Result: {result.retcode}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Order check failed: {result.retcode}")
            print(f"Comment: {getattr(result, 'comment', 'No comment')}")
        else:
            print("✅ Order check passed")
    
    # Try to send order (in demo mode)
    print("\n🎯 Attempting to send order...")
    
    # Clear any previous errors
    mt5.last_error()
    
    # Send order
    send_result = mt5.order_send(request)
    
    if send_result is None:
        print("❌ Order send returned None")
        error = mt5.last_error()
        print(f"Last Error: {error}")
    else:
        print(f"Order Send Result: {send_result.retcode}")
        if send_result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Order successful! Ticket: {send_result.order}")
        else:
            print(f"❌ Order failed: {send_result.retcode}")
            print(f"Comment: {getattr(send_result, 'comment', 'No comment')}")
            
            # Try different filling types
            print("\n🔄 Trying different order filling types...")
            
            filling_types = [
                ("IOC", mt5.ORDER_FILLING_IOC),
                ("FOK", mt5.ORDER_FILLING_FOK), 
                ("RETURN", mt5.ORDER_FILLING_RETURN)
            ]
            
            for name, filling in filling_types:
                print(f"  Trying {name}...")
                request["type_filling"] = filling
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"  ✅ {name} worked! Ticket: {result.order}")
                    break
                elif result:
                    print(f"  ❌ {name} failed: {result.retcode}")
                else:
                    print(f"  ❌ {name} returned None")
    
    mt5.shutdown()
    print("\n✅ Test completed")

if __name__ == "__main__":
    test_mt5_order()
