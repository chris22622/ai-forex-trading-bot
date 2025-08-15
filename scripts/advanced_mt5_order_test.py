#!/usr/bin/env python3
"""
Advanced MT5 Order Testing and Debugging
"""
import asyncio
import MetaTrader5 as mt5
import time
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_mt5_order_placement():
    """Test MT5 order placement with detailed debugging"""
    print("🔍 Starting comprehensive MT5 order test...")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return
        
    print("✅ MT5 initialized successfully")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        print(f"❌ Failed to get account info: {mt5.last_error()}")
        return
        
    print(f"💰 Account Balance: ${account_info.balance:.2f}")
    print(f"🔒 Trade Allowed: {account_info.trade_allowed}")
    
    # Test symbol
    symbol = "Volatility 75 Index"
    print(f"\n📊 Testing symbol: {symbol}")
    
    # Check if symbol exists
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found")
        return
        
    print(f"✅ Symbol found: {symbol}")
    print(f"📈 Visible in Market Watch: {symbol_info.visible}")
    print(f"🎯 Trade Mode: {symbol_info.trade_mode}")
    print(f"💱 Currency Base: {symbol_info.currency_base}")
    print(f"📏 Volume Min: {symbol_info.volume_min}")
    print(f"📏 Volume Max: {symbol_info.volume_max}")
    print(f"📏 Volume Step: {symbol_info.volume_step}")
    
    # Ensure symbol is visible
    if not symbol_info.visible:
        print(f"🔧 Adding {symbol} to Market Watch...")
        if mt5.symbol_select(symbol, True):
            print(f"✅ {symbol} added to Market Watch")
            symbol_info = mt5.symbol_info(symbol)  # Refresh
        else:
            print(f"❌ Failed to add {symbol} to Market Watch: {mt5.last_error()}")
            return
    
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ Cannot get price for {symbol}: {mt5.last_error()}")
        return
        
    print(f"💰 Current Price - Bid: {tick.bid}, Ask: {tick.ask}")
    
    # Test terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"🖥️ Terminal - Trade Allowed: {terminal_info.trade_allowed}")
        print(f"🖥️ Terminal - Connected: {terminal_info.connected}")
    
    # Create a minimal test order
    print(f"\n🎯 Attempting test BUY order...")
    
    # Clear any previous errors
    mt5.last_error()
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": symbol_info.volume_min,  # Use minimum volume
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "sl": 0.0,
        "tp": 0.0,
        "deviation": 20,
        "magic": 123456,
        "comment": f"MT5Test_{int(time.time())}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print(f"📋 Order Request:")
    for key, value in request.items():
        print(f"   {key}: {value}")
    
    # Send order
    print(f"\n🚀 Sending order...")
    result = mt5.order_send(request)
    
    if result is None:
        last_error = mt5.last_error()
        print(f"❌ order_send returned None. Last error: {last_error}")
        
        # Try to get more error details
        if last_error:
            error_code, error_desc = last_error
            print(f"❌ Error Code: {error_code}")
            print(f"❌ Error Description: {error_desc}")
            
        # Test with different filling type
        print(f"\n🔄 Trying with different filling type...")
        request["type_filling"] = mt5.ORDER_FILLING_FOK
        result = mt5.order_send(request)
        
        if result is None:
            last_error = mt5.last_error()
            print(f"❌ Second attempt also failed: {last_error}")
            
            # Try market execution
            print(f"\n🔄 Trying with market execution...")
            request["type_filling"] = mt5.ORDER_FILLING_RETURN
            result = mt5.order_send(request)
            
            if result is None:
                last_error = mt5.last_error()
                print(f"❌ All attempts failed: {last_error}")
            else:
                print(f"✅ SUCCESS with market execution!")
                print(f"📊 Result: {result}")
        else:
            print(f"✅ SUCCESS with FOK filling!")
            print(f"📊 Result: {result}")
    else:
        print(f"✅ SUCCESS on first attempt!")
        print(f"📊 Result: {result}")
        if hasattr(result, 'retcode'):
            print(f"📊 Return Code: {result.retcode}")
        if hasattr(result, 'order'):
            print(f"📊 Order Ticket: {result.order}")
    
    print(f"\n🔍 Final MT5 status check...")
    final_error = mt5.last_error()
    print(f"📊 Final error state: {final_error}")
    
    mt5.shutdown()
    print(f"✅ MT5 test completed")

if __name__ == "__main__":
    asyncio.run(test_mt5_order_placement())
