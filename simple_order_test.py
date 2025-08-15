#!/usr/bin/env python3
"""
Simple MT5 Order Test with Fixed Parameters
"""

import MetaTrader5 as mt5
import time

print("🔧 FIXED MT5 Order Test")
print("=" * 40)

# Kill any existing MT5 processes first
try:
    mt5.shutdown()
    time.sleep(1)
except:
    pass

# Initialize MT5
if not mt5.initialize():
    print("❌ MT5 initialization failed")
    exit(1)

print("✅ MT5 Initialized")

symbol = "Volatility 10 Index"
symbol_info = mt5.symbol_info(symbol)

if not symbol_info:
    print(f"❌ Symbol {symbol} not found")
    mt5.shutdown()
    exit(1)

print(f"Symbol: {symbol}")
print(f"  Min Volume: {symbol_info.volume_min}")
print(f"  Max Volume: {symbol_info.volume_max}")
print(f"  Volume Step: {symbol_info.volume_step}")
print(f"  Visible: {symbol_info.visible}")

# Ensure symbol is visible
if not symbol_info.visible:
    mt5.symbol_select(symbol, True)
    print("  Symbol added to Market Watch")

# Get current tick
tick = mt5.symbol_info_tick(symbol)
if not tick:
    print("❌ No tick data available")
    mt5.shutdown()
    exit(1)

print(f"Current Price: {tick.ask} (ask) / {tick.bid} (bid)")

# Use the exact parameters our bot should use
volume = symbol_info.volume_min  # 0.5 for this symbol
price = tick.ask
magic = 234000
deviation = 3

# Create request with FOK filling (which we know works)
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": volume,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "sl": 0.0,
    "tp": 0.0,
    "deviation": deviation,
    "magic": magic,
    "comment": f"FixedBot_{int(time.time())}",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_FOK,
}

print(f"\nOrder Parameters:")
print(f"  Volume: {volume}")
print(f"  Price: {price}")
print(f"  Magic: {magic}")
print(f"  Deviation: {deviation}")
print(f"  Filling: FOK")

# Clear any previous errors
mt5.last_error()

print("\n🎯 Sending order...")
result = mt5.order_send(request)

if result is None:
    error = mt5.last_error()
    print(f"❌ Order send returned None")
    print(f"   Last Error: {error}")
else:
    print(f"Order Result:")
    print(f"  Return Code: {result.retcode}")
    
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"  🎉 SUCCESS! Ticket: {result.order}")
        print(f"  Volume: {result.volume}")
        print(f"  Price: {result.price}")
    else:
        print(f"  ❌ FAILED: {result.retcode}")
        print(f"  Comment: {getattr(result, 'comment', 'No comment')}")

mt5.shutdown()
print("\n✅ Test completed")
