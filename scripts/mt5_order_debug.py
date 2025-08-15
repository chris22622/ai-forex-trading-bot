#!/usr/bin/env python3
import MetaTrader5 as mt5

print("=== MT5 Order Debug Test ===")

# Initialize
if not mt5.initialize():
    print(f"❌ Init failed: {mt5.last_error()}")
    exit()

print("✅ MT5 initialized")

# Account check
account = mt5.account_info()
print(f"💰 Balance: ${account.balance:.2f}, Trade: {account.trade_allowed}")

# Symbol setup
symbol = "Volatility 75 Index"
mt5.symbol_select(symbol, True)  # Add to Market Watch

symbol_info = mt5.symbol_info(symbol)
print(f"📊 {symbol} - Visible: {symbol_info.visible}, Mode: {symbol_info.trade_mode}")

# Price check
tick = mt5.symbol_info_tick(symbol)
print(f"💰 Price: {tick.ask}/{tick.bid}")

# Terminal check
terminal = mt5.terminal_info()
print(f"🖥️ Terminal - Connected: {terminal.connected}, Trade: {terminal.trade_allowed}")

# Order test
print("\n🚀 Testing order placement...")
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 0.001,
    "type": mt5.ORDER_TYPE_BUY,
    "price": tick.ask,
    "deviation": 20,
    "magic": 999999,
    "comment": "DebugTest",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

print(f"📋 Request: {request}")

# Clear errors
mt5.last_error()

# Send order
result = mt5.order_send(request)

if result is None:
    error = mt5.last_error()
    print(f"❌ order_send returned None")
    print(f"❌ Last error: {error}")
else:
    print(f"✅ Result received: {result}")
    print(f"📊 Return code: {result.retcode}")

mt5.shutdown()
print("Test complete")
