"""
Quick test to check supported filling modes for Volatility 75 Index
"""
import MetaTrader5 as mt5

print("🔄 Testing MT5 filling modes...")

# Initialize and login
if not mt5.initialize():
    print("❌ Failed to initialize MT5")
    exit(1)

# Login to your account
authorized = mt5.login(31899532, password="Goon22622$", server="Deriv-Demo")
if not authorized:
    print(f"❌ Login failed: {mt5.last_error()}")
    exit(1)

print("✅ Connected to MT5")

# Check symbol info for Volatility 75 Index
symbol = "Volatility 75 Index"
symbol_info = mt5.symbol_info(symbol)

if symbol_info is None:
    print(f"❌ Symbol {symbol} not found")
    mt5.shutdown()
    exit(1)

print(f"📊 Symbol: {symbol}")
print(f"   Spread: {symbol_info.spread}")
print(f"   Min lot: {symbol_info.volume_min}")
print(f"   Max lot: {symbol_info.volume_max}")
print(f"   Lot step: {symbol_info.volume_step}")

# Check supported filling modes
print("\n📋 Testing filling modes directly:")

# For Deriv demo accounts, let's try the most common modes
print("� Testing ORDER_FILLING_FOK...")
test_request_fok = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 0.01,
    "type": mt5.ORDER_TYPE_BUY,
    "price": mt5.symbol_info_tick(symbol).ask,
    "deviation": 10,
    "magic": 234000,
    "comment": "Test FOK",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_FOK,
}

result_fok = mt5.order_send(test_request_fok)
if result_fok and result_fok.retcode == mt5.TRADE_RETCODE_DONE:
    print("✅ ORDER_FILLING_FOK works!")
    print("   Best filling mode: ORDER_FILLING_FOK")
else:
    print(f"❌ ORDER_FILLING_FOK failed: {result_fok.retcode if result_fok else 'No result'}")

    print("🔄 Testing ORDER_FILLING_IOC...")
    test_request_ioc = test_request_fok.copy()
    test_request_ioc["type_filling"] = mt5.ORDER_FILLING_IOC

    result_ioc = mt5.order_send(test_request_ioc)
    if result_ioc and result_ioc.retcode == mt5.TRADE_RETCODE_DONE:
        print("✅ ORDER_FILLING_IOC works!")
        print("   Best filling mode: ORDER_FILLING_IOC")
    else:
        print(f"❌ ORDER_FILLING_IOC failed: {result_ioc.retcode if result_ioc else 'No result'}")

        print("🔄 Testing ORDER_FILLING_RETURN...")
        test_request_return = test_request_fok.copy()
        test_request_return["type_filling"] = mt5.ORDER_FILLING_RETURN

        result_return = mt5.order_send(test_request_return)
        if result_return and result_return.retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ ORDER_FILLING_RETURN works!")
            print("   Best filling mode: ORDER_FILLING_RETURN")
        else:
                        print(f"❌ ORDER_FILLING_RETURN failed: {result_return.retcode "
            "if result_return else 'No result'}")

mt5.shutdown()
print("✅ Test completed")
