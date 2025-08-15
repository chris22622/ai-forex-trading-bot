#!/usr/bin/env python3
"""
Quick MT5 Diagnostic - Check connection and order capabilities
"""


import MetaTrader5 as mt5


def check_mt5_comprehensive():
    print("🔍 MT5 COMPREHENSIVE DIAGNOSTIC")
    print("=" * 50)

    # 1. Initialize MT5
    print("\n1. MT5 INITIALIZATION:")
    init_result = mt5.initialize()
    print(f"   Initialize result: {init_result}")

    if not init_result:
        print(f"   ❌ Initialization failed: {mt5.last_error()}")
        return

    # 2. Version info
    print(f"\n2. MT5 VERSION: {mt5.version()}")

    # 3. Terminal info
    print("\n3. TERMINAL INFO:")
    terminal = mt5.terminal_info()
    if terminal:
        print(f"   Name: {terminal.name}")
        print(f"   Version: {terminal.version}")
        print(f"   Build: {terminal.build}")
        print(f"   Path: {terminal.path}")
        print(f"   Data path: {terminal.data_path}")
        print(f"   Common path: {terminal.commondata_path}")
        print(f"   Language: {terminal.language}")
        print(f"   Company: {terminal.company}")
        print(f"   Connected: {terminal.connected}")
        print(f"   DLL allowed: {terminal.dlls_allowed}")
        print(f"   Trade allowed: {terminal.trade_allowed}")
        print(f"   Trade context busy: {terminal.tradeapi_disabled}")
        print(f"   Mail enabled: {terminal.mail_enabled}")
        print(f"   News enabled: {terminal.news_enabled}")
        print(f"   FTP enabled: {terminal.ftp_enabled}")
        print(f"   Notifications enabled: {terminal.notifications_enabled}")
        print(f"   Email: {terminal.email}")
        print(f"   FTP server: {terminal.ftp_server}")
        print(f"   FTP login: {terminal.ftp_login}")
        print(f"   FTP password: {terminal.ftp_password}")
        print(f"   News languages: {terminal.news_languages}")
        print(f"   Ping last: {terminal.ping_last}")
        print(f"   Community account: {terminal.community_account}")
        print(f"   Community connection: {terminal.community_connection}")
        print(f"   Community balance: {terminal.community_balance}")
        print(f"   Community state: {terminal.community_state}")
        print(f"   CPU cores: {terminal.cpu_cores}")
        print(f"   Physical memory: {terminal.physical_memory}")
        print(f"   Total memory: {terminal.total_memory}")
        print(f"   Disk space: {terminal.disk_space}")
        print(f"   Screen DPI: {terminal.screen_dpi}")
    else:
        print(f"   ❌ Terminal info failed: {mt5.last_error()}")

    # 4. Account info
    print("\n4. ACCOUNT INFO:")
    account = mt5.account_info()
    if account:
        print(f"   Login: {account.login}")
        print(f"   Trade mode: {account.trade_mode}")
        print(f"   Leverage: {account.leverage}")
        print(f"   Limit orders: {account.limit_orders}")
        print(f"   Margin so mode: {account.margin_so_mode}")
        print(f"   Trade allowed: {account.trade_allowed}")
        print(f"   Trade expert: {account.trade_expert}")
        print(f"   Margin mode: {account.margin_mode}")
        print(f"   Currency digits: {account.currency_digits}")
        print(f"   Fifo close: {account.fifo_close}")
        print(f"   Balance: {account.balance}")
        print(f"   Credit: {account.credit}")
        print(f"   Profit: {account.profit}")
        print(f"   Equity: {account.equity}")
        print(f"   Margin: {account.margin}")
        print(f"   Margin free: {account.margin_free}")
        print(f"   Margin level: {account.margin_level}")
        print(f"   Margin so call: {account.margin_so_call}")
        print(f"   Margin so so: {account.margin_so_so}")
        print(f"   Currency: {account.currency}")
        print(f"   Server: {account.server}")
        print(f"   Name: {account.name}")
        print(f"   Company: {account.company}")
    else:
        print(f"   ❌ Account info failed: {mt5.last_error()}")

    # 5. Check symbol info
    print("\n5. SYMBOL INFO (Volatility 75 Index):")
    symbol = "Volatility 75 Index"
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        print(f"   Name: {symbol_info.name}")
        print(f"   Custom: {symbol_info.custom}")
        print(f"   Chart mode: {symbol_info.chart_mode}")
        print(f"   Select: {symbol_info.select}")
        print(f"   Visible: {symbol_info.visible}")
        print(f"   Session deals: {symbol_info.session_deals}")
        print(f"   Session buy orders: {symbol_info.session_buy_orders}")
        print(f"   Session sell orders: {symbol_info.session_sell_orders}")
        print(f"   Volume: {symbol_info.volume}")
        print(f"   Volume high: {symbol_info.volumehigh}")
        print(f"   Volume low: {symbol_info.volumelow}")
        print(f"   Time: {symbol_info.time}")
        print(f"   Time msc: {symbol_info.time_msc}")
        print(f"   Digits: {symbol_info.digits}")
        print(f"   Spread: {symbol_info.spread}")
        print(f"   Spread float: {symbol_info.spread_float}")
        print(f"   Ticks value: {symbol_info.ticks_value}")
        print(f"   Ticks size: {symbol_info.ticks_size}")
        print(f"   Bid: {symbol_info.bid}")
        print(f"   Bidhigh: {symbol_info.bidhigh}")
        print(f"   Bidlow: {symbol_info.bidlow}")
        print(f"   Ask: {symbol_info.ask}")
        print(f"   Askhigh: {symbol_info.askhigh}")
        print(f"   Asklow: {symbol_info.asklow}")
        print(f"   Last: {symbol_info.last}")
        print(f"   Lasthigh: {symbol_info.lasthigh}")
        print(f"   Lastlow: {symbol_info.lastlow}")
        print(f"   Volume real: {symbol_info.volume_real}")
        print(f"   Volume high real: {symbol_info.volumehigh_real}")
        print(f"   Volume low real: {symbol_info.volumelow_real}")
        print(f"   Option strike: {symbol_info.option_strike}")
        print(f"   Point: {symbol_info.point}")
        print(f"   Trade calc mode: {symbol_info.trade_calc_mode}")
        print(f"   Trade mode: {symbol_info.trade_mode}")
        print(f"   Start time: {symbol_info.start_time}")
        print(f"   Expiration time: {symbol_info.expiration_time}")
        print(f"   Trade stops level: {symbol_info.trade_stops_level}")
        print(f"   Trade freeze level: {symbol_info.trade_freeze_level}")
        print(f"   Trade execution mode: {symbol_info.trade_exemode}")
        print(f"   Swap mode: {symbol_info.swap_mode}")
        print(f"   Swap rollover3days: {symbol_info.swap_rollover3days}")
        print(f"   Margin hedged use leg: {symbol_info.margin_hedged_use_leg}")
        print(f"   Expiration mode: {symbol_info.expiration_mode}")
        print(f"   Filling mode: {symbol_info.filling_mode}")
        print(f"   Order mode: {symbol_info.order_mode}")
        print(f"   Order GTC mode: {symbol_info.order_gtc_mode}")
        print(f"   Option mode: {symbol_info.option_mode}")
        print(f"   Option right: {symbol_info.option_right}")
        print(f"   Bid: {symbol_info.bid}")
        print(f"   Ask: {symbol_info.ask}")
        print(f"   Last: {symbol_info.last}")
        print(f"   Volume min: {symbol_info.volume_min}")
        print(f"   Volume max: {symbol_info.volume_max}")
        print(f"   Volume step: {symbol_info.volume_step}")
        print(f"   Volume limit: {symbol_info.volume_limit}")
        print(f"   Swap long: {symbol_info.swap_long}")
        print(f"   Swap short: {symbol_info.swap_short}")
        print(f"   Margin initial: {symbol_info.margin_initial}")
        print(f"   Margin maintenance: {symbol_info.margin_maintenance}")
        print(f"   Session volume: {symbol_info.session_volume}")
        print(f"   Session turnover: {symbol_info.session_turnover}")
        print(f"   Session interest: {symbol_info.session_interest}")
        print(f"   Session buy orders volume: {symbol_info.session_buy_orders_volume}")
        print(f"   Session sell orders volume: {symbol_info.session_sell_orders_volume}")
        print(f"   Session open: {symbol_info.session_open}")
        print(f"   Session close: {symbol_info.session_close}")
        print(f"   Session AW: {symbol_info.session_aw}")
        print(f"   Session price settlement: {symbol_info.session_price_settlement}")
        print(f"   Session price limit min: {symbol_info.session_price_limit_min}")
        print(f"   Session price limit max: {symbol_info.session_price_limit_max}")
        print(f"   Margin hedged: {symbol_info.margin_hedged}")
        print(f"   Price change: {symbol_info.price_change}")
        print(f"   Price volatility: {symbol_info.price_volatility}")
        print(f"   Price theoretical: {symbol_info.price_theoretical}")
        print(f"   Price greeks delta: {symbol_info.price_greeks_delta}")
        print(f"   Price greeks theta: {symbol_info.price_greeks_theta}")
        print(f"   Price greeks gamma: {symbol_info.price_greeks_gamma}")
        print(f"   Price greeks vega: {symbol_info.price_greeks_vega}")
        print(f"   Price greeks rho: {symbol_info.price_greeks_rho}")
        print(f"   Price greeks omega: {symbol_info.price_greeks_omega}")
        print(f"   Price sensitivity: {symbol_info.price_sensitivity}")
        print(f"   Basis: {symbol_info.basis}")
        print(f"   Category: {symbol_info.category}")
        print(f"   Currency base: {symbol_info.currency_base}")
        print(f"   Currency profit: {symbol_info.currency_profit}")
        print(f"   Currency margin: {symbol_info.currency_margin}")
        print(f"   Bank: {symbol_info.bank}")
        print(f"   Description: {symbol_info.description}")
        print(f"   Exchange: {symbol_info.exchange}")
        print(f"   Formula: {symbol_info.formula}")
        print(f"   ISIN: {symbol_info.isin}")
        print(f"   Name: {symbol_info.name}")
        print(f"   Page: {symbol_info.page}")
        print(f"   Path: {symbol_info.path}")
    else:
        print(f"   ❌ Symbol info failed: {mt5.last_error()}")

    # 6. Test order send with minimal parameters
    print("\n6. TEST ORDER SEND:")
    if symbol_info:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.001,
            "type": mt5.ORDER_TYPE_SELL,
            "price": symbol_info.bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "MT5 diagnostic test",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"   Test request: {request}")
        result = mt5.order_send(request)
        print(f"   Order result: {result}")
        print(f"   Last error: {mt5.last_error()}")

        if result is None:
            print("   ❌ order_send returned None - this is the issue!")

            # Try different filling modes
            for filling_mode in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                request["type_filling"] = filling_mode
                print(f"   Trying filling mode {filling_mode}...")
                result = mt5.order_send(request)
                print(f"   Result: {result}")
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"   ✅ Success with filling mode {filling_mode}!")
                    break

    # 7. Check positions
    print("\n7. CURRENT POSITIONS:")
    positions = mt5.positions_get()
    if positions:
        print(f"   Found {len(positions)} positions")
        for pos in positions:
            print(f"   Position: {pos.symbol} {pos.type} {pos.volume} @ {pos.price_open}")
    else:
        print("   No positions found")

    # 8. Check orders
    print("\n8. CURRENT ORDERS:")
    orders = mt5.orders_get()
    if orders:
        print(f"   Found {len(orders)} pending orders")
        for order in orders:
            print(f"   Order: {order.symbol} {order.type} {order.volume} @ {order.price_open}")
    else:
        print("   No pending orders found")

    print(f"\n9. FINAL ERROR CHECK: {mt5.last_error()}")

    # Cleanup
    mt5.shutdown()
    print("\n✅ Diagnostic complete!")

if __name__ == "__main__":
    check_mt5_comprehensive()
