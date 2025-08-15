#!/usr/bin/env python3
"""
🚀 DIRECT MT5 TRADING - IMMEDIATE EXECUTION
Bypasses all complex initialization for instant trading
"""

import logging
import random
import time

import MetaTrader5 as mt5

from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def init_mt5():
    """Initialize MT5 connection"""
    if not mt5.initialize():
        logger.error("❌ MT5 initialization failed")
        logger.error(f"   Error: {mt5.last_error()}")
        return False

    logger.info("✅ MT5 initialized successfully")

    # Login using config credentials
    login = MT5_LOGIN
    password = MT5_PASSWORD
    server = MT5_SERVER

    logger.info(f"🔐 Attempting login: {login} on {server}")

    authorized = mt5.login(login, password=password, server=server)
    if authorized:
        logger.info(f"✅ Connected to MT5 account: {login}")

        # Get account info
        account = mt5.account_info()
        if account:
            logger.info(f"💰 Account Balance: ${account.balance:.2f}")
            logger.info(f"💵 Account Equity: ${account.equity:.2f}")

        return True
    else:
        logger.error(f"❌ Failed to connect to MT5 account: {login}")
        logger.error(f"   Error: {mt5.last_error()}")
        return False

def get_current_price(symbol="R_75"):
    """Get current price for symbol"""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"❌ Failed to get tick for {symbol}")
        return None

    price = (tick.bid + tick.ask) / 2
    logger.info(f"💰 Current {symbol} price: {price:.5f}")
    return price

def place_trade(symbol="R_75", action="BUY", volume=0.1):
    """Place a trade immediately"""
    logger.info(f"🚀 PLACING {action} TRADE: {symbol} - Volume: {volume}")

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("❌ Cannot get current price")
        return False

    # Set up trade request
    if action == "BUY":
        trade_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - 0.001  # Stop loss
        tp = price + 0.002  # Take profit (2:1 ratio)
    else:  # SELL
        trade_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + 0.001  # Stop loss
        tp = price - 0.002  # Take profit (2:1 ratio)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "🚀 DIRECT TRADE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Send trade
    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"❌ Trade failed: {result.comment}")
        return False

    logger.info("✅ TRADE SUCCESSFUL!")
    logger.info(f"   Order: {result.order}")
    logger.info(f"   Price: {price:.5f}")
    logger.info(f"   SL: {sl:.5f}")
    logger.info(f"   TP: {tp:.5f}")

    return True

def main():
    """Main trading loop"""
    print("🚀 DIRECT MT5 TRADING - STARTING NOW!")
    print("💰 ULTRA-AGGRESSIVE MODE - IMMEDIATE TRADES!")

    # Initialize MT5
    if not init_mt5():
        print("❌ Cannot start trading - MT5 failed")
        return

    print("🔥 STARTING AGGRESSIVE TRADING LOOP!")

    trade_count = 0

    try:
        while trade_count < 5:  # Limit for safety
            # Get price
            price = get_current_price()
            if price is None:
                time.sleep(1)
                continue

            # Random action for immediate trading
            action = random.choice(["BUY", "SELL"])

            logger.info(f"🎯 Trade #{trade_count + 1}: {action}")

            # Place trade
            if place_trade(action=action):
                trade_count += 1
                logger.info(f"💎 TRADE #{trade_count} COMPLETED!")

                # Wait 10 seconds between trades
                logger.info("⏰ Waiting 10 seconds...")
                time.sleep(10)
            else:
                logger.error("❌ Trade failed, retrying...")
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")

    finally:
        mt5.shutdown()
        logger.info("🔚 MT5 connection closed")

if __name__ == "__main__":
    main()
