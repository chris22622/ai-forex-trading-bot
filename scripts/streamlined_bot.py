#!/usr/bin/env python3
"""
STREAMLINED MT5 TRADING BOT
Bypasses complex initialization for immediate trading action
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict

import MetaTrader5 as mt5
import requests

# Configuration
MT5_LOGIN = 31899532
MT5_PASSWORD = "Goon22622$"
MT5_SERVER = "Deriv-Demo"
SYMBOL = "Volatility 75 Index"
LOT_SIZE = 0.01
TRADE_AMOUNT = 1.0
CONFIDENCE_THRESHOLD = 0.50

# Telegram configuration
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# Trading tracking
trade_history = []
starting_balance = 0.0
current_balance = 0.0
total_trades = 0
winning_trades = 0
total_profit = 0.0

def send_telegram_message(message: str):
    """Send detailed message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            print(f"📱 Telegram sent: {message[:50]}...")
        else:
            print(f"❌ Telegram failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def connect_mt5() -> bool:
    """Quick MT5 connection"""
    global starting_balance, current_balance

    print("🔄 Connecting to MT5...")

    if not mt5.initialize():
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return False

    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"❌ Login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    account_info = mt5.account_info()
    if account_info is None:
        print("❌ No account info")
        mt5.shutdown()
        return False

    starting_balance = account_info.balance
    current_balance = account_info.balance

    print(f"✅ Connected: Account {account_info.login}")
    print(f"💰 Starting Balance: ${starting_balance:.2f}")

    return True

def get_current_price() -> float:
    """Get current price"""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return 0.0
    return (tick.ask + tick.bid) / 2

def simple_ai_prediction(prices: list) -> Dict[str, Any]:
    """Enhanced AI prediction with better BUY/SELL logic"""
    if len(prices) < 5:
        return {"prediction": "HOLD", "confidence": 0.3}

    # Calculate multiple indicators
    recent_prices = prices[-10:]  # Last 10 prices

    # 1. Trend analysis
    short_trend = recent_prices[-1] - recent_prices[-3]  # Last 3 ticks
    medium_trend = recent_prices[-1] - recent_prices[-5]  # Last 5 ticks
    long_trend = recent_prices[-1] - recent_prices[0]    # Full period

    # 2. Volatility analysis
    price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
    volatility = sum(abs(change) for change in price_changes) / len(price_changes)

    # 3. Momentum analysis
    momentum = sum(price_changes[-3:]) / 3  # Average of last 3 changes

    # 4. Support/Resistance levels
    recent_high = max(recent_prices)
    recent_low = min(recent_prices)
    current_price = recent_prices[-1]

    # Position in range (0 = at low, 1 = at high)
    range_position = (current_price - recent_low) / max(0.01, recent_high - recent_low)

    # Decision logic
    buy_signals = 0
    sell_signals = 0

    # Trend signals
    if short_trend > 0 and medium_trend > 0:
        buy_signals += 2
    elif short_trend < 0 and medium_trend < 0:
        sell_signals += 2

    # Momentum signals
    if momentum > volatility * 0.5:
        buy_signals += 1
    elif momentum < -volatility * 0.5:
        sell_signals += 1

    # Range signals (contrarian)
    if range_position < 0.3:  # Near support, expect bounce
        buy_signals += 1
    elif range_position > 0.7:  # Near resistance, expect pullback
        sell_signals += 1

    # Volatility-based signals
    if volatility > sum(abs(p - recent_prices[0]) for p in recent_prices) / len(recent_prices):
        # High volatility - trend continuation
        if long_trend > 0:
            buy_signals += 1
        else:
            sell_signals += 1

    # Make decision
    total_signals = buy_signals + sell_signals
    if total_signals == 0:
        return {"prediction": "HOLD", "confidence": 0.4}

    if buy_signals > sell_signals:
        action = "BUY"
        confidence = min(0.9, 0.5 + (buy_signals - sell_signals) * 0.1)
        reason = f"BUY signals: {buy_signals}, trend: {short_trend:.2f}, momentum: {momentum:.2f}"
    elif sell_signals > buy_signals:
        action = "SELL"
        confidence = min(0.9, 0.5 + (sell_signals - buy_signals) * 0.1)
        reason = f"SELL signals: {sell_signals}, trend: {short_trend:.2f}, momentum: {momentum:.2f}"
    else:
        action = "HOLD"
        confidence = 0.4
        reason = f"Signals tied: BUY={buy_signals}, SELL={sell_signals}"

    return {
        "prediction": action,
        "confidence": confidence,
        "reason": reason,
        "signals": {"buy": buy_signals, "sell": sell_signals},
        "metrics": {
            "short_trend": short_trend,
            "momentum": momentum,
            "volatility": volatility,
            "range_position": range_position
        }
    }

def place_trade(action: str) -> bool:
    """Place MT5 trade with detailed tracking"""
    global total_trades, winning_trades, total_profit, current_balance

    try:
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            print(f"❌ Symbol {SYMBOL} not found")
            return False

        price = mt5.symbol_info_tick(SYMBOL)
        if price is None:
            print(f"❌ No price for {SYMBOL}")
            return False

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        price_to_use = price.ask if action == "BUY" else price.bid

        # Get current balance before trade
        account_info = mt5.account_info()
        balance_before = account_info.balance if account_info else current_balance

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOT_SIZE,
            "type": order_type,
            "price": price_to_use,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Bot {action} #{total_trades + 1}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Trade failed: {result.retcode} - {result.comment}")

            # Send failure notification
            failure_msg = "❌ <b>TRADE FAILED</b>\\n"
            failure_msg += f"🔸 Action: {action}\\n"
            failure_msg += f"🔸 Error: {result.comment}\\n"
            failure_msg += f"🔸 Time: {datetime.now().strftime('%H:%M:%S')}"
            send_telegram_message(failure_msg)

            return False

        # Wait a moment for trade to settle
        time.sleep(2)

        # Get updated balance after trade
        account_info = mt5.account_info()
        balance_after = account_info.balance if account_info else balance_before

        # Calculate profit/loss
        trade_profit = balance_after - balance_before
        current_balance = balance_after

        # Update statistics
        total_trades += 1
        total_profit += trade_profit

        if trade_profit > 0:
            winning_trades += 1
            result_emoji = "✅"
            result_text = "PROFIT"
        else:
            result_emoji = "❌"
            result_text = "LOSS"

        # Calculate win rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # Store trade record
        trade_record = {
            "timestamp": datetime.now(),
            "ticket": result.order,
            "action": action,
            "symbol": SYMBOL,
            "volume": LOT_SIZE,
            "price": price_to_use,
            "balance_before": balance_before,
            "balance_after": balance_after,
            "profit": trade_profit,
            "result": result_text
        }
        trade_history.append(trade_record)

        print(f"✅ {action} trade placed! Ticket: {result.order}")
        print(f"💰 Profit/Loss: ${trade_profit:.2f}")
        print(f"📊 Balance: ${balance_before:.2f} → ${balance_after:.2f}")

        # Send detailed Telegram notification
        message = f"{result_emoji} <b>TRADE {result_text}</b>\\n"
        message += "━━━━━━━━━━━━━━━━━━━\\n"
        message += f"🎯 <b>Trade #{total_trades}</b>\\n"
        message += f"🔸 Action: <b>{action}</b>\\n"
        message += f"🔸 Symbol: {SYMBOL}\\n"
        message += f"🔸 Volume: {LOT_SIZE} lots\\n"
        message += f"🔸 Price: <b>{price_to_use:.2f}</b>\\n"
        message += f"🔸 Ticket: #{result.order}\\n"
        message += "━━━━━━━━━━━━━━━━━━━\\n"
        message += f"💰 P&L: <b>${trade_profit:.2f}</b>\\n"
        message += f"💳 Balance: ${balance_before:.2f} → <b>${balance_after:.2f}</b>\\n"
        message += f"📈 Total P&L: <b>${total_profit:.2f}</b>\\n"
        message += f"🎯 Win Rate: <b>{win_rate:.1f}%</b> ({winning_trades}/{total_trades})\\n"

        if total_trades > 1:
            roi = ((current_balance - starting_balance) / starting_balance) * 100
            message += f"📊 Session ROI: <b>{roi:.2f}%</b>\\n"

        message += f"� Time: {datetime.now().strftime('%H:%M:%S')}"

        send_telegram_message(message)

        return True

    except Exception as e:
        print(f"❌ Trade error: {e}")

        # Send error notification
        error_msg = "❌ <b>TRADING ERROR</b>\\n"
        error_msg += f"🔸 Action: {action}\\n"
        error_msg += f"🔸 Error: {str(e)}\\n"
        error_msg += f"🔸 Time: {datetime.now().strftime('%H:%M:%S')}"
        send_telegram_message(error_msg)

        return False

async def main():
    """Main trading loop"""
    print("🚀 STREAMLINED MT5 TRADING BOT")
    print("=" * 50)

    # Connect to MT5
    if not connect_mt5():
        print("❌ MT5 connection failed")
        return

    # Send startup message
    startup_msg = "🚀 <b>ENHANCED TRADING BOT STARTED</b>\\n"
    startup_msg += "━━━━━━━━━━━━━━━━━━━\\n"
    startup_msg += "📊 <b>Account Details:</b>\\n"
    startup_msg += f"🔸 Account: {MT5_LOGIN}\\n"
    startup_msg += f"🔸 Server: {MT5_SERVER}\\n"
    startup_msg += f"🔸 Symbol: {SYMBOL}\\n"
    startup_msg += f"🔸 Volume: {LOT_SIZE} lots\\n"
    startup_msg += "━━━━━━━━━━━━━━━━━━━\\n"
    startup_msg += f"💰 Starting Balance: <b>${starting_balance:.2f}</b>\\n"
    startup_msg += f"🎯 Confidence Threshold: <b>{CONFIDENCE_THRESHOLD*100}%</b>\\n"
    startup_msg += "🧠 AI Mode: <b>Enhanced Multi-Signal</b>\\n"
    startup_msg += "� Trading: <b>BUY & SELL</b>\\n"
    startup_msg += f"🕒 Started: {datetime.now().strftime('%H:%M:%S')}"

    send_telegram_message(startup_msg)

    print("🎯 Starting trading loop...")
    print(f"🎲 Confidence threshold: {CONFIDENCE_THRESHOLD*100}%")

    price_history = []
    trade_count = 0

    try:
        while True:
            # Get current price
            current_price = get_current_price()
            if current_price > 0:
                price_history.append(current_price)

                # Keep only last 20 prices
                if len(price_history) > 20:
                    price_history = price_history[-20:]

                print(f"📊 Price: {current_price:.2f} | History: {len(price_history)} points")

                # Need at least 5 prices for prediction
                if len(price_history) >= 5:
                    # Get AI prediction
                    prediction = simple_ai_prediction(price_history)

                    f"🧠 AI: {prediction['prediction']}"
                    f"(confidence: {prediction['confidence']:.1%})"

                    # Check if we should trade
                    if (prediction['prediction'] in ['BUY', 'SELL'] and
                        prediction['confidence'] >= CONFIDENCE_THRESHOLD):

                        print("🎯 HIGH CONFIDENCE SIGNAL DETECTED!")
                        print(f"🔥 Placing {prediction['prediction']} trade...")

                        if place_trade(prediction['prediction']):
                            trade_count += 1
                            print(f"✅ Trade #{trade_count} completed!")

                            # Wait a bit after trade
                            print("⏰ Waiting 30 seconds before next analysis...")
                            await asyncio.sleep(30)
                        else:
                            print("❌ Trade failed, continuing...")
                    else:
                        print("📊 Waiting for higher confidence signal...")

            # Wait 5 seconds between price checks
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        print("\\n⏹️ Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        mt5.shutdown()
        print("✅ MT5 disconnected")

if __name__ == "__main__":
    asyncio.run(main())
