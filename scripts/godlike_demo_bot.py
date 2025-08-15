"""
🚀 GODLIKE MT5 TRADING BOT - SIMPLIFIED WORKING VERSION
Direct MT5 connection with immediate trading capability
"""

import MetaTrader5 as mt5
import time
import random
import asyncio
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Your MT5 credentials
MT5_LOGIN = 31899532
MT5_PASSWORD = "Goon22622$"
MT5_SERVER = "Deriv-Demo"

# Trading settings
SYMBOL = "Volatility 75 Index"
LOT_SIZE = 0.01
MAGIC_NUMBER = 234000
SLIPPAGE = 3

class SimplifiedMT5Bot:
    def __init__(self):
        self.connected = False
        self.balance = 0.0
        self.running = False
        
    def connect_mt5(self):
        """Connect to MT5 with your credentials"""
        try:
            print("🔄 Connecting to MetaTrader 5...")
            
            # Initialize MT5
            if not mt5.initialize():
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
            
            print("✅ MT5 initialized")
            
            # Login with your credentials
            print(f"🔐 Logging in to {MT5_SERVER}...")
            authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
            
            if not authorized:
                error = mt5.last_error()
                print(f"❌ Login failed: {error}")
                mt5.shutdown()
                return False
            
            print("✅ Login successful!")
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                self.balance = account_info.balance
                print(f"💰 Account Balance: ${self.balance:.2f}")
                print(f"🏦 Account: {account_info.login}")
                print(f"🌐 Server: {account_info.server}")
                print(f"💳 Currency: {account_info.currency}")
                self.connected = True
                return True
            else:
                print("❌ Could not get account info")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_current_price(self):
        """Get current price for the symbol"""
        try:
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick:
                return tick.bid
            return None
        except Exception as e:
            print(f"❌ Error getting price: {e}")
            return None
    
    def place_trade(self, action):
        """Place a simple trade"""
        try:
            current_price = self.get_current_price()
            if not current_price:
                print("❌ Cannot get current price")
                return False
            
            # Determine order type
            if action.upper() == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(SYMBOL).ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(SYMBOL).bid
            
            # Prepare order request with correct filling mode
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": LOT_SIZE,
                "type": order_type,
                "price": price,
                "slippage": SLIPPAGE,
                "magic": MAGIC_NUMBER,
                "comment": "Godlike Bot Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,  # Use confirmed working mode
            }
            
            # Send order
            print(f"📊 Placing {action} order at ${price:.5f}...")
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"❌ Order failed: {result.retcode} - {result.comment}")
                return False
            else:
                print(f"✅ Order successful! Ticket: {result.order}")
                print(f"💰 Volume: {result.volume}")
                print(f"💵 Price: ${result.price:.5f}")
                return True
                
        except Exception as e:
            print(f"❌ Trade error: {e}")
            return False
    
    def simple_ai_decision(self, price_history):
        """Simple AI decision based on price movement"""
        if len(price_history) < 5:
            return "HOLD"
        
        # Simple trend analysis
        recent_prices = price_history[-5:]
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Add some randomness for demo
        confidence = random.uniform(0.6, 0.9)
        
        if trend > 0.0001 and confidence > 0.7:
            return "BUY"
        elif trend < -0.0001 and confidence > 0.7:
            return "SELL"
        else:
            return "HOLD"
    
    async def trading_loop(self):
        """Main trading loop"""
        print("\n🎯 Starting GODLIKE trading loop...")
        print("=" * 50)
        
        price_history = []
        trade_count = 0
        max_trades = 5  # Limit for demo
        
        while self.running and trade_count < max_trades:
            try:
                # Get current price
                current_price = self.get_current_price()
                if current_price:
                    price_history.append(current_price)
                    print(f"📈 Current price: ${current_price:.5f}")
                    
                    # Keep only last 20 prices
                    if len(price_history) > 20:
                        price_history = price_history[-20:]
                    
                    # AI decision
                    decision = self.simple_ai_decision(price_history)
                    
                    if decision in ["BUY", "SELL"]:
                        print(f"🧠 AI Decision: {decision}")
                        
                        # Place trade
                        if self.place_trade(decision):
                            trade_count += 1
                            print(f"📊 Trade #{trade_count} completed")
                            
                            # Update balance
                            account_info = mt5.account_info()
                            if account_info:
                                new_balance = account_info.balance
                                profit = new_balance - self.balance
                                self.balance = new_balance
                                print(f"💰 New Balance: ${self.balance:.2f} (Change: {profit:+.2f})")
                        
                        # Wait between trades
                        print("⏰ Waiting 30 seconds between trades...")
                        await asyncio.sleep(30)
                    else:
                        print(f"🤖 AI Decision: {decision} - No trade")
                        await asyncio.sleep(10)  # Check more frequently when not trading
                
                else:
                    print("⚠️ Could not get price, retrying...")
                    await asyncio.sleep(5)
                    
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(10)
        
        print(f"\n🎉 Demo completed! Total trades: {trade_count}")
        print("🛑 Bot stopping...")
    
    async def run(self):
        """Run the simplified bot"""
        print("=" * 60)
        print("🚀 GODLIKE MT5 TRADING BOT - SIMPLIFIED VERSION")
        print("🤖 Direct MT5 Connection Demo")
        print("=" * 60)
        
        # Connect to MT5
        if not self.connect_mt5():
            print("❌ Cannot start bot - MT5 connection failed")
            return
        
        print("\n✅ All systems ready!")
        print("💎 Symbol: Volatility 75 Index")
        print("💰 Lot Size: 0.01")
        print("🎯 Mode: Demo Trading")
        print("🧠 AI: Simple Trend Analysis")
        
        self.running = True
        
        try:
            await self.trading_loop()
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Bot error: {e}")
        finally:
            self.running = False
            mt5.shutdown()
            print("✅ MT5 connection closed")

async def main():
    bot = SimplifiedMT5Bot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
