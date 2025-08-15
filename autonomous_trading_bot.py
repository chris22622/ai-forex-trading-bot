#!/usr/bin/env python3
"""
Autonomous Trading Bot - Completely Self-Running with Smart Simulation
Designed to work 24/7 without any user input, automatically handling all errors
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Set up autonomous logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - AUTONOMOUS - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/autonomous_bot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the main bot with error handling
try:
    from main import DerivTradingBot
    logger.info("✅ Main bot imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import main bot: {e}")
    sys.exit(1)

class AutonomousTradingBot:
    """Wrapper for the main bot with autonomous error handling and recovery"""
    
    def __init__(self):
        self.bot: Optional[DerivTradingBot] = None
        self.running = True
        self.restart_count = 0
        self.max_restarts = 10
        
    async def initialize_bot(self) -> bool:
        """Initialize the trading bot with error handling"""
        try:
            logger.info("🤖 Initializing Autonomous Trading Bot...")
            
            # Force simulation mode for full autonomy
            self.bot = DerivTradingBot()
            
            # Override settings for autonomous operation
            self.bot.using_mt5 = False  # Start in simulation mode
            self.bot.connected = True
            self.bot.current_balance = 1000.0  # Demo balance
            self.bot.force_demo_mode = True
            self.bot.force_paper_trading = True
            
            logger.info("✅ Bot initialized in autonomous simulation mode")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            return False
    
    async def run_autonomous_loop(self) -> None:
        """Main autonomous trading loop"""
        logger.info("🚀 Starting Autonomous Trading Loop")
        
        while self.running and self.restart_count < self.max_restarts:
            try:
                if not self.bot:
                    if not await self.initialize_bot():
                        await asyncio.sleep(10)
                        continue
                
                # Start the bot's simulation trading loop directly
                await self.run_smart_simulation()
                
            except KeyboardInterrupt:
                logger.info("⏸️ Keyboard interrupt - shutting down gracefully")
                self.running = False
                break
                
            except Exception as e:
                logger.error(f"❌ Autonomous loop error: {e}")
                self.restart_count += 1
                
                if self.restart_count < self.max_restarts:
                    logger.info(f"🔄 Restarting bot (attempt {self.restart_count}/{self.max_restarts})")
                    self.bot = None
                    await asyncio.sleep(5)
                else:
                    logger.error("❌ Maximum restart attempts reached")
                    break
    
    async def run_smart_simulation(self) -> None:
        """Run intelligent simulation that mimics real trading"""
        logger.info("🎮 Starting Smart Simulation Mode")
        
        # Initialize price history with realistic starting data
        if not self.bot.price_history:
            base_price = 1000.0  # Starting price for Volatility 75 Index
            for i in range(50):  # Give it some history to work with
                price_change = (i - 25) * 0.1  # Gradual price movement
                self.bot.price_history.append(base_price + price_change)
        
        trade_counter = 0
        last_trade_time = datetime.now()
        
        while self.running:
            try:
                # Generate realistic price data
                await self.generate_smart_price_data()
                
                # Check if we should analyze for trading
                if len(self.bot.price_history) >= 20:
                    # Run AI analysis and potentially place trades
                    await self.smart_trading_analysis()
                
                # Monitor existing trades
                await self.bot.check_trade_results()
                
                # Send periodic status updates
                trade_counter += 1
                if trade_counter % 100 == 0:  # Every 100 cycles
                    await self.send_status_update()
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Smart simulation error: {e}")
                await asyncio.sleep(5)
    
    async def generate_smart_price_data(self) -> None:
        """Generate realistic price movements"""
        try:
            if not self.bot.price_history:
                base_price = 1000.0
            else:
                base_price = self.bot.price_history[-1]
            
            # Create realistic volatility patterns
            import random
            import math
            
            # Add trend and volatility
            trend = random.uniform(-0.001, 0.001)  # Small trend
            volatility = random.uniform(0.002, 0.008)  # Realistic volatility
            
            # Use sine wave for some periodicity
            time_factor = len(self.bot.price_history) * 0.1
            sine_factor = math.sin(time_factor) * 0.002
            
            # Combine factors
            change = trend + (random.gauss(0, volatility)) + sine_factor
            new_price = base_price * (1 + change)
            
            # Keep price in reasonable bounds
            new_price = max(900.0, min(1100.0, new_price))
            
            self.bot.price_history.append(new_price)
            
            # Update symbol data
            symbol = self.bot.get_effective_symbol()
            if symbol not in self.bot.symbol_data:
                self.bot.symbol_data[symbol] = {}
            
            self.bot.symbol_data[symbol].update({
                'last_price': new_price,
                'timestamp': datetime.now(),
                'bid': new_price - 0.1,
                'ask': new_price + 0.1
            })
            
            # Keep only recent history
            if len(self.bot.price_history) > 1000:
                self.bot.price_history = self.bot.price_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error generating price data: {e}")
    
    async def smart_trading_analysis(self) -> None:
        """Perform smart trading analysis and execution"""
        try:
            # Check basic trading conditions
            if not await self.bot.should_trade():
                return
            
            # Get AI prediction
            prediction = await self.bot.get_ai_prediction()
            if not prediction:
                return
            
            action = prediction['action']
            confidence = prediction['confidence']
            
            # Only trade with high confidence
            if confidence < 0.75:
                logger.debug(f"⚠️ Confidence {confidence:.2f} below threshold")
                return
            
            # Calculate trade amount
            trade_amount = self.bot.calculate_dynamic_position_size(
                self.bot.get_effective_trade_amount(),
                confidence
            )
            
            # Execute the trade
            logger.info(f"🎯 AUTONOMOUS TRADE: {action} ${trade_amount:.2f} (Confidence: {confidence:.0%})")
            
            trade_data = await self.bot.simulate_trade(action, self.bot.get_effective_symbol(), trade_amount)
            
            if trade_data:
                # Send trade notification
                await self.send_trade_notification(trade_data, confidence)
                
        except Exception as e:
            logger.error(f"Error in trading analysis: {e}")
    
    async def send_trade_notification(self, trade_data: Dict[str, Any], confidence: float) -> None:
        """Send trade notification via Telegram"""
        try:
            if hasattr(self.bot, 'notifier') and self.bot.notifier:
                message = f"""🤖 AUTONOMOUS TRADE EXECUTED

🎯 Action: {trade_data['action']}
💰 Amount: ${trade_data['amount']:.2f}
📊 Symbol: {trade_data['symbol']}
🧠 AI Confidence: {confidence:.1%}
⚡ Entry Price: {trade_data['entry_price']:.2f}
🔥 Mode: Autonomous Simulation

🕒 Time: {datetime.now().strftime('%H:%M:%S')}
💎 Contract ID: {trade_data['contract_id']}"""
                
                await self.bot.notifier.telegram.send_message(message, parse_mode=None)
        except Exception as e:
            logger.warning(f"Trade notification failed: {e}")
    
    async def send_status_update(self) -> None:
        """Send periodic status update"""
        try:
            if hasattr(self.bot, 'notifier') and self.bot.notifier:
                total_trades = len(self.bot.session_stats.get('wins', [])) + len(self.bot.session_stats.get('losses', []))
                balance = self.bot.current_balance
                
                message = f"""🤖 AUTONOMOUS BOT STATUS

💰 Balance: ${balance:.2f}
📊 Total Trades: {total_trades}
🎯 Active Trades: {len(self.bot.active_trades)}
🔄 Restarts: {self.restart_count}
⏰ Uptime: {datetime.now().strftime('%H:%M:%S')}

✅ Status: Running Autonomously"""
                
                await self.bot.notifier.telegram.send_message(message, parse_mode=None)
        except Exception as e:
            logger.warning(f"Status update failed: {e}")
    
    def setup_signal_handlers(self) -> None:
        """Setup graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point for autonomous bot"""
    logger.info("🚀 AUTONOMOUS TRADING BOT STARTING")
    logger.info("=" * 50)
    
    autonomous_bot = AutonomousTradingBot()
    autonomous_bot.setup_signal_handlers()
    
    try:
        await autonomous_bot.run_autonomous_loop()
    except Exception as e:
        logger.error(f"❌ Critical error in autonomous bot: {e}")
    finally:
        logger.info("🛑 Autonomous bot shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
