"""
Clean Trading Bot for Deriv/MT5 Trading
Simplified and optimized for reliable MT5 trading
"""

import asyncio
import json
import time
import signal
import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import traceback
import warnings
import threading
from collections import deque

# Import MetaTrader5 with proper error handling
try:
    import MetaTrader5 as mt5
    mt5_available = True
except ImportError:
    mt5_available = False
    mt5 = None

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import our modules
from config import *
from indicators import TechnicalIndicators
from ai_model import TradingAI
from telegram_bot import NotificationManager
from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

# Global bot instance
global_bot_instance: Optional['TradingBot'] = None

class MarketCondition(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class TradingStrategy(Enum):
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    BREAKOUT = "breakout"

class TradeAction(Enum):
    BUY = "buy"
    SELL = "sell"

class TelegramHandler:
    """Simplified Telegram command handler"""
    
    def __init__(self, bot_instance: 'TradingBot'):
        self.bot = bot_instance
        self.is_monitoring = False
        
    async def start_monitoring(self) -> None:
        """Start monitoring Telegram commands"""
        if not TELEGRAM_BOT_TOKEN or len(TELEGRAM_BOT_TOKEN) < 10:
            logger.warning("Invalid Telegram token - skipping monitoring")
            return
            
        if self.is_monitoring:
            return
            
        try:
            from telegram.ext import Application, CommandHandler
            
            # Create application
            app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
            
            # Add handlers
            app.add_handler(CommandHandler("start", self.handle_start))
            app.add_handler(CommandHandler("stop", self.handle_stop))
            app.add_handler(CommandHandler("status", self.handle_status))
            app.add_handler(CommandHandler("balance", self.handle_balance))
            app.add_handler(CommandHandler("trades", self.handle_trades))
            
            # Start polling
            await app.initialize()
            await app.start()
            await app.updater.start_polling()
            
            self.is_monitoring = True
            logger.info("✅ Telegram monitoring started")
            
        except Exception as e:
            logger.error(f"Telegram monitoring failed: {e}")
    
    async def handle_start(self, update, context):
        """Handle /start command"""
        try:
            if self.bot.running:
                await update.message.reply_text("🤖 Bot is already running!")
            else:
                await update.message.reply_text("🚀 Starting trading bot...")
                asyncio.create_task(self.bot.start())
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def handle_stop(self, update, context):
        """Handle /stop command"""
        try:
            await update.message.reply_text("🛑 Stopping bot...")
            await self.bot.stop()
            await update.message.reply_text("✅ Bot stopped successfully!")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def handle_status(self, update, context):
        """Handle /status command"""
        try:
            status = f"""📊 <b>Bot Status</b>
            
🤖 Running: {'✅' if self.bot.running else '❌'}
🔗 MT5 Connected: {'✅' if self.bot.mt5_connected else '❌'}
💰 Balance: ${self.bot.current_balance:.2f}
📈 Trades Today: {self.bot.trades_today}
🎯 Active Trades: {len(self.bot.active_trades)}
📊 Win Rate: {self.bot.get_win_rate():.1f}%
💵 Daily P&L: ${self.bot.daily_profit:+.2f}"""
            
            await update.message.reply_text(status, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def handle_balance(self, update, context):
        """Handle /balance command"""
        try:
            balance = await self.bot.get_account_balance()
            msg = f"💰 Account Balance: ${balance:.2f}\n📊 Daily P&L: ${self.bot.daily_profit:+.2f}"
            await update.message.reply_text(msg)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def handle_trades(self, update, context):
        """Handle /trades command"""
        try:
            if not self.bot.active_trades:
                await update.message.reply_text("📊 No active trades")
                return
            
            trades_msg = "📈 <b>Active Trades</b>\n\n"
            for trade_id, trade in self.bot.active_trades.items():
                trades_msg += f"🔸 {trade.get('symbol', 'Unknown')} - {trade.get('action', 'Unknown')}\n"
                trades_msg += f"   Amount: ${trade.get('amount', 0):.2f}\n"
                trades_msg += f"   Time: {trade.get('timestamp', 'Unknown')}\n\n"
            
            await update.message.reply_text(trades_msg, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

class TradingBot:
    """Main trading bot class - clean and simplified"""
    
    def __init__(self):
        self.running = False
        self.mt5_connected = False
        self.mt5_interface = None
        
        # Trading state
        self.current_balance = 0.0
        self.active_trades: Dict[str, Any] = {}
        self.price_history: List[float] = []
        self.trades_today = 0
        self.daily_profit = 0.0
        self.consecutive_losses = 0
        
        # Performance tracking
        self.session_stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0
        }
        self.win_rate_tracker = deque(maxlen=20)
        self.market_condition = MarketCondition.SIDEWAYS
        self.current_strategy = TradingStrategy.MOMENTUM
        
        # Components
        self.indicators = TechnicalIndicators()
        self.ai_model = TradingAI()
        self.notifier = NotificationManager()
        self.telegram_handler = TelegramHandler(self)
        
        # Load AI model
        self.ai_model.load_model()
        
        # Initialize MT5
        self._initialize_mt5()
        
        logger.info("🚀 Trading Bot initialized")
    
    def _initialize_mt5(self):
        """Initialize MT5 interface"""
        try:
            if mt5_available:
                from mt5_integration import MT5TradingInterface
                self.mt5_interface = MT5TradingInterface()
                logger.info("✅ MT5 interface created")
            else:
                logger.warning("⚠️ MT5 not available - simulation mode only")
        except Exception as e:
            logger.error(f"MT5 initialization failed: {e}")
    
    async def start(self):
        """Start the trading bot"""
        logger.info("🚀 Starting trading bot...")
        self.running = True
        
        try:
            # Connect to MT5
            await self._connect_mt5()
            
            # Start Telegram monitoring
            if ENABLE_TELEGRAM_ALERTS:
                asyncio.create_task(self.telegram_handler.start_monitoring())
            
            # Send startup notification
            await self._send_startup_notification()
            
            # Start main trading loop
            await self._main_trading_loop()
            
        except Exception as e:
            logger.error(f"Bot startup error: {e}")
            await self.stop()
    
    async def _connect_mt5(self):
        """Connect to MT5"""
        if not self.mt5_interface:
            logger.error("❌ No MT5 interface available")
            return
        
        try:
            success = await self.mt5_interface.initialize()
            if success:
                self.mt5_connected = True
                self.current_balance = await self.mt5_interface.get_account_balance()
                logger.info(f"✅ MT5 connected! Balance: ${self.current_balance:.2f}")
            else:
                logger.error("❌ MT5 connection failed")
                raise ConnectionError("MT5 connection failed")
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            raise
    
    async def _send_startup_notification(self):
        """Send startup notification"""
        try:
            msg = f"""🚀 <b>Trading Bot Started</b>

✅ MT5: {'Connected' if self.mt5_connected else 'Disconnected'}
💰 Balance: ${self.current_balance:.2f}
🎯 Symbol: {DEFAULT_SYMBOL}
💵 Trade Amount: ${TRADE_AMOUNT:.2f}
🕒 Started: {datetime.now().strftime('%H:%M:%S')}

📱 Bot is scanning for trading opportunities..."""
            
            await self.notifier.telegram.send_message(msg, parse_mode='HTML')
        except Exception as e:
            logger.warning(f"Startup notification failed: {e}")
    
    async def _main_trading_loop(self):
        """Main trading loop"""
        logger.info("🎯 Starting main trading loop...")
        
        while self.running:
            try:
                # Get current price
                current_price = await self._get_current_price()
                if current_price:
                    self.price_history.append(current_price)
                    
                    # Keep price history manageable
                    if len(self.price_history) > 1000:
                        self.price_history = self.price_history[-500:]
                    
                    # Check for trading opportunities
                    await self._check_trading_opportunity(current_price)
                
                # Check active trades
                await self._check_active_trades()
                
                # Update balance periodically
                if len(self.price_history) % 20 == 0:
                    await self._update_balance()
                
                # Wait before next iteration
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _get_current_price(self) -> Optional[float]:
        """Get current price for default symbol"""
        try:
            if self.mt5_interface and self.mt5_connected:
                return await self.mt5_interface.get_current_price(DEFAULT_SYMBOL)
            return None
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None
    
    async def _check_trading_opportunity(self, current_price: float):
        """Check for trading opportunities"""
        try:
            # Need at least 20 price points for analysis
            if len(self.price_history) < 20:
                return
            
            # Analyze market conditions
            await self._analyze_market_conditions()
            
            # Get AI prediction
            features = self._prepare_features()
            if not features:
                return
            
            prediction = self.ai_model.predict(features)
            if not prediction:
                return
            
            action = prediction.get('action')
            confidence = prediction.get('confidence', 0.5)
            
            # Only trade with high confidence
            if confidence < AI_CONFIDENCE_THRESHOLD:
                return
            
            # Check risk limits
            if not self._check_risk_limits():
                return
            
            # Check trade timing
            if not self._check_trade_timing():
                return
            
            # Place trade
            if action in ['BUY', 'SELL']:
                await self._place_trade(action, confidence)
                
        except Exception as e:
            logger.error(f"Error checking trading opportunity: {e}")
    
    async def _analyze_market_conditions(self):
        """Analyze current market conditions"""
        try:
            if len(self.price_history) < 20:
                return
            
            recent_prices = self.price_history[-20:]
            
            # Calculate trend
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            # Calculate volatility
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            
            # Determine market condition
            if abs(trend) < 0.001:
                self.market_condition = MarketCondition.SIDEWAYS
            elif trend > 0.001:
                self.market_condition = MarketCondition.TRENDING_UP
            elif trend < -0.001:
                self.market_condition = MarketCondition.TRENDING_DOWN
            
            if volatility > 0.005:
                self.market_condition = MarketCondition.VOLATILE
                
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
    
    def _prepare_features(self) -> Optional[List[float]]:
        """Prepare features for AI model"""
        try:
            if len(self.price_history) < 20:
                return None
            
            recent_prices = self.price_history[-20:]
            
            # Basic features
            features = [
                recent_prices[-1],  # Current price
                np.mean(recent_prices),  # Average price
                np.std(recent_prices),  # Volatility
                max(recent_prices) - min(recent_prices),  # Range
                (recent_prices[-1] - recent_prices[0]) / recent_prices[0]  # Return
            ]
            
            # Technical indicators
            try:
                rsi = self.indicators.calculate_rsi(recent_prices)
                if rsi:
                    features.append(rsi[-1])
                
                macd = self.indicators.calculate_macd(recent_prices)
                if macd and len(macd) > 0:
                    features.append(macd[-1])
                    
            except Exception as ind_error:
                logger.debug(f"Indicator calculation error: {ind_error}")
                features.extend([50.0, 0.0])  # Default values
            
            return features
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return None
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits allow trading"""
        try:
            # Check daily loss limit
            if self.daily_profit < -MAX_DAILY_LOSS:
                logger.warning("Daily loss limit reached")
                return False
            
            # Check consecutive losses
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                logger.warning("Consecutive loss limit reached")
                return False
            
            # Check balance
            if self.current_balance < TRADE_AMOUNT * 2:
                logger.warning("Insufficient balance for trading")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False
    
    def _check_trade_timing(self) -> bool:
        """Check if enough time has passed since last trade"""
        try:
            if not hasattr(self, 'last_trade_time') or not self.last_trade_time:
                return True
            
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            return time_since_last >= MIN_TRADE_INTERVAL
            
        except Exception as e:
            logger.error(f"Trade timing check error: {e}")
            return True
    
    async def _place_trade(self, action: str, confidence: float):
        """Place a trade"""
        try:
            if not self.mt5_interface or not self.mt5_connected:
                logger.error("❌ Cannot place trade - MT5 not connected")
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(confidence)
            
            logger.info(f"🚀 Placing {action} trade: ${position_size:.2f} (Confidence: {confidence:.0%})")
            
            # Place trade through MT5
            result = await self.mt5_interface.place_trade(
                action=action,
                symbol=DEFAULT_SYMBOL,
                amount=position_size
            )
            
            if result:
                # Store trade info
                trade_id = result.get('order', f"trade_{len(self.active_trades)}")
                self.active_trades[trade_id] = {
                    'action': action,
                    'symbol': DEFAULT_SYMBOL,
                    'amount': position_size,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'entry_price': result.get('price', 0.0)
                }
                
                self.trades_today += 1
                self.last_trade_time = datetime.now()
                
                # Send notification
                await self._send_trade_notification(action, position_size, confidence)
                
                logger.info(f"✅ Trade placed successfully: {trade_id}")
            else:
                logger.error("❌ Trade placement failed")
                
        except Exception as e:
            logger.error(f"Trade placement error: {e}")
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence and risk management"""
        try:
            # Base risk: 1% of balance
            base_risk = self.current_balance * 0.01
            
            # Adjust based on confidence (0.5x to 1.5x)
            confidence_multiplier = 0.5 + confidence
            
            # Adjust based on consecutive losses
            if self.consecutive_losses > 0:
                loss_multiplier = max(0.5, 1 - (self.consecutive_losses * 0.1))
            else:
                loss_multiplier = 1.0
            
            position_size = base_risk * confidence_multiplier * loss_multiplier
            
            # Ensure minimum and maximum limits
            position_size = max(1.0, min(position_size, TRADE_AMOUNT * 2))
            
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return TRADE_AMOUNT
    
    async def _send_trade_notification(self, action: str, amount: float, confidence: float):
        """Send trade notification"""
        try:
            msg = f"""🚀 <b>New Trade Placed</b>

📊 Action: {action}
💰 Amount: ${amount:.2f}
🎯 Symbol: {DEFAULT_SYMBOL}
🧠 Confidence: {confidence:.0%}
🕒 Time: {datetime.now().strftime('%H:%M:%S')}

💡 Trade sent to MT5 terminal"""
            
            await self.notifier.telegram.send_message(msg, parse_mode='HTML')
        except Exception as e:
            logger.warning(f"Trade notification failed: {e}")
    
    async def _check_active_trades(self):
        """Check status of active trades"""
        try:
            if not self.active_trades:
                return
            
            completed_trades = []
            
            for trade_id, trade in self.active_trades.items():
                # Check trade status via MT5
                if self.mt5_interface and self.mt5_connected:
                    position_info = await self.mt5_interface.get_position_info(trade_id)
                    
                    if not position_info:  # Trade completed
                        # Try to get close info
                        close_info = await self.mt5_interface.close_position(trade_id)
                        if close_info:
                            await self._handle_trade_completion(trade_id, trade, close_info)
                            completed_trades.append(trade_id)
            
            # Remove completed trades
            for trade_id in completed_trades:
                self.active_trades.pop(trade_id, None)
                
        except Exception as e:
            logger.error(f"Error checking active trades: {e}")
    
    async def _handle_trade_completion(self, trade_id: str, trade: Dict[str, Any], close_info: Dict[str, Any]):
        """Handle completed trade"""
        try:
            profit = close_info.get('profit', 0.0)
            
            # Update statistics
            self.session_stats['total_trades'] += 1
            self.session_stats['total_profit'] += profit
            self.daily_profit += profit
            
            if profit > 0:
                self.session_stats['wins'] += 1
                self.win_rate_tracker.append('WIN')
                self.consecutive_losses = 0
                result = "WIN"
            else:
                self.session_stats['losses'] += 1
                self.win_rate_tracker.append('LOSS')
                self.consecutive_losses += 1
                result = "LOSS"
            
            # Send completion notification
            await self._send_completion_notification(trade, profit, result)
            
            # Save AI learning data
            self.ai_model.add_training_data(
                trade.get('confidence', 0.5),
                1 if profit > 0 else 0
            )
            
            logger.info(f"✅ Trade completed: {trade_id} - {result} - ${profit:+.2f}")
            
        except Exception as e:
            logger.error(f"Trade completion handling error: {e}")
    
    async def _send_completion_notification(self, trade: Dict[str, Any], profit: float, result: str):
        """Send trade completion notification"""
        try:
            emoji = "🟢" if result == "WIN" else "🔴"
            
            msg = f"""{emoji} <b>Trade Completed</b>

📊 Result: {result}
💰 Profit/Loss: ${profit:+.2f}
🎯 Symbol: {trade.get('symbol', 'Unknown')}
📈 Action: {trade.get('action', 'Unknown')}
🧠 Confidence: {trade.get('confidence', 0):.0%}

📊 Session Stats:
🔸 Total Trades: {self.session_stats['total_trades']}
🔸 Win Rate: {self.get_win_rate():.1f}%
🔸 Total P&L: ${self.session_stats['total_profit']:+.2f}"""
            
            await self.notifier.telegram.send_message(msg, parse_mode='HTML')
        except Exception as e:
            logger.warning(f"Completion notification failed: {e}")
    
    async def _update_balance(self):
        """Update account balance"""
        try:
            if self.mt5_interface and self.mt5_connected:
                new_balance = await self.mt5_interface.get_account_balance()
                if new_balance and new_balance > 0:
                    self.current_balance = new_balance
        except Exception as e:
            logger.error(f"Balance update error: {e}")
    
    async def get_account_balance(self) -> float:
        """Get current account balance"""
        await self._update_balance()
        return self.current_balance
    
    def get_win_rate(self) -> float:
        """Calculate current win rate"""
        try:
            if not self.win_rate_tracker:
                return 0.0
            
            wins = sum(1 for result in self.win_rate_tracker if result == 'WIN')
            return (wins / len(self.win_rate_tracker)) * 100
        except Exception:
            return 0.0
    
    async def stop(self):
        """Stop the trading bot"""
        logger.info("🛑 Stopping trading bot...")
        self.running = False
        
        try:
            # Close active trades if needed
            if self.active_trades:
                logger.info("Closing active trades...")
                # Implementation would depend on requirements
            
            # Save AI model
            self.ai_model.save_model()
            
            # Send shutdown notification
            await self._send_shutdown_notification()
            
            logger.info("✅ Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    async def _send_shutdown_notification(self):
        """Send shutdown notification"""
        try:
            msg = f"""🛑 <b>Bot Stopped</b>

📊 Final Session Stats:
🔸 Total Trades: {self.session_stats['total_trades']}
🔸 Wins: {self.session_stats['wins']}
🔸 Losses: {self.session_stats['losses']}
🔸 Win Rate: {self.get_win_rate():.1f}%
🔸 Total P&L: ${self.session_stats['total_profit']:+.2f}
💰 Final Balance: ${self.current_balance:.2f}"""
            
            await self.notifier.telegram.send_message(msg, parse_mode='HTML')
        except Exception as e:
            logger.warning(f"Shutdown notification failed: {e}")

async def main():
    """Main function"""
    try:
        # Create and start bot
        global global_bot_instance
        bot = TradingBot()
        global_bot_instance = bot
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            bot.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start bot
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        if global_bot_instance:
            await global_bot_instance.stop()
    except Exception as e:
        logger.error(f"Main error: {e}")
        traceback.print_exc()
    finally:
        if global_bot_instance and global_bot_instance.running:
            await global_bot_instance.stop()

if __name__ == "__main__":
    asyncio.run(main())
