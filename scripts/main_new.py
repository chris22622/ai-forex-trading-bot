"""
Clean Trading Bot for MT5 Trading
Streamlined for reliable trading operations
"""

import asyncio
import signal
import time
import traceback
import warnings
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# Import MetaTrader5
try:
    import MetaTrader5 as mt5  # type: ignore
    mt5_available = True
except ImportError:
    mt5_available = False
    mt5 = None

# Suppress warnings
warnings.filterwarnings("ignore")

# Import modules
from telegram_bot import NotificationManager

from ai_model import TradingAI
from config import *
from indicators import TechnicalIndicators
from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

class MarketCondition(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class TradingBot:
    """Main trading bot for MT5"""

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
        self.last_trade_time: Optional[datetime] = None

        # Performance tracking
        self.session_stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0
        }
        self.win_rate_tracker = deque(maxlen=20)
        self.market_condition = MarketCondition.SIDEWAYS

        # Components
        self.indicators = TechnicalIndicators()
        self.ai_model = TradingAI()
        self.notifier = NotificationManager()

        # Initialize
        self._initialize_mt5()
        self.ai_model.load_model()

        logger.info("🚀 Trading Bot initialized")

    def _initialize_mt5(self):
        """Initialize MT5 interface"""
        try:
            if mt5_available:
                from mt5_integration import MT5TradingInterface
                self.mt5_interface = MT5TradingInterface()
                logger.info("✅ MT5 interface created")
            else:
                logger.warning("⚠️ MT5 not available")
        except Exception as e:
            logger.error(f"MT5 initialization failed: {e}")

    async def start(self):
        """Start the trading bot"""
        logger.info("🚀 Starting trading bot...")
        self.running = True

        try:
            # Connect to MT5
            await self._connect_mt5()

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
            raise ConnectionError("MT5 interface not available")

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
            msg = f"""🚀 Trading Bot Started

✅ MT5: {'Connected' if self.mt5_connected else 'Disconnected'}
💰 Balance: ${self.current_balance:.2f}
🎯 Symbol: {DEFAULT_SYMBOL}
💵 Trade Amount: ${TRADE_AMOUNT:.2f}
🕒 Started: {datetime.now().strftime('%H:%M:%S')}

📱 Bot is scanning for opportunities..."""

            await self.notifier.telegram.send_message(msg)
            logger.info("📱 Startup notification sent")
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
        """Get current price"""
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
            # Need at least 20 price points
            if len(self.price_history) < 20:
                return

            # Analyze market
            self._analyze_market_conditions()

            # Get AI prediction
            features = self._prepare_features()
            if not features:
                return

            prediction = self.ai_model.predict(features)
            if not prediction:
                return

            action = prediction.get('action')
            confidence = prediction.get('confidence', 0.5)

            # Check confidence threshold
            if confidence < AI_CONFIDENCE_THRESHOLD:
                return

            # Check risk limits
            if not self._check_risk_limits():
                return

            # Check timing
            if not self._check_trade_timing():
                return

            # Place trade
            if action in ['BUY', 'SELL']:
                await self._place_trade(action, confidence)

        except Exception as e:
            logger.error(f"Error checking opportunity: {e}")

    def _analyze_market_conditions(self):
        """Analyze market conditions"""
        try:
            if len(self.price_history) < 20:
                return

            recent_prices = self.price_history[-20:]

            # Calculate trend
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]

            # Calculate volatility
            volatility = np.std(recent_prices) / np.mean(recent_prices)

            # Determine condition
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
        """Prepare features for AI"""
        try:
            if len(self.price_history) < 20:
                return None

            recent_prices = self.price_history[-20:]

            features = [
                recent_prices[-1],  # Current price
                np.mean(recent_prices),  # Average
                np.std(recent_prices),  # Volatility
                max(recent_prices) - min(recent_prices),  # Range
                (recent_prices[-1] - recent_prices[0]) / recent_prices[0]  # Return
            ]

            # Add technical indicators
            try:
                rsi = self.indicators.calculate_rsi(recent_prices)
                if rsi:
                    features.append(rsi[-1])
                else:
                    features.append(50.0)

                macd = self.indicators.calculate_macd(recent_prices)
                if macd and len(macd) > 0:
                    features.append(macd[-1])
                else:
                    features.append(0.0)

            except Exception:
                features.extend([50.0, 0.0])

            return features

        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return None

    def _check_risk_limits(self) -> bool:
        """Check risk limits"""
        try:
            # Check daily loss
            if self.daily_profit < -MAX_DAILY_LOSS:
                logger.warning("Daily loss limit reached")
                return False

            # Check consecutive losses
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                logger.warning("Consecutive loss limit reached")
                return False

            # Check balance
            if self.current_balance < TRADE_AMOUNT * 2:
                logger.warning("Insufficient balance")
                return False

            return True

        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False

    def _check_trade_timing(self) -> bool:
        """Check trade timing"""
        try:
            if not self.last_trade_time:
                return True

            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            return time_since_last >= MIN_TRADE_INTERVAL

        except Exception as e:
            logger.error(f"Timing check error: {e}")
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

            # Place trade
            result = await self.mt5_interface.place_trade(
                action=action,
                symbol=DEFAULT_SYMBOL,
                amount=position_size
            )

            if result:
                # Store trade
                trade_id = result.get('order', f"trade_{int(time.time())}")
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

                logger.info(f"✅ Trade placed: {trade_id}")
            else:
                logger.error("❌ Trade placement failed")

        except Exception as e:
            logger.error(f"Trade placement error: {e}")

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size"""
        try:
            # Base risk: 1% of balance
            base_risk = self.current_balance * 0.01

            # Adjust for confidence (0.5x to 1.5x)
            confidence_multiplier = 0.5 + confidence

            # Adjust for losses
            if self.consecutive_losses > 0:
                loss_multiplier = max(0.5, 1 - (self.consecutive_losses * 0.1))
            else:
                loss_multiplier = 1.0

            position_size = base_risk * confidence_multiplier * loss_multiplier

            # Apply limits
            position_size = max(1.0, min(position_size, TRADE_AMOUNT * 2))

            return round(position_size, 2)

        except Exception as e:
            logger.error(f"Position size error: {e}")
            return TRADE_AMOUNT

    async def _send_trade_notification(self, action: str, amount: float, confidence: float):
        """Send trade notification"""
        try:
            msg = f"""🚀 New Trade Placed

📊 Action: {action}
💰 Amount: ${amount:.2f}
🎯 Symbol: {DEFAULT_SYMBOL}
🧠 Confidence: {confidence:.0%}
🕒 Time: {datetime.now().strftime('%H:%M:%S')}

💡 Trade sent to MT5"""

            await self.notifier.telegram.send_message(msg)
        except Exception as e:
            logger.warning(f"Trade notification failed: {e}")

    async def _check_active_trades(self):
        """Check active trades"""
        try:
            if not self.active_trades:
                return

            completed_trades = []

            for trade_id, trade in self.active_trades.items():
                if self.mt5_interface and self.mt5_connected:
                    # Check if position still exists
                    position_info = await self.mt5_interface.get_position_info(trade_id)

                    if not position_info:  # Trade completed
                        # Get close info
                        close_info = await self.mt5_interface.close_position(trade_id)
                        if close_info:
                            await self._handle_trade_completion(trade_id, trade, close_info)
                            completed_trades.append(trade_id)

            # Remove completed trades
            for trade_id in completed_trades:
                self.active_trades.pop(trade_id, None)

        except Exception as e:
            logger.error(f"Error checking trades: {e}")

    async def _handle_trade_completion(self, trade_id: str, trade: Dict[str, Any], close_info: Dict[str, Any]):
        """Handle trade completion"""
        try:
            profit = close_info.get('profit', 0.0)

            # Update stats
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

            # Send notification
            await self._send_completion_notification(trade, profit, result)

            # Save AI data
            self.ai_model.add_training_data(
                trade.get('confidence', 0.5),
                1 if profit > 0 else 0
            )

            logger.info(f"✅ Trade completed: {trade_id} - {result} - ${profit:+.2f}")

        except Exception as e:
            logger.error(f"Trade completion error: {e}")

    async def _send_completion_notification(self, trade: Dict[str, Any], profit: float, result: str):
        """Send completion notification"""
        try:
            emoji = "🟢" if result == "WIN" else "🔴"

            msg = f"""{emoji} Trade Completed

📊 Result: {result}
💰 P&L: ${profit:+.2f}
🎯 Symbol: {trade.get('symbol', 'Unknown')}
📈 Action: {trade.get('action', 'Unknown')}
🧠 Confidence: {trade.get('confidence', 0):.0%}

📊 Session Stats:
🔸 Total: {self.session_stats['total_trades']}
🔸 Win Rate: {self.get_win_rate():.1f}%
🔸 Total P&L: ${self.session_stats['total_profit']:+.2f}"""

            await self.notifier.telegram.send_message(msg)
        except Exception as e:
            logger.warning(f"Completion notification failed: {e}")

    async def _update_balance(self):
        """Update balance"""
        try:
            if self.mt5_interface and self.mt5_connected:
                new_balance = await self.mt5_interface.get_account_balance()
                if new_balance and new_balance > 0:
                    self.current_balance = new_balance
        except Exception as e:
            logger.error(f"Balance update error: {e}")

    def get_win_rate(self) -> float:
        """Get win rate"""
        try:
            if not self.win_rate_tracker:
                return 0.0

            wins = sum(1 for result in self.win_rate_tracker if result == 'WIN')
            return (wins / len(self.win_rate_tracker)) * 100
        except Exception:
            return 0.0

    async def stop(self):
        """Stop the bot"""
        logger.info("🛑 Stopping bot...")
        self.running = False

        try:
            # Save AI model
            self.ai_model.save_model()

            # Send shutdown notification
            await self._send_shutdown_notification()

            logger.info("✅ Bot stopped")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    async def _send_shutdown_notification(self):
        """Send shutdown notification"""
        try:
            msg = f"""🛑 Bot Stopped

📊 Session Stats:
🔸 Total Trades: {self.session_stats['total_trades']}
🔸 Wins: {self.session_stats['wins']}
🔸 Losses: {self.session_stats['losses']}
🔸 Win Rate: {self.get_win_rate():.1f}%
🔸 Total P&L: ${self.session_stats['total_profit']:+.2f}
💰 Final Balance: ${self.current_balance:.2f}"""

            await self.notifier.telegram.send_message(msg)
        except Exception as e:
            logger.warning(f"Shutdown notification failed: {e}")

# Global bot instance
global_bot_instance: Optional[TradingBot] = None

async def main():
    """Main function"""
    try:
        global global_bot_instance

        # Create bot
        bot = TradingBot()
        global_bot_instance = bot

        # Setup signal handlers
        def signal_handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}")
            bot.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start bot
        await bot.start()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        if global_bot_instance:
            await global_bot_instance.stop()
    except Exception as e:
        logger.error(f"Main error: {e}")
        traceback.print_exc()
        if global_bot_instance:
            await global_bot_instance.stop()

if __name__ == "__main__":
    asyncio.run(main())
