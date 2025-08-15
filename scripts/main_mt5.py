"""
Main Trading Bot with MetaTrader 5 Integration
Enhanced version that trades through MT5 instead of WebSocket API
"""

import asyncio
import signal
import time
import traceback
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*TensorFlow.*")
warnings.filterwarnings("ignore", message=".*matplotlib.*")
warnings.filterwarnings("ignore", message=".*seaborn.*")
warnings.filterwarnings("ignore", message=".*ReportLab.*")

# Import our modules
from ai_model import TradingAI
from config import *
from indicators import TechnicalIndicators
from mt5_integration import MT5TradingInterface
from safe_logger import get_safe_logger
from telegram_bot import NotificationManager
from utils import (
    check_risk_limits,
    format_currency,
    trade_logger,
    validate_trade_parameters,
)

# Use safe logger for Windows compatibility
logger = get_safe_logger(__name__)

# Import enhanced AI system
enhanced_ai_available = True
AIModelManager = None
try:
    from ai_integration_system import AIModelManager
except ImportError:
    enhanced_ai_available = False
    logger.warning("Enhanced AI system not available, using basic AI")
from collections import deque
from enum import Enum


class MarketCondition(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class TradingStrategy(Enum):
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    BREAKOUT = "breakout"

class DerivMT5TradingBot:
    """Enhanced Trading Bot with MetaTrader 5 Integration"""

    def __init__(self):
        self.running = False
        self.connected = False
        self.mt5_interface = None

        # Trading state
        self.current_balance: float = 0.0
        self.active_trades: Dict[str, Any] = {}
        self.price_history: List[float] = []
        self.last_trade_time = None
        self.consecutive_losses = 0
        self.daily_profit = 0.0
        self.trades_today = 0

        # Enhanced features
        self.market_condition = MarketCondition.SIDEWAYS
        self.current_strategy = TradingStrategy.MOMENTUM
        self.cooldown_until: Optional[datetime] = None
        self.win_rate_tracker: deque[str] = deque(maxlen=20)
        self.confidence_heatmap: List[Dict[str, Any]] = []
        self.rl_data: List[Dict[str, Any]] = []
        self.multi_timeframe_data: Dict[str, List[float]] = {
            '1m': [],
            '5m': [],
            '15m': []
        }
        self.strategy_performance: Dict[TradingStrategy, Dict[str, float]] = {
            strategy: {'wins': 0, 'losses': 0, 'total_profit': 0.0}
            for strategy in TradingStrategy
        }

        # Components
        self.indicators = TechnicalIndicators()

        # Use enhanced AI if available
        if enhanced_ai_available and AIModelManager is not None:
            self.ai_manager = AIModelManager(initial_balance=1000.0)
            self.ai_model = None
            logger.info("🧠 Enhanced AI system loaded")
        else:
            self.ai_model = TradingAI()
            self.ai_manager = None
            self.ai_model.load_model()
            logger.info("🧠 Basic AI system loaded")

        self.notifier = NotificationManager()

        # Performance tracking
        self.start_time = datetime.now()
        self.last_heartbeat = datetime.now()
        self.session_stats: Dict[str, Any] = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_streak': 0,
            'confidence_wins': [],
            'confidence_losses': []
        }

        logger.info("🚀 Enhanced Deriv MT5 Trading Bot initialized")

    # Copy all the analysis methods from the original bot
    def calculate_dynamic_position_size(self, base_amount: float, confidence: float,
                                      daily_win_rate: float = 0.5) -> float:
        """Calculate dynamic position size based on confidence and performance"""
        try:
            confidence_multiplier = 0.5 + (confidence * 1.5)
            win_rate_multiplier = 0.7 + (daily_win_rate * 0.6)
            loss_penalty = max(0.3, 1.0 - (self.consecutive_losses * 0.15))

            dynamic_amount = base_amount * confidence_multiplier * win_rate_multiplier * loss_penalty

            min_amount = base_amount * 0.3
            max_amount = base_amount * 3.0

            return max(min_amount, min(max_amount, dynamic_amount))

        except Exception as e:
            logger.error(f"Error calculating dynamic position size: {e}")
            return base_amount

    def analyze_market_condition(self) -> MarketCondition:
        """Analyze current market condition"""
        try:
            if len(self.price_history) < 50:
                return MarketCondition.SIDEWAYS

            recent_prices = self.price_history[-50:]
            price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
            positive_changes = sum(1 for change in price_changes if change > 0)
            trend_ratio = positive_changes / len(price_changes)

            volatility = np.std(price_changes) / np.mean(recent_prices) if np.mean(recent_prices) > 0 else 0

            if volatility > 0.002:
                return MarketCondition.VOLATILE
            elif trend_ratio > 0.65:
                return MarketCondition.TRENDING_UP
            elif trend_ratio < 0.35:
                return MarketCondition.TRENDING_DOWN
            else:
                return MarketCondition.SIDEWAYS

        except Exception as e:
            logger.error(f"Error analyzing market condition: {e}")
            return MarketCondition.SIDEWAYS

    def select_optimal_strategy(self) -> TradingStrategy:
        """Select optimal trading strategy based on market conditions and performance"""
        try:
            best_strategy = TradingStrategy.MOMENTUM
            best_score = -float('inf')

            for strategy, stats in self.strategy_performance.items():
                total_trades = stats['wins'] + stats['losses']
                if total_trades >= 3:
                    win_rate = stats['wins'] / total_trades
                    avg_profit = stats['total_profit'] / total_trades
                    score = (win_rate * 0.6) + (avg_profit * 0.4)

                    if score > best_score:
                        best_score = score
                        best_strategy = strategy

            if self.market_condition == MarketCondition.TRENDING_UP:
                return TradingStrategy.MOMENTUM
            elif self.market_condition == MarketCondition.TRENDING_DOWN:
                return TradingStrategy.MOMENTUM
            elif self.market_condition == MarketCondition.SIDEWAYS:
                return TradingStrategy.REVERSAL
            elif self.market_condition == MarketCondition.VOLATILE:
                return TradingStrategy.BREAKOUT

            return best_strategy

        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            return TradingStrategy.MOMENTUM

    def update_multi_timeframe_data(self, price: float, timestamp: int) -> None:
        """Update multi-timeframe analysis data"""
        try:
            self.multi_timeframe_data['1m'].append(price)
            if len(self.multi_timeframe_data['1m']) > 60:
                self.multi_timeframe_data['1m'] = self.multi_timeframe_data['1m'][-60:]

            if len(self.price_history) % 5 == 0:
                self.multi_timeframe_data['5m'].append(price)
                if len(self.multi_timeframe_data['5m']) > 60:
                    self.multi_timeframe_data['5m'] = self.multi_timeframe_data['5m'][-60:]

            if len(self.price_history) % 15 == 0:
                self.multi_timeframe_data['15m'].append(price)
                if len(self.multi_timeframe_data['15m']) > 60:
                    self.multi_timeframe_data['15m'] = self.multi_timeframe_data['15m'][-60:]

        except Exception as e:
            logger.error(f"Error updating multi-timeframe data: {e}")

    def check_timeframe_alignment(self) -> bool:
        """Check if all timeframes agree on direction"""
        try:
            alignments: List[bool] = []

            for _, prices in self.multi_timeframe_data.items():
                if len(prices) >= 10:
                    recent_trend = prices[-1] - prices[-10]
                    alignments.append(recent_trend > 0)

            if len(alignments) >= 2:
                return len(set(alignments)) == 1

            return True

        except Exception as e:
            logger.error(f"Error checking timeframe alignment: {e}")
            return True

    async def initialize_mt5(self) -> bool:
        """Initialize MetaTrader 5 connection"""
        try:
            logger.info("🔄 Initializing MetaTrader 5 connection...")

            # Create MT5 interface
            mt5_interface = MT5TradingInterface()

            # Initialize connection
            if await mt5_interface.initialize():
                self.mt5_interface = mt5_interface
                self.connected = True

                # Update balance from MT5
                self.current_balance = await mt5_interface.get_current_equity()

                logger.info("✅ MT5 connection established successfully!")

                # Send startup notification
                try:
                    await self.notifier.notify_status({
                        "status": "CONNECTED",
                        "platform": "MetaTrader 5",
                        "balance": self.current_balance
                    })
                except Exception as e:
                    logger.warning(f"Notification failed: {e}")

                return True
            else:
                logger.error("❌ Failed to initialize MT5 connection")
                return False

        except Exception as e:
            logger.error(f"❌ MT5 initialization error: {e}")
            return False

    async def place_trade_mt5(self, action: str, symbol: str, amount: float) -> Optional[Dict[str, Any]]:
        """Place trade through MetaTrader 5"""
        try:
            if not self.mt5_interface:
                logger.error("❌ MT5 interface not initialized")
                return None

            # Validate parameters
            if not validate_trade_parameters(symbol, amount, action):
                return None

            # Check risk limits
            daily_stats = trade_logger.get_daily_stats()
            if not check_risk_limits(
                self.consecutive_losses,
                abs(daily_stats['total_profit']) if daily_stats['total_profit'] < 0 else 0,
                self.current_balance
            ):
                logger.warning("🚨 Risk limits exceeded, skipping trade")
                return None

            # Check minimum interval between trades
            if (self.last_trade_time and
                (datetime.now() - self.last_trade_time).total_seconds() < MIN_TRADE_INTERVAL):
                logger.info("⏰ Trade interval too short, waiting...")
                return None

            # Calculate lot size
            lot_size = self.mt5_interface.calculate_lot_size(symbol, amount)

            logger.info(f"🎯 Placing MT5 trade: {action} {symbol} ${amount} ({lot_size} lots)")

            # Place order through MT5
            if action == "BUY":
                result = await self.mt5_interface.place_buy_order(
                    symbol, lot_size, comment="AI Bot Trade"
                )
            else:  # SELL
                result = await self.mt5_interface.place_sell_order(
                    symbol, lot_size, comment="AI Bot Trade"
                )

            if result:
                # Create trade data compatible with original bot
                trade_data = {
                    "contract_id": str(result["deal"]),
                    "action": action,
                    "symbol": symbol,
                    "amount": amount,
                    "buy_price": amount,
                    "start_time": datetime.now(),
                    "entry_price": result["price"],
                    "mt5_order": result["order"],
                    "mt5_deal": result["deal"],
                    "lot_size": lot_size
                }

                # Add to active trades
                self.active_trades[str(result["deal"])] = trade_data
                self.last_trade_time = datetime.now()
                self.trades_today += 1

                logger.info(f"✅ MT5 trade placed: Deal {result['deal']}")
                return trade_data

            return None

        except Exception as e:
            logger.error(f"❌ Error placing MT5 trade: {e}")
            return None

    async def check_mt5_positions(self) -> None:
        """Check MT5 positions and update trade results"""
        try:
            if not self.mt5_interface:
                return

            # Get current MT5 positions
            positions = await self.mt5_interface.get_open_positions()
            current_deals = {str(pos["ticket"]): pos for pos in positions}

            # Check for closed positions
            closed_trades = []
            for contract_id, trade_data in list(self.active_trades.items()):
                if "mt5_deal" in trade_data:
                    deal_id = str(trade_data["mt5_deal"])
                    if deal_id not in current_deals:
                        closed_trades.append((contract_id, trade_data))

            # Process closed trades
            for contract_id, trade_data in closed_trades:
                # Get recent trade history to find close details
                history = await self.mt5_interface.get_trade_history(days=1)

                close_profit = 0.0
                for deal in history:
                    if str(deal["ticket"]) == contract_id:
                        close_profit = deal.get("profit", 0.0)
                        break

                # Create result for processing
                result = {
                    "status": "won" if close_profit > 0 else "lost",
                    "payout": trade_data["amount"] + close_profit if close_profit > 0 else 0,
                    "profit_loss": close_profit,
                    "exit_price": 0
                }

                # Process the trade result
                await self.process_trade_result(contract_id, result)

                # Remove from active trades
                if contract_id in self.active_trades:
                    del self.active_trades[contract_id]

                logger.info(f"📊 MT5 trade closed: {contract_id} P&L: ${close_profit:.2f}")

        except Exception as e:
            logger.error(f"❌ Error checking MT5 positions: {e}")

    async def process_trade_result(self, contract_id: str, result: Dict[str, Any]) -> None:
        """Process completed trade result"""
        try:
            trade_data = self.active_trades.get(contract_id)
            if not trade_data:
                return

            status = result["status"]
            profit_loss = result["profit_loss"]

            is_win = status == "won"
            result_type = "WIN" if is_win else "LOSS"

            # Update tracking
            if is_win:
                self.consecutive_losses = 0
                self.session_stats['wins'] += 1
                self.win_rate_tracker.append('WIN')
            else:
                self.consecutive_losses += 1
                self.session_stats['losses'] += 1
                self.win_rate_tracker.append('LOSS')

            self.daily_profit += profit_loss
            self.session_stats['total_trades'] += 1
            self.session_stats['total_profit'] += profit_loss

            # Update strategy performance
            strategy = trade_data.get('strategy', TradingStrategy.MOMENTUM)
            if isinstance(strategy, TradingStrategy):
                if is_win:
                    self.strategy_performance[strategy]['wins'] += 1
                else:
                    self.strategy_performance[strategy]['losses'] += 1
                self.strategy_performance[strategy]['total_profit'] += profit_loss

            # Update current balance from MT5
            if self.mt5_interface:
                self.current_balance = await self.mt5_interface.get_current_equity()

            # Create trade record
            duration = (datetime.now() - trade_data["start_time"]).total_seconds()
            trade_record = {
                "trade_id": contract_id,
                "action": trade_data["action"],
                "symbol": trade_data["symbol"],
                "amount": trade_data["amount"],
                "price": trade_data["entry_price"],
                "confidence": 0.75,
                "reason": "AI prediction",
                "duration": int(duration),
                "result": result_type,
                "profit_loss": profit_loss,
                "balance": self.current_balance
            }

            # Log trade
            trade_logger.log_trade(trade_record)

            # Send notifications
            await self.notifier.notify_result({
                "symbol": trade_data["symbol"],
                "result": result_type,
                "profit_loss": profit_loss,
                "duration": int(duration)
            })

            # Update AI model performance
            if self.ai_model:
                self.ai_model.update_performance(contract_id, result_type)

            logger.info(f"📊 Trade completed: {result_type} {format_currency(profit_loss)} "
                       f"(Balance: ${self.current_balance:.2f})")

        except Exception as e:
            logger.error(f"Error processing trade result: {e}")

    async def analyze_market_and_trade(self) -> None:
        """Enhanced trading logic with MT5 execution"""
        try:
            if len(self.price_history) < 30:
                return

            # Update market analysis
            self.market_condition = self.analyze_market_condition()
            self.current_strategy = self.select_optimal_strategy()

            # Check timeframe alignment
            if not self.check_timeframe_alignment():
                logger.debug("📊 Timeframes not aligned, skipping trade")
                return

            # Get technical analysis
            indicators_data = self.indicators.get_indicator_summary()
            if not indicators_data or not indicators_data.get('price'):
                return

            # Get AI prediction
            if self.ai_manager:
                ai_prediction = self.ai_manager.get_trading_prediction(
                    indicators_data,
                    self.price_history[-50:],
                    TRADE_AMOUNT
                )
            elif self.ai_model:
                ai_prediction = self.ai_model.predict_trade(
                    indicators_data,
                    self.price_history[-50:]
                )
            else:
                logger.error("No AI model available!")
                return

            # Calculate current win rate
            recent_wins = sum(1 for result in list(self.win_rate_tracker) if result == 'WIN')
            current_win_rate = recent_wins / max(1, len(self.win_rate_tracker))

            # Check if we should trade
            confidence_threshold = AI_CONFIDENCE_THRESHOLD

            if self.market_condition == MarketCondition.VOLATILE:
                confidence_threshold += 0.1
            elif self.market_condition == MarketCondition.SIDEWAYS:
                confidence_threshold += 0.05

            if (ai_prediction['prediction'] in ['BUY', 'SELL'] and
                ai_prediction['confidence'] >= confidence_threshold):

                # Additional safety checks
                if len(self.active_trades) >= 3:
                    logger.info("📊 Max active trades reached, waiting...")
                    return

                # Calculate dynamic position size
                dynamic_amount = self.calculate_dynamic_position_size(
                    TRADE_AMOUNT,
                    ai_prediction['confidence'],
                    current_win_rate
                )

                # Use enhanced position size if available
                if 'position_size' in ai_prediction and ai_prediction['position_size'] > 0:
                    dynamic_amount = ai_prediction['position_size']

                # Place trade through MT5
                symbol = self.mt5_interface.default_symbol if self.mt5_interface else DEFAULT_SYMBOL
                trade_result = await self.place_trade_mt5(
                    ai_prediction['prediction'],
                    symbol,
                    dynamic_amount
                )

                if trade_result:
                    # Update strategy tracking
                    trade_result['strategy'] = self.current_strategy
                    trade_result['market_condition'] = self.market_condition

                    # Send trade notification
                    await self.notifier.notify_trade({
                        "action": ai_prediction['prediction'],
                        "symbol": symbol,
                        "amount": dynamic_amount,
                        "price": indicators_data.get('price', 0),
                        "confidence": ai_prediction['confidence'],
                        "reason": ai_prediction.get('reason', 'AI prediction'),
                        "strategy": self.current_strategy.value,
                        "market_condition": self.market_condition.value
                    })

                    logger.info(f"🎯 MT5 Trade executed: {ai_prediction['prediction']} "
                               f"${dynamic_amount:.2f} (Confidence: {ai_prediction['confidence']:.1%})")

        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            traceback.print_exc()

    async def run_mt5_trading_loop(self) -> None:
        """Main MT5 trading loop"""
        try:
            if not self.mt5_interface:
                logger.error("❌ MT5 interface not available")
                return

            symbol = self.mt5_interface.default_symbol
            logger.info(f"🎯 Starting MT5 trading loop for {symbol}")

            while self.running and self.connected:
                try:
                    # Get current price from MT5
                    price = await self.mt5_interface.get_current_price(symbol)
                    if price:
                        timestamp = int(time.time())

                        # Update price history
                        self.price_history.append(price)
                        if len(self.price_history) > 1000:
                            self.price_history = self.price_history[-500:]

                        # Update indicators
                        self.indicators.update_price_data(price, timestamp)

                        # Update multi-timeframe data
                        self.update_multi_timeframe_data(price, timestamp)

                        # Update balance from MT5
                        self.current_balance = await self.mt5_interface.get_current_equity()

                        # Check for closed positions
                        await self.check_mt5_positions()

                        # Analyze market and potentially trade every 5 price updates
                        if len(self.price_history) % 5 == 0:
                            await self.analyze_market_and_trade()

                    # Wait before next update
                    await asyncio.sleep(2.0)  # 2 second intervals

                except Exception as e:
                    logger.error(f"❌ Error in MT5 trading loop: {e}")
                    await asyncio.sleep(5.0)

        except Exception as e:
            logger.error(f"❌ MT5 trading loop error: {e}")
            traceback.print_exc()

    async def start(self) -> None:
        """Start the MT5 trading bot"""
        logger.info("🚀 Starting Enhanced Deriv MT5 Trading Bot...")

        self.running = True

        try:
            # Initialize MT5 connection
            if await self.initialize_mt5():
                logger.info("✅ MT5 connection established, starting trading...")

                # Start the main trading loop
                await self.run_mt5_trading_loop()
            else:
                logger.error("❌ Failed to initialize MT5 - check MetaTrader 5 is running")

        except KeyboardInterrupt:
            logger.info("⏹️ Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
            traceback.print_exc()
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading bot"""
        logger.info("⏹️ Stopping MT5 Trading Bot...")

        self.running = False

        try:
            # Close MT5 positions if needed
            if self.mt5_interface and self.active_trades:
                logger.info(f"🔄 Closing {len(self.active_trades)} open positions...")
                await self.mt5_interface.close_all_positions()

            # Save AI model state
            if self.ai_model:
                self.ai_model.save_model()

            # Shutdown MT5 connection
            if self.mt5_interface:
                self.mt5_interface.shutdown()

            logger.info("✅ MT5 Trading Bot stopped successfully")

        except Exception as e:
            logger.error(f"❌ Error stopping bot: {e}")

# Copy other utility methods from original bot
    def generate_session_summary(self) -> Dict[str, Any]:
        """Generate comprehensive session summary"""
        try:
            session_duration = datetime.now() - self.start_time

            summary = {
                'session_duration': str(session_duration).split('.')[0],
                'total_trades': self.session_stats['total_trades'],
                'wins': self.session_stats['wins'],
                'losses': self.session_stats['losses'],
                'win_rate': self.session_stats['wins'] / max(1, self.session_stats['total_trades']),
                'total_profit': self.session_stats['total_profit'],
                'final_balance': self.current_balance,
                'platform': 'MetaTrader 5'
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {}

# ==================== MAIN EXECUTION ====================
async def main():
    """Main function for MT5 bot"""
    print("=" * 60)
    print("🚀 DERIV TRADING BOT - MT5 EDITION")
    print("🤖 AI-Powered Trading via MetaTrader 5")
    print("=" * 60)
    print("📡 Platform: MetaTrader 5")
    print(f"💰 Trade Amount: ${TRADE_AMOUNT}")
    print(f"🧠 AI Model: {AI_MODEL_TYPE.title()}")
    print(f"📱 Telegram: {'Enabled' if ENABLE_TELEGRAM_ALERTS else 'Disabled'}")
    print("=" * 60)
    print()
    print("🔧 REQUIREMENTS:")
    print("1. MetaTrader 5 must be running")
    print("2. Deriv account connected to MT5")
    print("3. Allow automated trading in MT5")
    print("4. Ensure Deriv symbols are available")
    print("=" * 60)

    # Create and start MT5 bot
    bot = DerivMT5TradingBot()

    # Setup signal handlers
    def signal_handler(signum: int, frame: Any) -> None:
        bot.running = False
        logger.info("🛑 Shutdown signal received")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await bot.start()
    except Exception as e:
        logger.error(f"❌ Main error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
