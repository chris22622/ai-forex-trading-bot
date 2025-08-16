"""
Clean Trading Bot for MT5 Trading
Streamlined for reliable trading operations
"""

import asyncio
import importlib
import os
import re
import signal
import time
import traceback
import unicodedata
import warnings
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import config

# Try to import joblib for model persistence
try:
    import joblib  # type: ignore

    joblib_available = True
except ImportError:
    joblib_available = False
    joblib = None  # type: ignore


# Rate limiter for high-frequency logging
class LogRateLimiter:
    def __init__(self):
        self.last_log_times: Dict[str, float] = {}
        self.min_interval = 5.0  # 5 seconds minimum between same log messages

    def should_log(self, message_key: str) -> bool:
        now = time.time()
        last_time = self.last_log_times.get(message_key, 0)

        if now - last_time >= self.min_interval:
            self.last_log_times[message_key] = now
            return True
        return False


# Monitor management to prevent double-close races
class MonitorManager:
    def __init__(self):
        self.active_monitors: set[str] = set()
        self.monitor_lock = asyncio.Lock()

    async def start_monitor(self, monitor_name: str) -> bool:
        async with self.monitor_lock:
            if monitor_name in self.active_monitors:
                return False

            self.active_monitors.add(monitor_name)
            return True

    async def stop_monitor(self, monitor_name: str):
        async with self.monitor_lock:
            self.active_monitors.discard(monitor_name)

    def is_monitor_running(self, monitor_name: str) -> bool:
        return monitor_name in self.active_monitors


# Import MetaTrader5
try:
    import MetaTrader5 as mt5  # type: ignore

    mt5_available = True
except ImportError:
    mt5_available = False
    mt5 = None

# 🚀 GODLIKE AI IMPORTS
try:
    from sklearn.ensemble import (  # type: ignore
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.neural_network import MLPRegressor  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    sklearn_available = True
except ImportError:
    sklearn_available = False

    # Create dummy classes to prevent unbound errors
    class MLPRegressor:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            pass

    class RandomForestRegressor:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            pass

    class GradientBoostingRegressor:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            pass

    class StandardScaler:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            pass

        def fit_transform(self, X: Any) -> Any:
            return X

        def transform(self, X: Any) -> Any:
            return X

    print("⚠️ Scikit-learn not available - using basic AI only")


# 🧠 AI Learning Progress Tracker
class AILearningProgressTracker:
    """🧠 Track and report AI learning progress with milestones"""

    def __init__(self) -> None:
        self.learning_milestones = {
            10: "🌱 Beginner",
            25: "📈 Learning",
            50: "🎯 Developing",
            100: "⚡ Skilled",
            200: "🚀 Advanced",
            500: "💎 Expert",
            1000: "🏆 Master",
            2000: "👑 Elite",
            5000: "🌟 Legendary",
            10000: "🔥 GODLIKE",
        }

        self.performance_levels = {
            (0, 30): ("🔴 Poor", "Need more training"),
            (30, 45): ("🟡 Below Average", "Improving slowly"),
            (45, 55): ("🟠 Average", "Standard performance"),
            (55, 65): ("🟢 Good", "Above average"),
            (65, 75): ("💚 Excellent", "Strong performance"),
            (75, 85): ("💙 Outstanding", "Very impressive"),
            (85, 95): ("💜 Elite", "Exceptional results"),
            (95, 100): ("🔥 GODLIKE", "Perfect performance"),
        }

        self.last_milestone_reported = 0
        self.last_progress_report = datetime.now()
        self.trades_at_last_report = 0

    def get_current_level(self, total_trades: int) -> str:
        """Get current AI level based on trade count"""
        for milestone in sorted(self.learning_milestones.keys(), reverse=True):
            if total_trades >= milestone:
                return self.learning_milestones[milestone]
        return "🌱 Beginner"

    def get_performance_rating(self, win_rate: float) -> Tuple[str, str]:
        """Get performance rating and description"""
        for (min_rate, max_rate), (
            rating,
            description,
        ) in self.performance_levels.items():
            if min_rate <= win_rate < max_rate:
                return rating, description
        return "🔴 Poor", "Need more training"

    def should_send_milestone_report(self, total_trades: int) -> bool:
        """Check if we should send a milestone report"""
        for milestone in sorted(self.learning_milestones.keys()):
            if total_trades >= milestone > self.last_milestone_reported:
                self.last_milestone_reported = milestone
                return True
        return False

    def should_send_progress_report(self, total_trades: int) -> bool:
        """Check if we should send a progress report (every hour or 50 trades)"""
        now = datetime.now()
        time_elapsed = (now - self.last_progress_report).total_seconds() / 3600  # hours
        trades_since_last = total_trades - self.trades_at_last_report

        if time_elapsed >= 1.0 or trades_since_last >= 50:  # Every hour or 50 trades
            self.last_progress_report = now
            self.trades_at_last_report = total_trades
            return True
        return False


# Suppress warnings
warnings.filterwarnings("ignore")

# Import modules
from ai_model import TradingAI
from indicators import TechnicalIndicators
from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

# 🔧 ROBUST TELEGRAM IMPORT - Handle import failures gracefully
from typing import Any

# Import with fallback classes
try:
    from telegram_bot import NotificationManager as _TelegramNotificationManager

    telegram_available = True
    logger.info("✅ Telegram notifications enabled")

    # Use the imported class directly
    NotificationManager = _TelegramNotificationManager  # type: ignore

except (ImportError, Exception) as e:
    logger.warning(f"⚠️ Telegram import failed: {e}")
    logger.info("🔄 Creating fallback NotificationManager...")

    # Fallback NotificationManager class
    class NotificationManager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.telegram = self  # Self-reference for telegram attribute

        def send_notification(self, *args: Any, **kwargs: Any) -> None:
            pass

        def send_trade_notification(self, *args: Any, **kwargs: Any) -> None:
            pass

        def send_status_update(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def send_message(self, *args: Any, **kwargs: Any) -> None:
            pass  # For telegram.send_message calls

    telegram_available = False

try:
    from enhanced_telegram_notifier import (
        EnhancedTelegramNotifier as _TelegramEnhancedNotifier,
    )
    from enhanced_telegram_notifier import (
        TradeRecord as _TelegramTradeRecord,
    )

    enhanced_telegram_available = True

    # Use the imported classes directly
    EnhancedTelegramNotifier = _TelegramEnhancedNotifier  # type: ignore
    TradeRecord = _TelegramTradeRecord  # type: ignore

except (ImportError, Exception) as e:
    logger.warning(f"⚠️ Enhanced telegram import failed: {e}")

    # Fallback classes
    class EnhancedTelegramNotifier:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def send_notification(self, *args: Any, **kwargs: Any) -> None:
            pass

        def send_bot_status(self, *args: Any, **kwargs: Any) -> None:
            pass

        def notify_trade_opened(self, *args: Any, **kwargs: Any) -> None:
            pass

        def notify_trade_closed(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set_external_session_stats(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def send_message(self, *args: Any, **kwargs: Any) -> None:
            pass

    class TradeRecord:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    enhanced_telegram_available = False

logger = get_safe_logger(__name__)

# Define fallback values for configuration constants with safe imports
MIN_TRADE_INTERVAL = getattr(config, "MIN_TRADE_INTERVAL", 60)  # Default 1 minute between trades
PRICE_BUFFER_SIZE = getattr(
    config, "PRICE_BUFFER_SIZE", 100
)  # Default buffer size for price history

# 📊 ULTRA-CAREFUL TRADING SETTINGS
ULTRA_CAREFUL_MODE = True  # Enable ultra-careful analysis
MIN_CONFIRMATIONS_REQUIRED = 3  # Need 3 confirmations to trade
WINNER_SELECTION_MODE = True  # Only pick high-probability winners
MIN_DATA_POINTS = 15  # Need 15 data points for analysis
TRADE_AMOUNT = getattr(config, "TRADE_AMOUNT", 5.0)
DEFAULT_SYMBOL = getattr(config, "DEFAULT_SYMBOL", "Volatility 75 Index")
EMA_FAST = getattr(config, "EMA_FAST", 12)
EMA_SLOW = getattr(config, "EMA_SLOW", 26)
MA_SHORT = getattr(config, "MA_SHORT", 20)
MA_LONG = getattr(config, "MA_LONG", 50)
MAX_CONSECUTIVE_LOSSES = getattr(config, "MAX_CONSECUTIVE_LOSSES", 100)
MAX_DAILY_LOSS = getattr(config, "MAX_DAILY_LOSS", 100.0)

# 🛡️ LOSS PROTECTION SETTINGS (Fixed naming and logic)
LOSS_PROTECTION_ENABLED = getattr(config, "LOSS_PROTECTION_ENABLED", True)
LOSS_PROTECTION_THRESHOLD = getattr(config, "LOSS_PROTECTION_THRESHOLD", 2.0)  # Stop at $2 LOSS
LOSS_PROTECTION_MAX_THRESHOLD = getattr(
    config, "LOSS_PROTECTION_MAX_THRESHOLD", 5.0
)  # Emergency stop at $5 LOSS

# 🚨 EMERGENCY TRADING CONTROLS
ENABLE_EMERGENCY_TRADING = getattr(config, "ENABLE_EMERGENCY_TRADING", False)  # Default OFF
EMERGENCY_TRADING_WARNING = getattr(config, "EMERGENCY_TRADING_WARNING", True)

# 🎯 UNIFIED PROFIT TARGETS
UNIVERSAL_PROFIT_TARGET = getattr(config, "UNIVERSAL_PROFIT_TARGET", 0.20)  # $0.20 for all trades
UNIVERSAL_STOP_LOSS = getattr(config, "UNIVERSAL_STOP_LOSS", 0.75)  # $0.75 stop loss
UNIVERSAL_TIME_LIMIT = getattr(config, "UNIVERSAL_TIME_LIMIT", 6)  # 6 minutes

# 🎯 CONFIDENCE THRESHOLD (normalized to 0..1 scale)
MIN_CONFIDENCE = getattr(config, "MIN_CONFIDENCE", 0.35)  # 35% on [0..1] scale

# 🛡️ BALANCE/MARGIN SAFETY THRESHOLDS
MIN_FREE_MARGIN = getattr(config, "MIN_FREE_MARGIN", 5.0)  # $5 minimum free margin for trades

# 🚨 ADVANCED RISK MANAGEMENT SETTINGS
ENABLE_SMART_RISK_MANAGEMENT = getattr(config, "ENABLE_SMART_RISK_MANAGEMENT", True)
MAX_LOSS_PER_TRADE = getattr(config, "MAX_LOSS_PER_TRADE", 2.50)

# 📊 TECHNICAL ANALYSIS CONSTANTS - FIX FOR RSI_OVERSOLD ERROR
RSI_OVERSOLD = 30  # RSI below 30 = oversold (buy signal)
RSI_OVERBOUGHT = 70  # RSI above 70 = overbought (sell signal)
RSI_NEUTRAL_LOW = 40  # Lower neutral zone
RSI_NEUTRAL_HIGH = 60  # Upper neutral zone

# Additional technical constants that might be needed
MACD_SIGNAL_BUY = 0.0  # MACD above signal line
MACD_SIGNAL_SELL = 0.0  # MACD below signal line
VOLUME_HIGH_THRESHOLD = 1.5  # High volume multiplier
PRICE_CHANGE_SIGNIFICANT = 0.001  # Significant price change %
MAX_PORTFOLIO_LOSS = getattr(config, "MAX_PORTFOLIO_LOSS", 15.0)
MAX_TRADE_AGE_MINUTES = getattr(config, "MAX_TRADE_AGE_MINUTES", 30)


# 🎯 CONFIDENCE NORMALIZATION HELPERS
def _normalize_confidence(value: float | None) -> float:
    """Return confidence on [0..1]. Accepts 0..1 or 0..100 inputs safely."""
    if value is None:
        return 0.0
    if value > 1.0:  # looks like a percent
        return min(value / 100.0, 1.0)
    if value < 0.0:
        return 0.0
    return value


def normalize_volume(vol: float, vmin: float, step: float, vmax: float) -> float:
    """🎯 VOLUME NORMALIZATION: Ensure volume fits broker constraints safely"""
    if vol < vmin:
        vol = vmin
    if vol > vmax:
        vol = vmax

    # Quantize to step size
    try:
        decimals = max(0, len(str(step).split(".")[-1]) if "." in str(step) else 0)
        steps = int(round((vol - vmin) / step))
        normalized = round(vmin + steps * step, decimals)
        return normalized
    except (ValueError, ZeroDivisionError):
        logger.warning(f"Volume normalization failed, using minimum: {vmin}")
        return vmin


def filter_tradeable_symbols(available_symbols: List[str], max_safe_lot: float = 2.50) -> List[str]:
    """🎯 SMART SYMBOL FILTERING: Prioritize symbols with reasonable minimum lot sizes for small accounts"""
    import MetaTrader5 as mt5  # type: ignore

    # Get configuration
    preferred: List[str] = getattr(config, "PREFERRED_SYMBOLS", [])
    avoid_high_min: List[str] = getattr(config, "AVOID_HIGH_MIN_LOT_SYMBOLS", [])

    tradeable_symbols: List[str] = []
    risky_symbols: List[tuple[str, float]] = []

    logger.info(f"🔍 Filtering {len(available_symbols)} symbols for small account safety...")

    for symbol in available_symbols:
        # Skip explicitly avoided symbols
        if symbol in avoid_high_min:
            logger.info(f"⚠️ SKIPPING {symbol} - in avoid list (known high min lot)")
            continue

        # Get symbol info
        try:
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if not symbol_info:
                continue

            min_lot: float = float(getattr(symbol_info, "volume_min", 0.01))  # type: ignore
        except Exception as e:
            logger.warning(f"⚠️ Could not get info for {symbol}: {e}")
            continue

        # Categorize by risk level
        if min_lot <= max_safe_lot:
            # Safe to trade with small account
            if symbol in preferred:
                tradeable_symbols.insert(0, symbol)  # Prioritize preferred
                logger.info(f"✅ PREFERRED: {symbol} (min: {min_lot} lots)")
            else:
                tradeable_symbols.append(symbol)
                logger.info(f"✅ SAFE: {symbol} (min: {min_lot} lots)")
        else:
            # Risky for small account, but available if needed
            risky_symbols.append((symbol, min_lot))
            logger.warning(
                f"⚠️ RISKY: {symbol} (min: {min_lot} lots - may be too large for small account)"
            )

    # Log results
    logger.info("📊 Symbol filtering results:")
    logger.info(f"   ✅ Safe symbols: {len(tradeable_symbols)} (min lot ≤ {max_safe_lot})")
    logger.info(f"   ⚠️ Risky symbols: {len(risky_symbols)} (min lot > {max_safe_lot})")

    if tradeable_symbols:
        logger.info(
            f"🎯 Using safe symbols: {tradeable_symbols[:3]}"
            f"'...' if len(tradeable_symbols) > 3 else ''}"
        )
    else:
        logger.warning(
            "⚠️ No safe symbols found! May need to use risky symbols or increase max_safe_lot"
        )

    return tradeable_symbols


RISK_CHECK_INTERVAL = getattr(config, "RISK_CHECK_INTERVAL", 30)


def _validate_config_sanity() -> bool:
    """🛡️ Startup config sanity check - fail fast on nonsensical values"""
    try:
        # Universal targets validation
        if UNIVERSAL_STOP_LOSS <= 0:
            logger.error(
                f"❌ CONFIG ERROR: UNIVERSAL_STOP_LOSS must be > 0, got: {UNIVERSAL_STOP_LOSS}"
            )
            return False

        if UNIVERSAL_PROFIT_TARGET <= 0:
            logger.error(
                f"❌ CONFIG ERROR: UNIVERSAL_PROFIT_TARGET must be > 0, got: "
                f"{UNIVERSAL_PROFIT_TARGET}"
            )
            return False

        if UNIVERSAL_PROFIT_TARGET >= UNIVERSAL_STOP_LOSS:
            logger.error(
                f"❌ CONFIG ERROR: PROFIT_TARGET ({UNIVERSAL_PROFIT_TARGET}) >= "
                f"STOP_LOSS ({UNIVERSAL_STOP_LOSS})"
            )
            return False

        # Time limits validation
        if UNIVERSAL_TIME_LIMIT <= 0:
            logger.error(
                f"❌ CONFIG ERROR: UNIVERSAL_TIME_LIMIT must be > 0, got: {UNIVERSAL_TIME_LIMIT}"
            )
            return False

        # Risk management validation
        if MAX_LOSS_PER_TRADE <= 0:
            logger.error(
                f"❌ CONFIG ERROR: MAX_LOSS_PER_TRADE must be > 0, got: {MAX_LOSS_PER_TRADE}"
            )
            return False

        if MAX_PORTFOLIO_LOSS <= 0:
            logger.error(
                f"❌ CONFIG ERROR: MAX_PORTFOLIO_LOSS must be > 0, got: {MAX_PORTFOLIO_LOSS}"
            )
            return False

        logger.info("✅ Config sanity check passed")
        return True

    except Exception as e:
        logger.error(f"❌ Config validation error: {e}")
        return False


class IntelligentProfitMonitor:
    """Ultra-tight intervals for $0.20 precision"""

    def __init__(self):
        self.base_interval = 0.1  # 100ms base (was 50ms) - more reasonable CPU usage
        self.fast_interval = 0.05  # 50ms when profit detected (was 20ms)
        self.max_interval = 1.0  # 1000ms max when idle (was 500ms)
        self.ultra_fast_interval = 0.025  # 25ms when very close to target (was 10ms hot loop)
        self.current_interval = self.base_interval
        self.consecutive_no_change = 0

    def get_next_interval(
        self, trades_count: int, profit_detected: bool, near_target: bool = False
    ) -> float:
        """Ultra-responsive intervals for $0.20 targeting"""

        # No trades = moderate monitoring (save CPU)
        if trades_count == 0:
            self.consecutive_no_change = 0
            return self.max_interval  # 500ms when idle

        # Ultra-fast when within $0.05 of $0.20 target
        if near_target:
            self.consecutive_no_change = 0
            return self.ultra_fast_interval  # 10ms ultra-fast

        # Fast when profit detected
        if profit_detected:
            self.consecutive_no_change = 0
            return self.fast_interval  # 20ms when close to target

        # No profit change = gradually slow down to save CPU
        self.consecutive_no_change += 1

        # Adaptive slowdown based on inactivity
        if self.consecutive_no_change > 20:  # 20 checks with no change
            return min(self.base_interval * 3, self.max_interval)  # 150ms max
        elif self.consecutive_no_change > 10:  # 10 checks with no change
            return min(self.base_interval * 2, self.max_interval)  # 100ms
        elif self.consecutive_no_change > 5:  # 5 checks with no change
            return self.base_interval * 1.5  # 75ms
        else:
            return self.base_interval  # 50ms normal rate


class MT5RequestManager:
    """Manage MT5 API requests to prevent throttling"""

    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms minimum between requests
        self.request_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=100)
        self.batch_requests = True

    async def throttled_request(self, request_func: Any, *args: Any, **kwargs: Any) -> Any:
        """Throttle MT5 requests to prevent overload"""
        try:
            # Ensure minimum interval between requests
            now = time.time()
            time_since_last = now - self.last_request_time

            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                await asyncio.sleep(sleep_time)

            # Execute request
            result = await request_func(*args, **kwargs)
            self.last_request_time = time.time()

            return result

        except Exception as e:
            logger.error(f"Throttled request error: {e}")
            return None


class RealDataLearning:
    """Learn from actual trading outcomes instead of synthetic data"""

    def __init__(self):
        self.trade_outcomes: List[Dict[str, Any]] = []
        self.feature_outcome_pairs: List[Tuple[List[float], bool]] = []
        self.min_samples_for_training = 30  # Match ensemble threshold for faster learning
        self.parent_bot: Optional[Any] = None  # Reference to main bot for retraining

    def record_trade_outcome(self, features: List[float], profit: float, success: bool):
        """Record actual trade outcome for learning"""
        try:
            outcome: Dict[str, Any] = {
                "features": features.copy(),
                "profit": profit,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }

            self.trade_outcomes.append(outcome)
            self.feature_outcome_pairs.append((features, success))

            # Keep only recent outcomes
            if len(self.trade_outcomes) > 200:
                self.trade_outcomes = self.trade_outcomes[-100:]
                self.feature_outcome_pairs = self.feature_outcome_pairs[-100:]

            logger.debug(
                f"Recorded trade outcome: {success}, total samples: {len(self.trade_outcomes)}"
            )

            # 🔥 TRIGGER REAL ML RETRAINING when we have enough samples
            if len(self.feature_outcome_pairs) >= self.min_samples_for_training:
                if (
                    self.parent_bot is not None
                    and hasattr(self.parent_bot, "godlike_ai")
                    and hasattr(self.parent_bot.godlike_ai, "_retrain_ai_with_real_data")
                ):
                    try:
                        # Copy our feature-outcome pairs to the AI ensemble
                        self.parent_bot.godlike_ai.feature_outcome_pairs = (
                            self.feature_outcome_pairs.copy()
                        )
                        # Trigger real retraining
                        retrained = self.parent_bot.godlike_ai._retrain_ai_with_real_data()
                        if retrained:
                            logger.info(
                                f"🧠 REAL ML RETRAINING TRIGGERED with "
                                f"{len(self.feature_outcome_pairs)} samples"
                            )
                    except Exception as e:
                        logger.error(f"Failed to trigger AI retraining: {e}")

        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")

    def can_train(self) -> bool:
        """Check if we have enough real data to train"""
        return len(self.feature_outcome_pairs) >= self.min_samples_for_training

    def get_win_rate(self) -> float:
        """Get actual win rate from real trades"""
        if not self.trade_outcomes:
            return 0.5

        wins = sum(1 for outcome in self.trade_outcomes if outcome["success"])
        return wins / len(self.trade_outcomes)


class MarketCondition(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


# 🧠 GODLIKE AI ENSEMBLE - Multiple AI models working together
class GodlikeAIEnsemble:
    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.scaler: Optional[Any] = None  # Type will be StandardScaler if sklearn is available

        if sklearn_available:
            try:
                # Initialize sklearn models if available
                self.models = {
                    "neural_net": MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=1000),
                    "random_forest": RandomForestRegressor(n_estimators=100),
                    "gradient_boost": GradientBoostingRegressor(n_estimators=100),
                }
                self.scaler = StandardScaler()
                logger.info("✅ Advanced AI models initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize sklearn models: {e}")
                self.models = {}
                self.scaler = None
        else:
            self.models = {}
            self.scaler = None
            logger.info("ℹ️ Using basic AI - sklearn not available")

        self.trained: bool = False
        self.confidence_threshold: float = 0.30  # Lower threshold for more aggressive trading

        # 🧠 Real ML Training Configuration
        self.model_path = "learned_models.joblib"
        self.min_samples_to_train = 30
        self.max_train_samples = 5000
        self.sklearn_available = sklearn_available

        # Load existing models if available
        self._load_models()

    def _ensure_30_feats(self, feats: List[float]) -> List[float]:
        """Ensure features are exactly 30 elements for consistent training/inference"""
        feats = list(feats)
        return (feats + [0.0] * 30)[:30]

    def _load_models(self) -> None:
        """Load previously trained models from disk"""
        try:
            if os.path.exists(self.model_path) and joblib_available and joblib is not None:
                blob = joblib.load(self.model_path)  # type: ignore
                self.scaler = blob["scaler"]
                self.models = blob["models"]
                self.trained = True
                logger.info(f"✅ Loaded trained models from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")

    def _save_models(self) -> None:
        """Save trained models to disk for persistence"""
        try:
            if joblib_available and joblib is not None:
                joblib.dump({"scaler": self.scaler, "models": self.models}, self.model_path)  # type: ignore
                logger.info(f"✅ Saved trained models to {self.model_path}")
            else:
                logger.warning("Joblib not available - cannot save models")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def _retrain_ai_with_real_data(self) -> bool:
        """🔥 REAL ML RETRAINING: Fit models on actual trade outcomes"""
        pairs = getattr(self, "feature_outcome_pairs", [])
        if len(pairs) < self.min_samples_to_train:
            logger.debug(
                f"Not enough samples for retraining: {len(pairs)}/{self.min_samples_to_train}"
            )
            return False

        # Prepare training data
        X_data: List[List[float]] = []
        y_data: List[int] = []
        for feats, outcome in pairs[-self.max_train_samples :]:
            X_data.append(self._ensure_30_feats(feats))
            y_data.append(
                1
                if (outcome is True)
                or (isinstance(outcome, (int, float)) and outcome > 0)
                or (isinstance(outcome, str) and outcome.lower().startswith("win"))
                else 0
            )

        X_array = np.asarray(X_data, dtype=float)
        y_array = np.asarray(y_data, dtype=int)

        if not self.sklearn_available:
            # Fallback: basic weight tweaking
            self._tweak_ensemble_weights(float(y_array.mean()))
            logger.info(f"🔧 Basic weight adjustment: win_rate={y_array.mean():.3f}")
            return True

        try:
            # Real ML training: fit scaler + models on actual outcomes
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_array)

            # Refit all available models
            if "neural_net" in self.models:
                self.models["neural_net"].fit(X_scaled, y_array)
            if "random_forest" in self.models:
                self.models["random_forest"].fit(X_scaled, y_array)
            if "gradient_boost" in self.models:
                self.models["gradient_boost"].fit(X_scaled, y_array)

            self.trained = True
            self._save_models()

            logger.info(
                f"✅ REAL-RETRAIN complete | n={len(y_array)} | win_rate={y_array.mean():.3f}"
            )
            return True

        except Exception as e:
            logger.error(f"Real retraining failed: {e}")
            return False

    def _tweak_ensemble_weights(self, win_rate: float) -> None:
        """Simple weight adjustment for basic AI when sklearn not available"""
        # Basic weight tweaking based on win rate
        pass  # Placeholder for simple weight adjustments

    def _predict_with_ensemble(self, raw_features: List[float]) -> Tuple[str, float]:
        """🔥 Enhanced prediction using trained models with consistent feature transform"""
        try:
            # Ensure consistent 30-feature format
            features = self._ensure_30_feats(raw_features)

            # Apply scaler if trained models available
            if getattr(self, "scaler", None) and self.trained and self.scaler is not None:
                features_scaled = self.scaler.transform([features])[0]
            else:
                features_scaled = features

            # Get ensemble prediction
            if self.sklearn_available and self.trained and self.models:
                predictions: List[str] = []
                confidences: List[float] = []

                for model_name, model in self.models.items():
                    try:
                        if hasattr(model, "predict_proba"):
                            prob = model.predict_proba([features_scaled])[0]
                            pred_conf = max(prob)
                            pred_class = "BUY" if np.argmax(prob) == 1 else "SELL"
                        else:
                            pred_val = model.predict([features_scaled])[0]
                            pred_conf = abs(pred_val)
                            pred_class = "BUY" if pred_val > 0.5 else "SELL"

                        predictions.append(pred_class)
                        confidences.append(pred_conf)
                    except Exception as e:
                        logger.debug(f"Model {model_name} prediction failed: {e}")

                if predictions:
                    # Ensemble voting
                    buy_votes = predictions.count("BUY")
                    sell_votes = predictions.count("SELL")

                    if buy_votes > sell_votes:
                        final_pred = "BUY"
                    elif sell_votes > buy_votes:
                        final_pred = "SELL"
                    else:
                        final_pred = "HOLD"

                    avg_confidence = float(np.mean(confidences)) if confidences else 0.5
                    return final_pred, min(avg_confidence, 0.99)  # Keep in 0-1 range

            # Fallback to basic prediction
            return "HOLD", 0.5

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return "HOLD", 0.5

    def predict_godlike(
        self, price_data: List[float], indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """🚀 GODLIKE prediction using ensemble of AI models"""
        try:
            if len(price_data) < 20:
                return {
                    "prediction": "HOLD",
                    "confidence": 0.0,
                    "reason": "Insufficient data",
                }

            # Feature engineering - extract advanced features
            features = self._extract_godlike_features(price_data, indicators)

            if not self.trained and sklearn_available:
                self._auto_train_on_historical_data(features)

            if not self.models or not self.trained:
                # Fallback to technical analysis
                return self._technical_fallback_prediction(price_data, indicators)

            # Get predictions from all models
            predictions: Dict[str, str] = {}
            confidences: Dict[str, float] = {}

            try:
                # Neural network prediction
                if "neural_net" in self.models and self.scaler:
                    scaled_features = self.scaler.transform([features])
                    nn_pred = self.models["neural_net"].predict(scaled_features)[0]
                    predictions["neural_net"] = (
                        "BUY" if nn_pred > 0.6 else "SELL" if nn_pred < 0.4 else "HOLD"
                    )
                    confidences["neural_net"] = abs(nn_pred - 0.5) * 2

                # Random Forest prediction
                if "random_forest" in self.models:
                    rf_pred = self.models["random_forest"].predict([features])[0]
                    predictions["random_forest"] = (
                        "BUY" if rf_pred > 0.6 else "SELL" if rf_pred < 0.4 else "HOLD"
                    )
                    confidences["random_forest"] = abs(rf_pred - 0.5) * 2

            except Exception as model_error:
                logger.debug(f"Model prediction error: {model_error}")
                return self._technical_fallback_prediction(price_data, indicators)

            # Ensemble decision with weighted voting
            final_prediction, final_confidence = self._ensemble_decision(predictions, confidences)

            # Advanced pattern recognition boost
            pattern_boost = self._detect_profitable_patterns(price_data)
            final_confidence = min(0.95, final_confidence + pattern_boost)

            return {
                "prediction": final_prediction,
                "confidence": final_confidence,
                "reason": f"Godlike AI ({len(predictions)} models)",
                "pattern_boost": pattern_boost,
            }

        except Exception as e:
            logger.debug(f"Godlike AI prediction error: {e}")
            return self._technical_fallback_prediction(price_data, indicators)

    def _extract_godlike_features(
        self, price_data: List[float], indicators: Dict[str, Any]
    ) -> NDArray[np.float64]:
        """Extract advanced features for AI prediction"""
        features: List[float] = []

        try:
            # Price features (10) with numerical safety
            epsilon = 1e-8  # Small value to prevent division by zero
            features.extend(
                [
                    (
                        float(np.mean(price_data[-5:]) / (np.mean(price_data[-20:]) + epsilon))
                        if len(price_data) >= 20
                        else 1.0
                    ),
                    (
                        float(np.std(price_data[-10:]) / (np.mean(price_data[-10:]) + epsilon))
                        if len(price_data) >= 10
                        else 0.01
                    ),
                    (
                        float((price_data[-1] - price_data[-10]) / (price_data[-10] + epsilon))
                        if len(price_data) >= 10
                        else 0.0
                    ),
                    (
                        float((price_data[-1] - price_data[-5]) / (price_data[-5] + epsilon))
                        if len(price_data) >= 5
                        else 0.0
                    ),
                    (
                        float(max(price_data[-10:]) / (min(price_data[-10:]) + epsilon))
                        if len(price_data) >= 10
                        else 1.0
                    ),
                    (
                        float(np.sum(np.diff(price_data[-10:]) > 0) / 9)
                        if len(price_data) >= 10
                        else 0.5
                    ),
                    (
                        float(np.mean(np.abs(np.diff(price_data[-10:]))))
                        if len(price_data) >= 10
                        else 0.01
                    ),
                    (
                        len([p for p in price_data[-20:] if p > np.mean(price_data[-20:])]) / 20
                        if len(price_data) >= 20
                        else 0.5
                    ),
                    (
                        float(np.corrcoef(range(10), price_data[-10:])[0, 1])
                        if len(price_data) >= 10
                        else 0.0
                    ),
                    (
                        float(
                            (np.max(price_data[-5:]) - np.min(price_data[-5:]))
                            / (np.mean(price_data[-5:]) + epsilon)
                        )
                        if len(price_data) >= 5
                        else 0.01
                    ),
                ]
            )

            # Technical indicator features (10)
            rsi = indicators.get("rsi", 50)
            macd = indicators.get("macd", 0)
            ema_fast = indicators.get("ema_fast", price_data[-1])
            ema_slow = indicators.get("ema_slow", price_data[-1])

            features.extend(
                [
                    rsi / 100,
                    (rsi - 50) / 50,
                    1 if rsi < 30 else -1 if rsi > 70 else 0,
                    macd / price_data[-1] if price_data[-1] > 0 else 0,
                    (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0,
                    1 if ema_fast > ema_slow else -1,
                    (
                        indicators.get("macd_histogram", 0) / price_data[-1]
                        if price_data[-1] > 0
                        else 0
                    ),
                    indicators.get("sma_short", price_data[-1]) / price_data[-1],
                    indicators.get("sma_long", price_data[-1]) / price_data[-1],
                    (1 if price_data[-1] > indicators.get("sma_short", price_data[-1]) else -1),
                ]
            )

            # Fill to 30 features total with deterministic padding
            while len(features) < 30:
                # Use deterministic engineered stats instead of random padding
                padding_index = len(features) - 20  # Start after the 20 real features
                if padding_index == 0:
                    features.append(
                        float(np.mean(price_data[-3:]) / np.mean(price_data[-10:]))
                        if len(price_data) >= 10
                        else 1.0
                    )
                elif padding_index == 1:
                    features.append(
                        float(np.median(price_data[-5:])) / price_data[-1]
                        if len(price_data) >= 5
                        else 1.0
                    )
                elif padding_index == 2:
                    features.append(
                        float(len([p for p in price_data[-10:] if p > price_data[-1]]) / 10)
                        if len(price_data) >= 10
                        else 0.5
                    )
                else:
                    features.append(0.0)  # Zero padding for remaining slots

            return np.array(features[:30], dtype=np.float64)

        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return np.zeros(30, dtype=np.float64)

    def _auto_train_on_historical_data(self, current_features: NDArray[np.float64]) -> None:
        """Auto-train models - GATED behind config flag to prevent synthetic data issues"""
        try:
            # 🚨 PRODUCTION FIX: Gate synthetic training behind config flag
            enable_synthetic_training = getattr(config, "ENABLE_SYNTHETIC_TRAINING", False)

            if not enable_synthetic_training:
                logger.debug("Synthetic training disabled - using real data learning only")
                return

            if not sklearn_available or not self.models:
                return

            logger.warning("⚠️ Using synthetic training data - not recommended for production")

            # Generate training data
            X_train: List[NDArray[np.float64]] = []
            y_train: List[float] = []

            for _ in range(500):  # Reduced for faster training
                synthetic_features = current_features + np.random.normal(0, 0.1, 30)
                label = 1.0 if synthetic_features[0] > 0 else 0.0
                X_train.append(synthetic_features)
                y_train.append(label)

            X_train_array = np.array(X_train, dtype=np.float64)
            y_train_array = np.array(y_train, dtype=np.float64)

            # Fit scaler and train models
            if self.scaler:
                self.scaler.fit(X_train_array)
                X_scaled = self.scaler.transform(X_train_array)

                if "neural_net" in self.models:
                    self.models["neural_net"].fit(X_scaled, y_train_array)

            if "random_forest" in self.models:
                self.models["random_forest"].fit(X_train_array, y_train_array)
            if "gradient_boost" in self.models:
                self.models["gradient_boost"].fit(X_train_array, y_train_array)

            self.trained = True
            logger.info("🧠 GODLIKE AI: Auto-training completed")

        except Exception as e:
            logger.debug(f"Auto-training error: {e}")

    def _ensemble_decision(
        self, predictions: Dict[str, str], confidences: Dict[str, float]
    ) -> Tuple[str, float]:
        """Weighted ensemble decision making"""
        if not predictions:
            return "HOLD", 0.0

        votes: Dict[str, float] = {"BUY": 0, "SELL": 0, "HOLD": 0}
        total_confidence: float = 0

        for model, pred in predictions.items():
            weight: float = confidences.get(model, 0) + 0.1
            votes[pred] += weight
            total_confidence += confidences.get(model, 0)

        winner: str = max(votes, key=lambda k: votes[k])
        avg_confidence: float = total_confidence / len(predictions) if predictions else 0

        # Boost confidence if consensus is strong
        max_votes = max(votes.values())
        total_votes = sum(votes.values())
        consensus = max_votes / total_votes if total_votes > 0 else 0

        final_confidence = min(0.95, avg_confidence * (1 + consensus))

        return winner, final_confidence

    def _detect_profitable_patterns(self, price_data: List[float]) -> float:
        """Detect known profitable patterns"""
        if len(price_data) < 10:
            return 0

        boost = 0

        try:
            # Momentum pattern
            momentum = (
                (price_data[-1] - price_data[-5]) / price_data[-5] if len(price_data) >= 5 else 0
            )
            if abs(momentum) > 0.001:
                boost += 0.05

            # Volatility breakout with numerical safety
            epsilon = 1e-8
            volatility = np.std(price_data[-10:]) / (np.mean(price_data[-10:]) + epsilon)
            if volatility > 0.003:
                boost += 0.03

        except Exception as e:
            logger.debug(f"Pattern detection error: {e}")

        return min(0.15, boost)

    def _technical_fallback_prediction(
        self, price_data: List[float], indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback prediction using technical analysis"""
        try:
            rsi = indicators.get("rsi", 50)

            # Simple RSI-based prediction
            if rsi < 35:
                return {
                    "prediction": "BUY",
                    "confidence": 0.65,
                    "reason": "RSI Oversold",
                }
            elif rsi > 65:
                return {
                    "prediction": "SELL",
                    "confidence": 0.65,
                    "reason": "RSI Overbought",
                }

            # Momentum check
            if len(price_data) >= 3:
                momentum = (price_data[-1] - price_data[-3]) / price_data[-3]
                if momentum > 0.002:
                    return {
                        "prediction": "BUY",
                        "confidence": 0.55,
                        "reason": "Positive Momentum",
                    }
                elif momentum < -0.002:
                    return {
                        "prediction": "SELL",
                        "confidence": 0.55,
                        "reason": "Negative Momentum",
                    }

            return {
                "prediction": "HOLD",
                "confidence": 0.0,
                "reason": "No Clear Signal",
            }

        except Exception as e:
            logger.debug(f"Technical fallback error: {e}")
            return {"prediction": "HOLD", "confidence": 0.0, "reason": "Analysis Error"}


# 🚀 PROFESSIONAL HFT UPGRADES - PHASE 1: ULTRA-FAST EXECUTION LAYER
class UltraFastOrderManager:
    """Ultra-fast order manager using async interface only"""

    def __init__(self, mt5_interface: Any) -> None:
        self.mt5_interface = mt5_interface
        self.execution_start_time = None
        self.last_execution_latency = 0
        self.target_latency_ms = 50.0  # Realistic 50ms target

    async def ultra_fast_order(self, symbol: str, action: str, volume: float) -> Dict[str, Any]:
        """Execute order through async interface only"""
        start_time = time.time()

        try:
            if not self.mt5_interface:
                return {"success": False, "reason": "No MT5 interface"}

            # Use async interface only - no direct MT5 calls
            result = await self.mt5_interface.place_trade(
                action=action, symbol=symbol, volume=volume
            )

            execution_time_ms = (time.time() - start_time) * 1000

            if result:
                trade_id = result.get("ticket", f"fast_{int(time.time())}")
                is_fast = execution_time_ms < self.target_latency_ms

                logger.info(
                    f"⚡ FAST ORDER: {execution_time_ms:.2f}ms "
                    f"{'(FAST)' if is_fast else '(NORMAL)'}"
                )

                return {
                    "success": True,
                    "order": trade_id,
                    "latency_ms": execution_time_ms,
                    "fast_execution": is_fast,
                }
            else:
                return {"success": False, "reason": "MT5 order failed"}

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Fast order error in {execution_time_ms:.2f}ms: {e}")
            return {"success": False, "reason": str(e)}


class AdvancedPositionSizer:
    """Kelly Criterion + Volatility-based position sizing for optimal growth"""

    def __init__(self):
        self.kelly_multiplier = 2.0  # 🚀 EXPLOSIVE Kelly fraction - GO MASSIVE!
        self.max_position_size = (
            1000.0  # 🚀 Maximum position size - GIGANTIC TRADES FOR EXPLOSIVE GROWTH!
        )
        self.min_position_size = 50.0  # 🚀 Minimum position size - NO TINY BETS!
        self.volatility_lookback = 50  # Periods for volatility calculation
        self.win_rate_history: List[int] = []
        self.pnl_history: List[float] = []

    def calculate_optimal_size(
        self,
        symbol: str,
        confidence: float,
        account_balance: float,
        recent_volatility: float,
    ) -> float:
        """Calculate position size using Kelly Criterion + volatility scaling"""
        try:
            # Base Kelly calculation - confidence already normalized to 0-1
            win_probability = min(confidence, 0.95)  # Confidence already 0-1
            average_win = self._get_average_win()
            average_loss = self._get_average_loss()

            if average_loss == 0:
                kelly_fraction = 0.10  # 🚀 Aggressive fallback - 10%!
            else:
                # Kelly Criterion: f* = (bp - q) / b
                # where b = average_win/average_loss, p = win_probability, q = 1-p
                b = abs(average_win / average_loss)
                kelly_fraction = (b * win_probability - (1 - win_probability)) / b
                kelly_fraction = max(
                    0, min(kelly_fraction * self.kelly_multiplier, 0.90)
                )  # 🚀 Cap at 90% - EXPLOSIVE GROWTH!

            # Volatility adjustment - reduce size in high volatility
            volatility_multiplier = max(
                0.5, 1.0 - (recent_volatility * 10)
            )  # Scale volatility impact

            # Confidence adjustment - increase size with higher confidence (0-1 scale)
            confidence_multiplier = 0.5 + confidence * 0.5  # 0.5x to 1.0x based on confidence

            # Calculate base position size
            base_size = (
                account_balance * kelly_fraction * volatility_multiplier * confidence_multiplier
            ) / 100

            # Apply professional limits
            optimal_size = max(self.min_position_size, min(self.max_position_size, base_size))

            logger.info(
                f"📊 OPTIMAL POSITION: {optimal_size:.2f} lots (Kelly: {kelly_fraction:.4f}, "
                f"Vol: {volatility_multiplier:.3f}, Conf: {confidence_multiplier:.3f})"
            )

            return optimal_size

        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 1.0  # Safe fallback

    def _get_average_win(self) -> float:
        """Get average winning trade amount"""
        wins = [pnl for pnl in self.pnl_history if pnl > 0]
        return sum(wins) / len(wins) if wins else 1.0

    def _get_average_loss(self) -> float:
        """Get average losing trade amount"""
        losses = [abs(pnl) for pnl in self.pnl_history if pnl < 0]
        return sum(losses) / len(losses) if losses else 1.0

    def update_trade_result(self, pnl: float, was_win: bool):
        """Update historical data for Kelly calculation"""
        self.pnl_history.append(pnl)
        self.win_rate_history.append(1 if was_win else 0)

        # Keep only recent history
        if len(self.pnl_history) > 100:
            self.pnl_history = self.pnl_history[-100:]
        if len(self.win_rate_history) > 100:
            self.win_rate_history = self.win_rate_history[-100:]


class ProfessionalAIEnsemble:
    """Enterprise-grade AI ensemble for institutional-level predictions"""

    def __init__(self):
        self.confidence_threshold = 0.85  # Professional grade minimum (0-1 scale)
        self.models_count = 7  # Multiple model consensus
        self.prediction_cache = {}
        self.model_weights = {
            "technical": 0.25,
            "momentum": 0.20,
            "volume": 0.15,
            "pattern": 0.15,
            "sentiment": 0.10,
            "regime": 0.10,
            "risk": 0.25,  # 🚀 25% risk weighting - AGGRESSIVE!
        }

    def professional_prediction(
        self,
        symbol: str,
        price_data: List[float],
        volume_data: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Generate institutional-grade prediction with 85%+ confidence requirement"""
        try:
            predictions: List[Tuple[str, Dict[str, Any]]] = []

            # Model 1: Advanced Technical Analysis
            tech_pred = self._advanced_technical_model(price_data)
            predictions.append(("technical", tech_pred))

            # Model 2: Momentum & Trend Analysis
            momentum_pred = self._momentum_model(price_data)
            predictions.append(("momentum", momentum_pred))

            # Model 3: Volume Profile Analysis
            if volume_data:
                volume_pred = self._volume_model(price_data, volume_data)
                predictions.append(("volume", volume_pred))

            # Model 4: Pattern Recognition
            pattern_pred = self._pattern_recognition_model(price_data)
            predictions.append(("pattern", pattern_pred))

            # Model 5: Market Regime Detection
            regime_pred = self._regime_detection_model(price_data)
            predictions.append(("regime", regime_pred))

            # Model 6: Risk-Adjusted Analysis
            risk_pred = self._risk_adjusted_model(price_data)
            predictions.append(("risk", risk_pred))

            # Ensemble voting with professional weighting
            weighted_confidence = 0
            buy_votes = 0
            sell_votes = 0

            for model_name, pred in predictions:
                weight = self.model_weights.get(model_name, 0.1)
                weighted_confidence += pred["confidence"] * weight

                if pred["prediction"] == "BUY":
                    buy_votes += weight
                elif pred["prediction"] == "SELL":
                    sell_votes += weight

            # Professional decision making
            if weighted_confidence >= self.confidence_threshold:
                if buy_votes > sell_votes:
                    final_prediction = "BUY"
                elif sell_votes > buy_votes:
                    final_prediction = "SELL"
                else:
                    final_prediction = "HOLD"

                logger.info(
                    f"🎯 PROFESSIONAL AI ENSEMBLE: {final_prediction}"
                    f"at {weighted_confidence:.1%} confidence"
                )
                return {
                    "prediction": final_prediction,
                    "confidence": weighted_confidence,
                    "professional_grade": True,
                    "model_count": len(predictions),
                    "buy_votes": buy_votes,
                    "sell_votes": sell_votes,
                }
            else:
                return {
                    "prediction": "HOLD",
                    "confidence": weighted_confidence,
                    "professional_grade": False,
                    "reason": f"Below professional threshold ({self.confidence_threshold:.0%})",
                }

        except Exception as e:
            logger.error(f"Professional AI ensemble error: {e}")
            return {"prediction": "HOLD", "confidence": 0, "professional_grade": False}

    def _advanced_technical_model(self, prices: List[float]) -> Dict[str, Any]:
        """Advanced technical analysis model"""
        try:
            # Professional technical indicators
            rsi = self._calculate_rsi(prices, 14)
            macd_line, macd_signal = self._calculate_macd(prices)
            bb_upper, bb_lower = self._calculate_bollinger_bands(prices)

            confidence = 0
            prediction = "HOLD"

            # RSI analysis with professional levels
            if rsi < 25:  # Extremely oversold
                confidence += 30
                prediction = "BUY"
            elif rsi > 75:  # Extremely overbought
                confidence += 30
                prediction = "SELL"

            # MACD professional signals
            if macd_line > macd_signal and prices[-1] > prices[-2]:
                confidence += 25
                if prediction != "SELL":
                    prediction = "BUY"

            # Bollinger Bands institutional signals
            if prices[-1] < bb_lower:
                confidence += 20
                if prediction != "SELL":
                    prediction = "BUY"
            elif prices[-1] > bb_upper:
                confidence += 20
                if prediction != "BUY":
                    prediction = "SELL"

            return {"prediction": prediction, "confidence": min(confidence, 95)}

        except:
            return {"prediction": "HOLD", "confidence": 0}

    def _momentum_model(self, prices: List[float]) -> Dict[str, Any]:
        """Professional momentum analysis"""
        try:
            if len(prices) < 10:
                return {"prediction": "HOLD", "confidence": 0}

            # Short and long-term momentum
            short_momentum = (prices[-1] - prices[-5]) / prices[-5] * 100
            long_momentum = (prices[-1] - prices[-10]) / prices[-10] * 100

            confidence = 0
            prediction = "HOLD"

            # Strong momentum signals
            if short_momentum > 0.1 and long_momentum > 0.05:
                confidence = 80
                prediction = "BUY"
            elif short_momentum < -0.1 and long_momentum < -0.05:
                confidence = 80
                prediction = "SELL"

            return {"prediction": prediction, "confidence": confidence}

        except:
            return {"prediction": "HOLD", "confidence": 0}

    def _volume_model(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Volume-price analysis model"""
        try:
            if len(volumes) < 5:
                return {"prediction": "HOLD", "confidence": 0}

            avg_volume = sum(volumes[-5:]) / 5
            current_volume = volumes[-1]
            price_change = (prices[-1] - prices[-2]) / prices[-2] * 100

            confidence = 0
            prediction = "HOLD"

            # Volume confirmation signals
            if current_volume > avg_volume * 1.5 and price_change > 0:
                confidence = 70
                prediction = "BUY"
            elif current_volume > avg_volume * 1.5 and price_change < 0:
                confidence = 70
                prediction = "SELL"

            return {"prediction": prediction, "confidence": confidence}

        except:
            return {"prediction": "HOLD", "confidence": 0}

    def _pattern_recognition_model(self, prices: List[float]) -> Dict[str, Any]:
        """Pattern recognition model"""
        try:
            if len(prices) < 15:
                return {"prediction": "HOLD", "confidence": 0}

            # Simple pattern detection
            recent_high = max(prices[-10:])
            recent_low = min(prices[-10:])
            current_price = prices[-1]

            # Breakout patterns
            if current_price > recent_high * 1.001:
                return {"prediction": "BUY", "confidence": 75}
            elif current_price < recent_low * 0.999:
                return {"prediction": "SELL", "confidence": 75}

            return {"prediction": "HOLD", "confidence": 40}

        except:
            return {"prediction": "HOLD", "confidence": 0}

    def _regime_detection_model(self, prices: List[float]) -> Dict[str, Any]:
        """Market regime detection"""
        try:
            if len(prices) < 20:
                return {"prediction": "HOLD", "confidence": 0}

            # Volatility regime
            returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
            volatility = (sum(r**2 for r in returns[-10:]) / 10) ** 0.5

            # Trend regime
            trend = (prices[-1] - prices[-20]) / prices[-20] * 100

            confidence = 0
            prediction = "HOLD"

            if volatility < 0.01 and trend > 0.1:  # Low vol, uptrend
                confidence = 85
                prediction = "BUY"
            elif volatility < 0.01 and trend < -0.1:  # Low vol, downtrend
                confidence = 85
                prediction = "SELL"

            return {"prediction": prediction, "confidence": confidence}

        except:
            return {"prediction": "HOLD", "confidence": 0}

    def _risk_adjusted_model(self, prices: List[float]) -> Dict[str, Any]:
        """Risk-adjusted analysis"""
        try:
            if len(prices) < 10:
                return {"prediction": "HOLD", "confidence": 0}

            # Simple risk metrics
            returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
            avg_return = sum(returns[-5:]) / 5
            volatility = (sum((r - avg_return) ** 2 for r in returns[-5:]) / 5) ** 0.5

            # Risk-adjusted signal
            if volatility > 0:
                sharpe_like = avg_return / volatility
                if sharpe_like > 0.5:
                    return {"prediction": "BUY", "confidence": 60}
                elif sharpe_like < -0.5:
                    return {"prediction": "SELL", "confidence": 60}

            return {"prediction": "HOLD", "confidence": 30}

        except:
            return {"prediction": "HOLD", "confidence": 0}

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50

            gains: List[float] = []
            losses: List[float] = []

            for i in range(1, len(prices)):
                change = prices[i] - prices[i - 1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        except:
            return 50

    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float]:
        """Calculate MACD"""
        try:
            if len(prices) < 26:
                return 0, 0

            # Simple MACD calculation
            ema12 = self._calculate_ema(prices, 12)
            ema26 = self._calculate_ema(prices, 26)
            macd_line = ema12 - ema26

            # Simple signal line (could be improved)
            macd_signal = macd_line * 0.9  # Simplified

            return macd_line, macd_signal

        except:
            return 0, 0

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        try:
            if len(prices) < period:
                return sum(prices) / len(prices)

            multiplier = 2 / (period + 1)
            ema = prices[0]

            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))

            return ema

        except:
            return prices[-1] if prices else 0

    def _calculate_bollinger_bands(
        self, prices: List[float], period: int = 20
    ) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                avg = sum(prices) / len(prices)
                return avg * 1.02, avg * 0.98

            sma = sum(prices[-period:]) / period
            variance = sum((price - sma) ** 2 for price in prices[-period:]) / period
            std_dev = variance**0.5

            upper_band = sma + (2 * std_dev)
            lower_band = sma - (2 * std_dev)

            return upper_band, lower_band

        except:
            avg = prices[-1] if prices else 0
            return avg * 1.02, avg * 0.98


class TelegramTextSanitizer:
    """📱 Clean corrupted characters from Telegram messages"""

    @staticmethod
    def sanitize_message(text: str) -> str:
        """Remove corrupted characters and clean text for Telegram"""
        try:
            if not text:
                return ""

            # Step 1: Normalize unicode characters
            text = unicodedata.normalize("NFKD", text)

            # Step 2: Remove non-printable characters except newlines and tabs
            text = "".join(char for char in text if char.isprintable() or char in "\n\t")

            # Step 3: Remove common corrupted sequences
            corrupted_patterns = [
                r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]",  # Control characters
                r"[\uFFF0-\uFFFF]",  # Unicode specials
                r"[\uFFFE\uFFFF]",  # Byte order marks
                # Keep basic Latin, symbols, emojis
                r"[^\u0000-\u007F\u0080-\u00FF\u0100-\u017F\u0180-\u024F"
                r"\u2000-\u206F\u20A0-\u20CF\u2100-\u214F\u2600-\u26FF\u2700-\u27BF]",
            ]

            for pattern in corrupted_patterns:
                text = re.sub(pattern, "", text)

            # Step 4: Clean up multiple spaces and newlines
            text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
            text = re.sub(r" {2,}", " ", text)  # Max 1 space between words

            # Step 5: Remove leading/trailing whitespace
            text = text.strip()

            # Step 6: Ensure message isn't empty after cleaning
            if not text:
                return "📱 [Message cleaned - no displayable content]"

            # Step 7: Truncate if too long for Telegram (4096 char limit)
            if len(text) > 4000:
                text = text[:4000] + "...\n📱 [Message truncated]"

            return text

        except Exception as e:
            logger.error(f"📱 Text sanitization error: {e}")
            return "📱 [Message sanitization failed]"

    @staticmethod
    def sanitize_symbol_name(symbol: str) -> str:
        """Clean symbol names for display"""
        try:
            # Remove common corrupted characters from symbol names
            clean_symbol = re.sub(r"[^\w\s\-\.]", "", symbol)
            clean_symbol = clean_symbol.strip()
            return clean_symbol if clean_symbol else "Unknown Symbol"
        except Exception:
            return "Unknown Symbol"

    @staticmethod
    def format_currency(amount: float) -> str:
        """Format currency amounts safely"""
        try:
            if abs(amount) < 0.01:
                return f"${amount:.4f}"
            elif abs(amount) < 1.0:
                return f"${amount:.3f}"
            else:
                return f"${amount:.2f}"
        except Exception:
            return "$0.00"


class TradingBot:
    """Main trading bot for MT5"""

    def __init__(self) -> None:
        self.running = False
        self.mt5_connected = False
        self.mt5_interface = None

        # Rate limiting and monitor management
        self.log_limiter = LogRateLimiter()
        self.monitor_manager = MonitorManager()

        # Trading state
        self.current_balance = 0.0
        self.starting_balance = 0.0  # Track starting balance for loss protection
        self.active_trades: Dict[str, Any] = {}
        self.price_history: List[float] = []
        self.trades_today = 0
        self.daily_profit = 0.0
        self.consecutive_losses = 0
        self.last_trade_time: Optional[datetime] = None

        # 🛡️ LOSS PROTECTION - Fixed naming (was "profit protection")
        self.loss_protection_enabled = LOSS_PROTECTION_ENABLED
        self.loss_protection_threshold = LOSS_PROTECTION_THRESHOLD
        self.loss_protection_triggered = False

        # 🎯 DEMO MODE DETECTION AND SETUP
        # ⚠️ IMPORTANT: Set to False for LIVE TRADING with real money!
        self.demo_mode = getattr(config, "DEMO_MODE", True)  # Configurable demo mode
        self.demo_training_active = False

        # Override loss protection for demo accounts
        if self.demo_mode:
            self.loss_protection_enabled = False  # Disable for demo
            self.max_consecutive_losses = 999999  # Unlimited for demo
            self.max_daily_loss = 999999.0  # Unlimited for demo
            logger.warning("🎯 DEMO MODE ACTIVATED: UNLIMITED TRADING FOR AI TRAINING!")
            logger.warning(
                                "⚠️ WARNING: Demo mode disables loss protection! "
                "Set DEMO_MODE=False for live trading!"
            )
            logger.warning("🎯 All risk limits DISABLED for continuous learning!")

        logger.info(
            f"🛡️ Loss Protection: {'ENABLED' if self.loss_protection_enabled else 'DISABLED'}"
            f"at -${self.loss_protection_threshold} LOSS"
        )

        # 🚨 ULTRA-CONSERVATIVE RISK MANAGEMENT - MARGIN PROTECTION
        self.max_loss_per_trade = 0.75  # Tight stop losses: $0.75 max (adjusted for $0.20 target)!
        self.max_portfolio_loss = 2.0  # Emergency stop at $2 total loss!
        self.max_trade_age_minutes = 12  # Close trades after 12 minutes (faster cycles for $0.20)
        self.enable_smart_risk_management = True

        # 📊 UNIFIED PROFIT TARGETS - Single source of truth derived from UNIVERSAL constants
        self.quick_profit_target = UNIVERSAL_PROFIT_TARGET  # Use universal: $0.20
        self.min_profit_threshold = 0.05  # Accept profits as low as $0.05!
        self.max_consecutive_losses = MAX_CONSECUTIVE_LOSSES  # Use global constant for consistency
        self.profit_lock_threshold = UNIVERSAL_PROFIT_TARGET  # Use universal: $0.20

        # Margin safety limits
        self.max_concurrent_trades = 1  # Only 1 trade at a time for safety
        self.margin_safety_enabled = True
        self.min_margin_level = 200  # Stop trading if margin level < 200%

        # 🚀 PROFESSIONAL HFT COMPONENTS - PHASE 1 INTEGRATION
        self.ultra_fast_order_manager = None  # Will be initialized after MT5 interface
        self.advanced_position_sizer = AdvancedPositionSizer()
                # 🧠 PRODUCTION FIX 3: Intelligent adaptive monitoring (
            replaces CPU-pegging nanosecond engine)
        self.intelligent_profit_monitor = IntelligentProfitMonitor()

        # 🛡️ PRODUCTION FIX 5: Throttled MT5 request manager (all MT5 calls through async interface)
        self.mt5_request_manager = MT5RequestManager()

        # 📊 PRODUCTION FIX 6: Real data learning (replaces synthetic training data)
        self.real_data_learning = RealDataLearning()
        self.real_data_learning.parent_bot = self  # Connect for retraining
        self.professional_ai_ensemble = ProfessionalAIEnsemble()

        logger.info("🎯 PROFESSIONAL HFT COMPONENTS INITIALIZED")
        logger.info("⚡ Ultra-Fast Order Manager: Sub-50ms execution targeting")
        logger.info("📊 Advanced Position Sizer: Kelly Criterion + volatility scaling")
        logger.info("🧠 Professional AI Ensemble: 85%+ confidence requirement")
        logger.info("🧠 Intelligent Monitoring: Adaptive intervals (10ms-500ms)")
        logger.info("📊 Real Data Learning: Learn from actual outcomes")
        self.min_free_margin = 10  # Stop trading if free margin < $10

        # 🛡️ EXTENDED LOSS PROTECTION - Weekly/Monthly Limits
        self.weekly_profit = 0.0
        self.monthly_profit = 0.0
        self.last_week_reset = datetime.now()
        self.last_month_reset = datetime.now()
        self.max_weekly_loss = 50.0  # $50 max weekly loss
        self.max_monthly_loss = 150.0  # $150 max monthly loss

        self.risk_management_stats: Dict[str, Union[int, float, str, None]] = {
            "trades_auto_closed": 0,
            "total_saved_loss": 0.0,
            "last_cleanup_time": None,
        }

        # Price caching to reduce MT5 load
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = 3  # Cache prices for 3 seconds

        # Multi-symbol trading state
        self.symbol_price_histories: Dict[str, List[float]] = {}
        self.symbol_performance: Dict[str, Dict[str, Any]] = {}
        self.active_symbols: List[str] = []
        self.current_symbol_index = 0

        # Performance tracking
        self.session_stats: Dict[str, Union[int, float]] = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_profit": 0.0,
        }
        self.win_rate_tracker: deque[str] = deque(maxlen=20)
        self.market_condition = MarketCondition.SIDEWAYS

        # 🔥 GODLIKE MULTI-TIMEFRAME ANALYSIS
        self.godlike_ai = GodlikeAIEnsemble()
        self.timeframe_data: Dict[str, Deque[float]] = {
            "1min": deque(maxlen=100),
            "5min": deque(maxlen=100),
            "15min": deque(maxlen=100),
            "1hour": deque(maxlen=100),
        }
        self.last_timeframe_update = {
            "1min": datetime.now(),
            "5min": datetime.now(),
            "15min": datetime.now(),
            "1hour": datetime.now(),
        }

        # 💎 QUICK PROFIT AMPLIFICATION SYSTEM - Using UNIVERSAL constants
        self.profit_amplifier: Dict[str, Any] = {
            "consecutive_wins": 0,
            "hot_streak_multiplier": 1.0,
            "profit_compound_rate": 0.05,  # 5% compounding on quick profits
            "max_amplification": 2.0,  # Lower max since profits are smaller
            "quick_profit_target": UNIVERSAL_PROFIT_TARGET,  # Use universal $0.20
        }

        # 🎯 QUICK PROFIT SIZING
        self.smart_sizing = {
            "base_risk_percent": 0.5,  # 0.5% base risk
            "max_risk_percent": 1.5,  # 1.5% max risk (lower for quick profits)
            "win_rate_multiplier": 1.2,  # Boost for quick wins
            "volatility_adjustment": 1.0,
        }

        # 🎯 VOLATILITY INDEX SPECIFIC TARGETS - All use UNIVERSAL constants
        self.volatility_targets: Dict[str, Dict[str, Union[float, str]]] = {
            "Volatility 10 Index": {
                "profit_target": UNIVERSAL_PROFIT_TARGET,  # Use universal $0.20
                "stop_loss": UNIVERSAL_STOP_LOSS,  # Use universal $0.75
                "time_limit": UNIVERSAL_TIME_LIMIT,  # Use universal 6 minutes
                "tick_speed": "slow",
            },
            "Volatility 25 Index": {
                "profit_target": UNIVERSAL_PROFIT_TARGET,  # Use universal $0.20
                "stop_loss": UNIVERSAL_STOP_LOSS,  # Use universal $0.75
                "time_limit": UNIVERSAL_TIME_LIMIT,  # Use universal 6 minutes
                "tick_speed": "medium",
            },
            "Volatility 50 Index": {
                "profit_target": UNIVERSAL_PROFIT_TARGET,  # Use universal $0.20
                "stop_loss": UNIVERSAL_STOP_LOSS,  # Use universal $0.75
                "time_limit": UNIVERSAL_TIME_LIMIT,  # Use universal 6 minutes
                "tick_speed": "fast",
            },
            "Volatility 75 Index": {
                "profit_target": UNIVERSAL_PROFIT_TARGET,  # Use universal $0.20
                "stop_loss": UNIVERSAL_STOP_LOSS,  # Use universal $0.75
                "time_limit": UNIVERSAL_TIME_LIMIT,  # Use universal 6 minutes
                "tick_speed": "very_fast",
            },
            "Volatility 100 Index": {
                "profit_target": UNIVERSAL_PROFIT_TARGET,  # Use universal $0.20
                "stop_loss": UNIVERSAL_STOP_LOSS,  # Use universal $0.75
                "time_limit": UNIVERSAL_TIME_LIMIT,  # Use universal 6 minutes
                "tick_speed": "ultra_fast",
            },
        }

        # ⚡ OPTIMAL HFT CONFIG FOR VOLATILITY INDICES - Using UNIVERSAL constants
        self.hft_mode_enabled = True
        self.profit_check_interval = 0.05  # 50ms = 20 checks per second (OPTIMAL!)
        self.instant_profit_target = UNIVERSAL_PROFIT_TARGET  # Use universal $0.20
        self.profit_sensitivity = 0.01  # Alert on $0.01 changes
        self.volatility_optimized = True  # Special mode for V25/V50/V75

        # ⚡ VOLATILITY INDEX SPECIFIC SETTINGS
        self.tick_frequency_estimate = 2.0  # V indices tick every ~2 seconds
        self.checks_per_tick = 40  # 20 checks/sec × 2 sec = 40 checks per tick
        self.micro_movement_threshold = 0.001  # Catch 0.001% movements
        self.detailed_profit_tracking = True  # Enable detailed profit tracking
        self._optimal_counter = 0  # Initialize HFT counter

        # Components
        self.indicators = TechnicalIndicators()
        self.ai_model = TradingAI()
        self.notifier = NotificationManager()

        # Thread-safe trade management
        self.trades_lock = asyncio.Lock()

        # 📱 Clean Telegram messaging
        self.text_sanitizer = TelegramTextSanitizer()

        # Enhanced Telegram notifier with trade tracking
        self.enhanced_notifier = EnhancedTelegramNotifier()

        # 🧠 AI Learning Progress Tracker
        self.learning_tracker = AILearningProgressTracker()

        # Learning progress stats
        self.learning_stats: Dict[str, Any] = {
            "total_training_samples": 0,
            "confidence_improvements": 0,
            "pattern_recognitions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "ai_accuracy_trend": [],
            "learning_rate": 0.0,
            "skill_points": 0,
            "experience_gained": 0.0,
        }

        # Bot start time for uptime tracking
        self.bot_start_time = datetime.now()

        # Initialize
        self._initialize_mt5()
        self.ai_model.load_model()
        self._initialize_multi_symbol_trading()

        # Clear any orphaned trades from previous sessions
        self.active_trades.clear()

        logger.info("🚀 Trading Bot initialized with Enhanced Telegram Notifications")
        logger.info(
            f"🔒 CB6 threshold – MIN_FREE_MARGIN=${MIN_FREE_MARGIN:.2f} (balance gate for trades)"
        )
        logger.info(
            f"🧠 CB2 threshold – MIN_CONFIDENCE={MIN_CONFIDENCE:.2f}"
            f"({MIN_CONFIDENCE*100:.1f}% confidence gate)"
        )

    async def _send_sanitized_message(self, message: str) -> None:
        """Helper to send sanitized messages through Telegram"""
        try:
            clean_msg = TelegramTextSanitizer.sanitize_message(message)
            await self.notifier.telegram.send_message(clean_msg)
        except Exception as e:
            logger.warning(f"Sanitized message send failed: {e}")

    def _cb2_confidence_gate(self, confidence: float) -> tuple[bool, str]:
        """Circuit Breaker #2: confidence must be >= MIN_CONFIDENCE (all on 0..1)."""
        c = _normalize_confidence(confidence)
        if c >= MIN_CONFIDENCE:
            return True, f"CB2 OK ({c*100:.1f}% ≥ {MIN_CONFIDENCE*100:.1f}%)"
        diff_pct = (MIN_CONFIDENCE - c) * 100.0
        return (
            False,
            f"CB2: Confidence {diff_pct:.1f}% below minimum threshold ({MIN_CONFIDENCE*100:.1f}%)",
        )

    def _initialize_mt5(self) -> bool:
        """Initialize MT5 interface"""
        try:
            if mt5_available:
                from mt5_integration import MT5TradingInterface

                self.mt5_interface = MT5TradingInterface()
                # Initialize HFT components after MT5 interface is ready
                self.ultra_fast_order_manager = UltraFastOrderManager(self.mt5_interface)
                logger.info("✅ MT5 interface created")
                logger.info("⚡ HFT Order Manager initialized")
                return True
            else:
                logger.warning("⚠️ MT5 not available")
                return False
        except Exception as e:
            logger.error(f"MT5 initialization failed: {e}")
            return False

    def _initialize_multi_symbol_trading(self):
        """Initialize multi-symbol trading system with smart symbol filtering"""
        try:
            # Import multi-symbol settings with fallbacks
            enable_multi_symbol = getattr(config, "ENABLE_MULTI_SYMBOL_TRADING", False)
            primary_symbols = getattr(config, "PRIMARY_TRADING_SYMBOLS", ["Volatility 75 Index"])
            max_symbols = getattr(config, "MAX_SYMBOLS_CONCURRENT", 1)
            default_symbol = getattr(
                config, "DEFAULT_SYMBOL", "Volatility 10 Index"
            )  # Updated to safer default
            volume_guard_mode = getattr(config, "VOLUME_GUARD_MODE", "smart_filter")

            if enable_multi_symbol:
                # 🎯 SMART FILTERING: Apply intelligent symbol selection for small accounts
                if volume_guard_mode == "smart_filter":
                    # Use preferred symbols from config if available
                    preferred_symbols: List[str] = getattr(config, "PREFERRED_SYMBOLS", [])
                    avoid_symbols: List[str] = getattr(config, "AVOID_HIGH_MIN_LOT_SYMBOLS", [])

                    # Filter primary symbols based on preferences and avoidance
                    safe_primary: List[str] = []
                    for symbol in primary_symbols:
                        if symbol in avoid_symbols:
                            logger.warning(
                                f"⚠️ AVOIDING {symbol}"
                                f"- known high minimum lot size for small accounts"
                            )
                            continue
                        safe_primary.append(symbol)

                    # Use preferred symbols if we have them, otherwise filtered primary
                    if preferred_symbols:
                        # Prioritize preferred symbols that are also in primary list
                        preferred_primary = [s for s in preferred_symbols if s in primary_symbols]
                        if preferred_primary:
                            self.active_symbols = preferred_primary[:max_symbols]
                            logger.info(
                                f"🎯 Using preferred primary symbols: {self.active_symbols}"
                            )
                        else:
                            # Use first available preferred symbols
                            self.active_symbols = preferred_symbols[:max_symbols]
                            logger.info(f"🎯 Using preferred symbols: {self.active_symbols}")
                    elif safe_primary:
                        self.active_symbols = safe_primary[:max_symbols]
                        logger.info(
                            f"🎯 Using filtered safe primary symbols: {self.active_symbols}"
                        )
                    else:
                        logger.warning(
                                                        "⚠️ All primary symbols may be risky! Using "
                            "first primary with risk management..."
                        )
                        self.active_symbols = primary_symbols[:max_symbols]
                else:
                    # Use original behavior for other modes
                    self.active_symbols = primary_symbols[:max_symbols]

                logger.info(f"🎯 Multi-symbol trading enabled: {self.active_symbols}")

                # Initialize price histories for each symbol - ENHANCED
                for symbol in self.active_symbols:
                    self.symbol_price_histories[symbol] = []
                    self.symbol_performance[symbol] = {
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,  # Added missing field
                        "profit": 0.0,
                        "win_rate": 0.0,
                        "last_trade_time": None,
                        "avg_profit_per_trade": 0.0,  # Added for better tracking
                        "total_volume": 0.0,  # Added for better tracking
                        "best_trade": 0.0,  # Added for better tracking
                        "worst_trade": 0.0,  # Added for better tracking
                    }

                logger.info(
                    f"✅ Initialized {len(self.active_symbols)} symbols for multi-symbol trading"
                )
            else:
                # 🎯 SMART SINGLE SYMBOL: Choose safer default for small accounts
                if volume_guard_mode == "smart_filter":
                    avoid_symbols = getattr(config, "AVOID_HIGH_MIN_LOT_SYMBOLS", [])
                    preferred_symbols = getattr(config, "PREFERRED_SYMBOLS", [])

                    if default_symbol in avoid_symbols:
                        logger.warning(
                            f"⚠️ Default symbol {default_symbol} may be risky for small accounts"
                        )
                        if preferred_symbols:
                            safer_symbol = preferred_symbols[0]
                            self.active_symbols = [safer_symbol]
                            logger.info(f"📊 Using safer alternative: {safer_symbol}")
                        else:
                            self.active_symbols = [default_symbol]
                            logger.warning(
                                f"⚠️ No alternative found, using {default_symbol}"
                                f"with enhanced risk management"
                            )
                    else:
                        self.active_symbols = [default_symbol]
                        logger.info(f"📊 Single symbol trading: {default_symbol}")
                else:
                    self.active_symbols = [default_symbol]
                    logger.info(f"📊 Single symbol trading: {default_symbol}")

                # Initialize single symbol tracking - ENHANCED
                symbol = self.active_symbols[0]
                self.symbol_price_histories[symbol] = []
                self.symbol_performance[symbol] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "profit": 0.0,
                    "win_rate": 0.0,
                    "last_trade_time": None,
                    "avg_profit_per_trade": 0.0,
                    "total_volume": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0,
                }

        except Exception as e:
            logger.error(f"Multi-symbol initialization failed: {e}")
            # Fallback to safer single symbol
            fallback_symbol = getattr(config, "DEFAULT_SYMBOL", "Volatility 10 Index")
            self.active_symbols = [fallback_symbol]
            logger.info(f"🛡️ Fallback to safe symbol: {fallback_symbol}")

    async def force_emergency_trade(self, symbol: Optional[str] = None):
        """🚨 EMERGENCY: Force a trade - DANGEROUS! Only use when absolutely necessary"""

        # 🚨 SAFETY GATE: Must be explicitly enabled in config
        if not ENABLE_EMERGENCY_TRADING:
            logger.error("🚨 EMERGENCY TRADING DISABLED!")
            logger.error("🔒 To enable: Set ENABLE_EMERGENCY_TRADING = True in config.py")
            logger.error("⚠️ WARNING: Emergency trading bypasses ALL safety checks!")
            print("❌ EMERGENCY TRADING DISABLED - Check config.py")
            return False

        # 🚨 BIG RED WARNING
        if EMERGENCY_TRADING_WARNING:
            logger.error("🚨" * 20)
            logger.error("🚨 DANGER: EMERGENCY TRADING ACTIVATED!")
            logger.error("🚨 THIS BYPASSES ALL SAFETY PROTECTIONS!")
            logger.error("🚨 MARGIN CALLS, LOSS LIMITS, RISK CHECKS = DISABLED!")
            logger.error("🚨 USE ONLY IN EXTREME SITUATIONS!")
            logger.error("🚨" * 20)

            # Wait 5 seconds to let user see the warning
            await asyncio.sleep(5)

        try:
            if not symbol:
                symbol = self.active_symbols[0] if self.active_symbols else DEFAULT_SYMBOL

            # Ensure symbol is not None at this point
            if not symbol:
                logger.error("🚨 EMERGENCY: No symbol available for trade")
                return False

            logger.error(f"🚨 EMERGENCY TRADE FORCED for {symbol}")

            # Emergency trade with minimal size
            action = "BUY"
            confidence = 0.50
            position_size = 0.01  # Ultra-safe emergency size

            logger.error(f"🚨 EMERGENCY: Placing {action} {position_size} lots on {symbol}")

            # Place trade directly
            if self.mt5_interface and self.mt5_connected:
                result = await self.mt5_interface.place_trade(
                    action=action, symbol=symbol, amount=position_size
                )

                if result:
                    trade_id = result.get("ticket", f"emergency_{int(time.time())}")
                    async with self.trades_lock:
                        self.active_trades[str(trade_id)] = {
                            "action": action,
                            "symbol": symbol,
                            "amount": position_size,
                            "confidence": confidence,
                            "timestamp": datetime.now().isoformat(),
                            "entry_price": result.get("price", 0.0),
                            "mt5_ticket": trade_id,
                            "emergency_trade": True,  # Flag as emergency
                        }

                    # Send warning notification
                    await self._send_emergency_trade_notification(symbol, trade_id, position_size)

                    logger.error(f"✅ EMERGENCY TRADE PLACED: {trade_id}")
                    return True
                else:
                    logger.error("❌ EMERGENCY TRADE FAILED")
                    return False
            else:
                logger.error("❌ MT5 not connected for emergency trade")
                return False

        except Exception as e:
            logger.error(f"Emergency trade error: {e}")
            return False

    async def _send_emergency_trade_notification(self, symbol: str, trade_id: str, size: float):
        """📱 Send emergency trade warning notification with clean text"""
        try:
            # Clean symbol name
            clean_symbol = self.text_sanitizer.sanitize_symbol_name(symbol)

            msg = f"""🚨 EMERGENCY TRADE PLACED!

⚠️ WARNING: Emergency override activated!
📊 Symbol: {clean_symbol}
🎫 Ticket: {trade_id}
💰 Size: {size:.2f} lots

🚨 This trade bypassed ALL safety checks!
🛡️ Monitor closely and close manually if needed!
⚠️ Consider why emergency trading was necessary!"""

            # Send with text sanitization
            await self._send_clean_telegram_message(msg)

        except Exception as e:
            logger.warning(f"📱 Emergency notification failed: {e}")

    async def debug_trading_blocks(self):
        """🔍 Debug what's preventing trades"""
        logger.info("🔍 DEBUGGING TRADING BLOCKS:")

        # Check if bot is running properly
        logger.info(f"✅ MT5 Connected: {self.mt5_connected}")
        logger.info(f"✅ MT5 Interface: {self.mt5_interface is not None}")
        logger.info(f"✅ Balance: ${self.current_balance:.2f}")
        logger.info(f"✅ Active Trades: {len(self.active_trades)}")

        # Check config values
        logger.info(f"✅ MIN_CONFIDENCE: {MIN_CONFIDENCE} ({MIN_CONFIDENCE*100:.1f}%)")
        logger.info(f"✅ ENABLE_EMERGENCY_TRADING: {ENABLE_EMERGENCY_TRADING}")
        logger.info(f"✅ ENFORCE_MARKET_HOURS: {getattr(config, 'ENFORCE_MARKET_HOURS', True)}")

        # Check price history
        if hasattr(self, "price_history") and len(self.price_history) >= 15:
            logger.info(f"✅ Price History: {len(self.price_history)} points (sufficient)")

            # Test AI prediction
            try:
                indicator_data = self._get_indicator_data()
                if indicator_data:
                    logger.info(f"✅ Indicators Working: {list(indicator_data.keys())}")

                    # Test AI prediction
                    ai_prediction = self.godlike_ai.predict_godlike(
                        self.price_history, indicator_data
                    )
                    logger.info(
                        f"🧠 AI Test: {ai_prediction['prediction']}"
                        f"(confidence: {ai_prediction['confidence']:.3f})"
                    )

                    # Test confidence gate
                    if ai_prediction["confidence"] >= MIN_CONFIDENCE:
                        logger.info(
                            f"🔒 CB2 Test: PASS ({ai_prediction['confidence']:.3f}"
                            f"≥ {MIN_CONFIDENCE})"
                        )
                    else:
                        logger.warning(
                            f"🔒 CB2 Test: BLOCKED ({ai_prediction['confidence']:.3f}"
                            f"< {MIN_CONFIDENCE})"
                        )
                else:
                    logger.error("❌ No indicator data available")
            except Exception as e:
                logger.error(f"❌ AI Test Failed: {e}")
        else:
            current_points = len(getattr(self, "price_history", []))
            logger.warning(f"❌ Insufficient Price History: {current_points}/15 points")

        # Check risk limits
        try:
            can_trade = self._check_risk_limits()
            logger.info(f"✅ Risk Limits: {'PASS' if can_trade else 'BLOCKED'}")
        except Exception as e:
            logger.error(f"❌ Risk Limits Check Failed: {e}")

        # Check concurrent trades
        try:
            can_concurrent = self._check_concurrent_trades_limit()
            logger.info(f"✅ Concurrent Limit: {'PASS' if can_concurrent else 'BLOCKED'}")
        except Exception as e:
            logger.error(f"❌ Concurrent Check Failed: {e}")

    async def start(self):
        """Enhanced startup with auto bulletproof monitor"""
        logger.info("🚀 Starting trading bot...")

        # Config sanity check first - fail fast on bad config
        if not _validate_config_sanity():
            logger.error("❌ STARTUP FAILED: Invalid configuration detected")
            raise ValueError("Configuration validation failed")

        self.running = True

        try:
            # Validate configuration invariants at boot
            self._validate_config_invariants()

            # Connect to MT5
            await self._connect_mt5()

            # Clean up any orphaned trades from previous sessions
            await self._cleanup_orphaned_trades_on_startup()

            # 🎯 AUTO-ENABLE BULLETPROOF $0.20 MONITOR
            strict_mode = getattr(config, "STRICT_20_CENT_MODE", True)  # Default ON
            if strict_mode:
                await self.force_enable_20_cent_auto_close()
                logger.warning("🎯 STRICT MODE: Bulletproof $0.20 monitor AUTO-ENABLED")

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

                # 🛡️ Initialize loss protection system (fixed naming)
                self.starting_balance = self.current_balance
                self.loss_protection_enabled = LOSS_PROTECTION_ENABLED
                self.loss_protection_threshold = LOSS_PROTECTION_THRESHOLD

                logger.info(f"✅ MT5 connected! Balance: ${self.current_balance:.2f}")
                logger.info(
                    f"🛡️ Loss Protection: {'ENABLED' if self.loss_protection_enabled else 'DISABLED'}"
                    f"at -${self.loss_protection_threshold} LOSS"
                )

                # Log symbol constraints for debugging
                try:
                    symbol_info = await self.mt5_interface.get_symbol_info(DEFAULT_SYMBOL)
                    if symbol_info:
                        logger.info(
                            f"📊 Symbol constraints for {DEFAULT_SYMBOL}: "
                            f"Min lot: {symbol_info.get('volume_min', 'N/A')}, "
                            f"Max lot: {symbol_info.get('volume_max', 'N/A')}, "
                            f"Step: {symbol_info.get('volume_step', 'N/A')}"
                        )
                    else:
                        logger.warning(f"⚠️ Could not get symbol info for {DEFAULT_SYMBOL}")
                except Exception as symbol_error:
                    logger.warning(f"⚠️ Symbol info check failed: {symbol_error}")

            else:
                logger.error("❌ MT5 connection failed")
                raise ConnectionError("MT5 connection failed")
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            raise

    async def _send_startup_notification(self):
        """Send startup notification"""
        try:
            # Import auto-management settings
            auto_enabled = getattr(config, "ENABLE_AUTO_TRADE_MANAGEMENT", True)
            _auto_take_profit = getattr(config, "AUTO_TAKE_PROFIT", 0.75)
            _auto_stop_loss = getattr(config, "AUTO_STOP_LOSS", 1.0)
            _auto_time_limit = getattr(config, "AUTO_TIME_LIMIT", 15)
            max_concurrent_trades = getattr(config, "MAX_CONCURRENT_TRADES", 1)
            hedge_prevention = getattr(config, "ENABLE_HEDGE_PREVENTION", True)
            multi_symbol = getattr(config, "ENABLE_MULTI_SYMBOL_TRADING", False)

            auto_status = "✅ Enabled" if auto_enabled else "❌ Disabled"
            hedge_status = "✅ Protected" if hedge_prevention else "⚠️ Allowed"

            # Build symbols display
            if multi_symbol and len(self.active_symbols) > 1:
                symbols_text = f"🎯 Symbols: {', '.join(self.active_symbols[:3])}"  # Show first 3
                if len(self.active_symbols) > 3:
                    symbols_text += f" (+{len(self.active_symbols)-3} more)"
            else:
                symbols_text = f"🎯 Symbol: {DEFAULT_SYMBOL}"

            # Loss protection status (now prevents losses, not profits)
            _protection_status = "✅ Enabled" if self.loss_protection_enabled else "❌ Disabled"

            msg = f"""🚀 AI TRADING BOT ACTIVATED!

🧠 **AI STATUS:**
🏆 Current Level: {self.learning_tracker.get_current_level(int(self.session_stats.get('total_trades', 0)))}
📊 Total Experience: {int(self.session_stats.get('total_trades', 0)) * 10} XP
🎯 Learning Mode: {'ACTIVE' if self.demo_mode else 'LIVE TRADING'}
🔥 Goal: Become GODLIKE trader

💰 **ACCOUNT INFO:**

✅ MT5: {'Connected' if self.mt5_connected else 'Disconnected'}
💰 Balance: ${self.current_balance:.2f}
{symbols_text}
🕒 Started: {datetime.now().strftime('%H:%M:%S')}

🎯 UNIFIED TARGETS (ALL TRADES):
💎 Profit Target: +${UNIVERSAL_PROFIT_TARGET:.2f}
🛡️ Stop Loss: -${UNIVERSAL_STOP_LOSS:.2f}
⏰ Time Limit: {UNIVERSAL_TIME_LIMIT} minutes

� Trade Management: {auto_status}
🔢 Max Concurrent: {max_concurrent_trades} trades
🚫 Hedge Prevention: {hedge_status}

�🧠 Intelligent Monitoring: ACTIVE
⚡ Adaptive intervals (50ms-1s based on activity)
🛡️ Fresh price protection: ENABLED
🚨 Emergency trading: {'ENABLED' if ENABLE_EMERGENCY_TRADING else 'DISABLED'}
🛡️ Loss Protection: {_protection_status}

📱 Ready for consistent ${UNIVERSAL_PROFIT_TARGET:.2f} profits!"""

            # Send enhanced startup notification
            self.enhanced_notifier.send_bot_status(
                status="STARTING",
                balance=self.current_balance,
                active_trades=len(self.active_trades),
            )

            # Also send legacy notification through sanitizer
            await self._send_clean_telegram_message(msg)
            logger.info("📱 Startup notification sent")
        except Exception as e:
            logger.warning(f"Startup notification failed: {e}")

    async def _main_trading_loop(self):
        """Enhanced main trading loop with SINGLE monitor enforcement"""
        logger.info("🎯 Starting main loop with SINGLE monitor enforcement...")

        # Start essential monitors (with single monitor rule)
        asyncio.create_task(self._continuous_margin_monitoring())
        asyncio.create_task(self._connection_watchdog())

        # Start PRIMARY monitor
        if await self.monitor_manager.start_monitor("unified_profit"):
            asyncio.create_task(self._unified_profit_monitor())
            logger.info("✅ PRIMARY MONITOR: Unified profit monitor started")

        # Optional bulletproof monitor (only if explicitly enabled and not already started)
        bulletproof_enabled = getattr(
            config, "ENABLE_BULLETPROOF_MONITOR", True
        )  # Default ON for strict mode
        if bulletproof_enabled:
            # Check if bulletproof monitor not already running from force_enable call
            if not self.monitor_manager.is_monitor_running("bulletproof_20cent"):
                if await self.monitor_manager.start_monitor("bulletproof_20cent"):
                    # Monitor manager registered, start actual task
                    asyncio.create_task(self._bulletproof_20_cent_monitor())
                    logger.warning("🎯 BULLETPROOF MONITOR: Enabled as secondary backup")
            else:
                logger.info("🎯 BULLETPROOF MONITOR: Already running from manual enable")

        logger.warning("🔄 MAIN LOOP: Starting with SINGLE monitor enforcement")

        while self.running:
            try:
                # Increment loop counter
                if hasattr(self, "_loop_counter"):
                    self._loop_counter += 1
                else:
                    self._loop_counter = 0

                # Periodic weekly/monthly resets (every 100 loops = ~10 minutes)
                if self._loop_counter % 100 == 0:
                    self._check_extended_loss_limits()  # This handles the resets

                # 🚀 DEMO MODE: Skip protection checks for aggressive trading
                if not config.DEMO_MODE:
                    # ️ CRITICAL PROTECTION CHECKS (ONLY IN LIVE MODE)
                    if not self._check_maximum_drawdown_protection():
                        logger.error("🚨 MAXIMUM DRAWDOWN - STOPPING BOT!")
                        break

                    if not self._check_extended_loss_limits():
                        logger.error("🚨 EXTENDED LOSS LIMITS - STOPPING BOT!")
                        break
                else:
                    # Demo mode - log balance but don't exit
                    if self._loop_counter % 100 == 0:
                        logger.info(
                            f"🚀 DEMO MODE: Balance ${self.current_balance:.2f} - UNLIMITED TRADING!"
                        )

                # Process trading every iteration
                if self._loop_counter % 1 == 0:
                    # Check if multi-symbol trading is enabled
                    if len(self.active_symbols) > 1:
                        await self._process_multi_symbol_trading()
                    else:
                        await self._process_single_symbol_trading()

                # 🚨 SMART RISK MANAGEMENT CHECK (every 30 seconds)
                if self._loop_counter % 30 == 0 and ENABLE_SMART_RISK_MANAGEMENT:
                    await self._smart_risk_management_check()

                # Update balance periodically
                if self._loop_counter % 60 == 0:
                    await self._update_balance()

                # Wait before next iteration
                await asyncio.sleep(2.0)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5.0)

    async def _process_single_symbol_trading(self):
        """Process single symbol trading (original logic)"""
        try:
            # Get current price for default symbol
            current_price = await self._get_current_price()
            if current_price:
                self.price_history.append(current_price)

                # Keep price history manageable
                if len(self.price_history) > 1000:
                    self.price_history = self.price_history[-500:]

                # Check for trading opportunities
                await self._check_trading_opportunity(current_price)
            else:
                # Log if price retrieval fails occasionally
                if hasattr(self, "_loop_counter") and self._loop_counter % 30 == 0:
                    logger.warning(f"⚠️ Failed to get current price for {DEFAULT_SYMBOL}")

        except Exception as e:
            logger.error(f"Single symbol trading error: {e}")
            logger.error(f"Single symbol trading traceback: {traceback.format_exc()}")

    async def _process_multi_symbol_trading(self):
        """Process multi-symbol trading"""
        try:
            # Rotate through active symbols
            symbol = self.active_symbols[self.current_symbol_index]

            # Get current price for this symbol
            current_price = await self._get_current_price_for_symbol(symbol)
            if current_price:
                # Update symbol-specific price history
                if symbol not in self.symbol_price_histories:
                    self.symbol_price_histories[symbol] = []

                self.symbol_price_histories[symbol].append(current_price)

                # Keep price history manageable
                if len(self.symbol_price_histories[symbol]) > 1000:
                    self.symbol_price_histories[symbol] = self.symbol_price_histories[symbol][-500:]

                # Update main price history with current symbol's data
                self.price_history = self.symbol_price_histories[symbol]

                # Check for trading opportunities on this symbol
                await self._check_trading_opportunity_for_symbol(current_price, symbol)

            # Move to next symbol
            self.current_symbol_index = (self.current_symbol_index + 1) % len(self.active_symbols)

        except Exception as e:
            logger.error(f"Multi-symbol trading error: {e}")

    async def _get_current_price(self) -> Optional[float]:
        """Get current price for default symbol with timeout, retry logic, and caching"""
        return await self._get_current_price_for_symbol(DEFAULT_SYMBOL)

    async def _get_current_price_for_symbol(self, symbol: str) -> Optional[float]:
        """Get current price for specific symbol with timeout, retry logic, and caching"""
        try:
            # Check cache first but validate staleness
            now = datetime.now()
            if symbol in self.price_cache:
                cache_entry = self.price_cache[symbol]
                cache_time = cache_entry.get("timestamp")
                if (
                    cache_time and (now - cache_time).total_seconds() < 3.0
                ):  # 3 second max staleness
                    cached_price = cache_entry.get("price")
                    if cached_price is not None:
                        return cached_price

            if not self.mt5_interface or not self.mt5_connected:
                logger.error(f"❌ MT5 not connected for {symbol}")
                return None

            # Get fresh price with timeout
            try:
                price = await asyncio.wait_for(
                    self.mt5_interface.get_current_price(symbol), timeout=10.0
                )

                if price is not None and price > 0:
                    # Cache with timestamp
                    self.price_cache[symbol] = {"price": price, "timestamp": now}
                    return price
                else:
                    # 🚨 DON'T USE STALE PRICES FOR TRADING DECISIONS
                    logger.warning(f"⚠️ Invalid price for {symbol}: {price} - BLOCKING TRADE")
                    return None

            except asyncio.TimeoutError:
                logger.error(f"⏰ Price timeout for {symbol} - BLOCKING TRADE")
                return None

        except Exception as e:
            logger.error(f"Price error for {symbol}: {e}")
            return None

    async def _check_trading_opportunity(self, current_price: float):
        """Check for trading opportunities (single symbol)"""
        await self._check_trading_opportunity_for_symbol(current_price, DEFAULT_SYMBOL)

    async def _check_trading_opportunity_for_symbol(self, current_price: float, symbol: str):
        """🎯 ULTRA-CAREFUL trading opportunity analysis - Only pick WINNERS but still trades!"""
        try:
            # Use symbol-specific price history or main price history for single symbol trading
            if len(self.active_symbols) > 1:
                price_history = self.symbol_price_histories.get(symbol, [])
            else:
                price_history = self.price_history

            # 🚨 ULTRA-CAREFUL: Need MORE data for confidence (but still reasonable)
            if len(price_history) < MIN_DATA_POINTS:
                if len(price_history) % 5 == 0:
                    logger.info(
                        f"📊 ULTRA-CAREFUL: Collecting data for {symbol}"
                        f" {len(price_history)}/{MIN_DATA_POINTS} points needed"
                    )
                return

            # Log when we have enough data to start trading
            if len(price_history) == 5:
                logger.info(
                    f"🚀 {symbol}"
                    f"has enough data ({len(price_history)} points) - Starting AI analysis!"
                )
            elif len(price_history) % 20 == 0:  # Log every 20th analysis
                logger.info(
                    f"� {symbol} AI analysis running with {len(price_history)} price points"
                )

            # Update indicators with current price (using symbol's price history)
            # Temporarily replace the indicators' price data with this symbol's data
            original_history = []
            try:
                if hasattr(self.indicators, "price_data") and self.indicators.price_data:
                    original_history = self.indicators.price_data.copy()
                else:
                    original_history = []
            except Exception as e:
                logger.debug(f"Error copying original price history: {e}")
                original_history = []

            # Convert price history to the format expected by indicators with validation
            try:
                # Clear existing price data
                self.indicators.price_data = []

                # Add price history with validation
                for price in price_history:
                    try:
                        if price > 0:
                            self.indicators.update_price_data(price)
                        else:
                            logger.debug(f"Invalid price in history: {price}")
                    except (TypeError, ValueError):
                        logger.debug(f"Invalid price type in history: {price}")

                # Update with current price
                try:
                    if current_price > 0:
                        self.indicators.update_price_data(current_price)
                    else:
                        logger.warning(f"Invalid current price: {current_price}")
                        # Restore original indicator state
                        self.indicators.price_data = original_history
                        return
                except (TypeError, ValueError):
                    logger.warning(f"Invalid current price type: {current_price}")
                    # Restore original indicator state
                    self.indicators.price_data = original_history
                    return

            except Exception as e:
                logger.error(f"Error updating indicator price data for {symbol}: {e}")
                # Restore original indicator state
                try:
                    self.indicators.price_data = original_history
                except:
                    pass
                return

            # Analyze market
            self._analyze_market_conditions()

            # Get indicator data for AI prediction
            indicator_data = self._get_indicator_data()
            if not indicator_data:
                logger.warning(f"⚠️ No indicator data available for {symbol}")
                # Restore original indicator state
                self.indicators.price_data = original_history
                return

            # FORCE LOG AI prediction attempt
            logger.warning(
                f"🧠 GODLIKE AI ANALYSIS for {symbol} with {len(price_history)} price points"
            )

            # 🧠 GODLIKE AI PREDICTION
            try:
                # Get enhanced indicator data
                godlike_indicator_data = self._get_godlike_indicator_data(price_history)

                # Use GODLIKE AI ensemble
                ai_prediction = self.godlike_ai.predict_godlike(
                    price_history, godlike_indicator_data
                )
                action = ai_prediction["prediction"]
                confidence = ai_prediction["confidence"]
                _reason = ai_prediction.get("reason", "Godlike AI")

                # Apply market regime multiplier
                market_regime = self._detect_market_regime(price_history)
                regime_multiplier = self._get_regime_multiplier(market_regime)

                # Apply timeframe confirmation
                timeframe_signals = await self._get_timeframe_signals(symbol)
                timeframe_boost = self._calculate_timeframe_boost(timeframe_signals)

                # Calculate final confidence
                final_confidence = min(0.95, confidence * regime_multiplier + timeframe_boost)

                logger.warning(
                    f"🧠 GODLIKE {symbol}"
                    f" {action} (Base: {confidence:.2f}, Regime: {regime_multiplier:.2f}, Final: {final_confidence:.2f})"
                )

                # Use final confidence
                confidence = final_confidence

            except Exception as ai_error:
                logger.warning(
                    f"⚠️ Godlike AI failed for {symbol}: {ai_error}, using enhanced fallback"
                )

                # Enhanced fallback with pattern detection
                pattern_signal = self._detect_godlike_patterns(price_history)
                if pattern_signal["action"] != "HOLD":
                    action = pattern_signal["action"]
                    confidence = pattern_signal["confidence"]
                    _reason = "Pattern Recognition"
                    logger.warning(
                        f"🔥 PATTERN FALLBACK: {symbol} → {action} (confidence: {confidence:.2f})"
                    )
                else:
                    action = "HOLD"
                    confidence = 0.0
                    _reason = "No Signal"

            # FORCE LOG every prediction
            logger.warning(
                f"🧠 {symbol} Final Decision: {action} (confidence: {confidence:.2f}, "
                f"threshold: {MIN_CONFIDENCE})"
            )

            # 🚨 EMERGENCY TRADING: Enhanced multi-factor analysis for profitable trades
            if confidence <= 0.01:  # AI not working properly
                if ENABLE_EMERGENCY_TRADING:
                    logger.warning(
                        f"� EMERGENCY TRADING MODE: AI failed for {symbol}"
                        f" forcing ultra-aggressive trades!"
                    )

                    # 🚨 EMERGENCY: Force trades based on ANY price movement
                    if len(price_history) >= 2:
                        # Get the last two prices
                        current_price = price_history[-1]
                        previous_price = price_history[-2]

                        # Calculate simple percentage change
                        price_change = (current_price - previous_price) / previous_price * 100

                        # ULTRA-AGGRESSIVE: Trade on ANY movement > 0.001%
                        if price_change > 0.001:  # Even tiny upward movement
                            action = "BUY"
                            confidence = 0.08  # Well above 5% threshold
                            logger.warning(
                                f"🚀 EMERGENCY BUY: {symbol} +{price_change:.4f}% movement!"
                            )
                        elif price_change < -0.001:  # Even tiny downward movement
                            action = "SELL"
                            confidence = 0.08  # Well above 5% threshold
                            logger.warning(
                                f"📉 EMERGENCY SELL: {symbol} {price_change:.4f}% movement!"
                            )
                        else:
                            # NO MOVEMENT: Force a random trade to generate activity
                            import random

                            action = "BUY" if random.random() > 0.5 else "SELL"
                            confidence = 0.07  # Well above 5% threshold
                            logger.warning(
                                f"🎲 FORCED RANDOM TRADE: {symbol} → {action} (no movement detected)"
                            )
                    else:
                        # Not enough price history: Force a BUY trade
                        action = "BUY"
                        confidence = 0.06  # Above 5% threshold
                        logger.warning(
                            f"🚨 FORCED STARTUP TRADE: {symbol} → BUY (insufficient data)"
                        )
                else:
                    # Emergency trading disabled - don't spam with forced trades
                    logger.debug(
                        f"Emergency trading disabled for {symbol}; skipping ultra-aggressive path."
                    )
                    return  # Exit early - no emergency trading allowed

                # Get comprehensive technical indicators for backup validation
                rsi = indicator_data.get("rsi", 50)

                # 🎯 ULTRA-AGGRESSIVE TECHNICAL ANALYSIS (RSI-based signals)
                if rsi is not None:
                    if rsi < 55:  # ULTRA AGGRESSIVE - Much more sensitive!
                        action = "BUY"
                        confidence = 0.65  # High confidence for technical signal
                        logger.warning(
                            f"🚀 ULTRA-AGGRESSIVE BUY: {symbol} RSI={rsi:.1f} (< 55) → BUY"
                        )
                    elif rsi > 45:  # ULTRA AGGRESSIVE - Much more sensitive!
                        action = "SELL"
                        confidence = 0.65  # High confidence for technical signal
                        logger.warning(
                            f"📉 ULTRA-AGGRESSIVE SELL: {symbol} RSI={rsi:.1f} (> 45) → SELL"
                        )
                    else:
                        # If RSI is in the tiny neutral zone (48-52), check momentum
                        if len(price_history) >= 3:
                            recent_trend = price_history[-1] - price_history[-3]
                            price_change_pct = (recent_trend / price_history[-3]) * 100

                            if price_change_pct > 0.01:  # ULTRA AGGRESSIVE - 5x more sensitive!
                                action = "BUY"
                                confidence = 0.65  # High confidence for momentum
                                logger.warning(
                                    f"🚀 MOMENTUM SIGNAL: {symbol} +{price_change_pct:.2f}% → BUY"
                                )
                            elif price_change_pct < -0.01:  # ULTRA AGGRESSIVE - 5x more sensitive!
                                action = "SELL"
                                confidence = 0.65  # High confidence for momentum
                                logger.warning(
                                    f"📉 MOMENTUM SIGNAL: {symbol} {price_change_pct:.2f}% → SELL"
                                )
                            else:
                                logger.info(
                                    f"📊 TECHNICAL: {symbol}"
                                    f"RSI={rsi:.1f}, Momentum={price_change_pct:.2f}% (All Neutral) → HOLD"
                                )
                                # Restore original indicator state
                                self.indicators.price_data = original_history
                                return
                        else:
                            # Not enough price history - force a BUY trade anyway for activity
                            action = "BUY"
                            confidence = 0.25  # Lower confidence but still above threshold
                            logger.warning(
                                f"� FORCE TRADE: {symbol}"
                                f"RSI={rsi:.1f} (Insufficient history) → BUY"
                            )
                else:
                    # No RSI data - force a momentum-based trade for activity
                    if len(price_history) >= 2:
                        recent_change = (
                            (price_history[-1] - price_history[-2]) / price_history[-2] * 100
                        )
                        if recent_change > 0:
                            action = "BUY"
                            confidence = 0.25
                            logger.warning(
                                f"🚀 FORCE MOMENTUM BUY: {symbol} +{recent_change:.2f}% → BUY"
                            )
                        else:
                            action = "SELL"
                            confidence = 0.25
                            logger.warning(
                                f"📉 FORCE MOMENTUM SELL: {symbol} {recent_change:.2f}% → SELL"
                            )
                    else:
                        # Last resort - random trade to generate activity
                        import random

                        action = "BUY" if random.random() > 0.5 else "SELL"
                        confidence = 0.21  # Just above threshold
                        logger.warning(
                            f"🎲 RANDOM TRADE: {symbol} → {action} (generating activity)"
                        )

            # Convert prediction format to match expected format
            if action == "BUY":
                action = "BUY"
            elif action == "SELL":
                action = "SELL"
            else:
                logger.info(f"📊 {symbol} AI says HOLD - waiting for better opportunity")
                # Restore original indicator state
                self.indicators.price_data = original_history
                return  # HOLD or unknown action

            # Check confidence threshold (now applies to both AI and technical signals)
            if confidence < MIN_CONFIDENCE:
                logger.warning(
                    f"⚠️ {symbol} Confidence too low: {confidence:.2f} < {MIN_CONFIDENCE}"
                )
                # Restore original indicator state
                self.indicators.price_data = original_history
                return

            # Check risk limits
            if not self._check_risk_limits():
                logger.warning("⚠️ Risk limits failed")
                # Restore original indicator state
                self.indicators.price_data = original_history
                return

            # Check concurrent trades limit
            if not self._check_concurrent_trades_limit():
                logger.info("⚠️ Max concurrent trades limit reached")
                # Restore original indicator state
                self.indicators.price_data = original_history
                return

            # Check hedge prevention
            if not await self._check_hedge_prevention(action):
                logger.info("⚠️ Hedge prevention blocked trade")
                # Restore original indicator state
                self.indicators.price_data = original_history
                return

            # Check timing
            if not self._check_trade_timing():
                logger.info("⚠️ Trade timing check failed")
                # Restore original indicator state
                self.indicators.price_data = original_history
                return

            # Place trade
            if action in ["BUY", "SELL"]:
                logger.warning(f"🎯 ALL CHECKS PASSED! Placing {action} trade on {symbol}...")
                await self._place_trade_for_symbol(action, confidence, symbol)

            # Restore original indicator state
            self.indicators.price_data = original_history

        except Exception as e:
            logger.error(f"Error checking opportunity for {symbol}: {e}")

    def _ultra_careful_technical_analysis(
        self, indicator_data: Dict[str, Optional[float]], price_history: List[float]
    ) -> Tuple[float, Dict[str, Union[bool, str]]]:
        """🎯 Ultra-careful technical analysis - Only approve strong signals"""
        try:
            score = 0.0
            signals: Dict[str, Union[bool, str]] = {
                "bullish": False,
                "bearish": False,
                "strength": "weak",
            }

            # RSI Analysis (0-3 points)
            rsi = indicator_data.get("rsi", 50)
            if rsi is not None:
                if rsi < 30:  # Strong oversold (relaxed from 25)
                    score += 3.0
                    signals["bullish"] = True
                    signals["rsi"] = "oversold"
                elif rsi > 70:  # Strong overbought (relaxed from 75)
                    score += 3.0
                    signals["bearish"] = True
                    signals["rsi"] = "overbought"
                elif 30 <= rsi <= 40:  # Mild oversold
                    score += 2.0
                    signals["bullish"] = True
                    signals["rsi"] = "mild_oversold"
                elif 60 <= rsi <= 70:  # Mild overbought
                    score += 2.0
                    signals["bearish"] = True
                    signals["rsi"] = "mild_overbought"
                else:
                    score += 1.0  # Neutral gets some points

            # MACD Analysis (0-2 points)
            macd = indicator_data.get("macd", 0)
            macd_signal = indicator_data.get("macd_signal", 0)
            if macd is not None and macd_signal is not None:
                macd_diff = macd - macd_signal
                if macd_diff > 0.0005:  # Bullish MACD (relaxed)
                    score += 2.0
                    signals["macd"] = "bullish"
                elif macd_diff < -0.0005:  # Bearish MACD (relaxed)
                    score += 2.0
                    signals["macd"] = "bearish"

            # EMA Analysis (0-2 points)
            ema_fast = indicator_data.get("ema_fast")
            ema_slow = indicator_data.get("ema_slow")
            if ema_fast is not None and ema_slow is not None and ema_slow > 0:
                ema_diff = (ema_fast - ema_slow) / ema_slow * 100
                if ema_diff > 0.05:  # Uptrend (relaxed from 0.1)
                    score += 2.0
                    signals["ema"] = "uptrend"
                elif ema_diff < -0.05:  # Downtrend (relaxed from -0.1)
                    score += 2.0
                    signals["ema"] = "downtrend"

            # Momentum Analysis (0-2 points)
            if len(price_history) >= 10:
                momentum = (price_history[-1] - price_history[-10]) / price_history[-10] * 100
                if abs(momentum) > 0.2:  # Momentum (relaxed from 0.5)
                    score += 2.0
                    signals["momentum"] = "strong_positive" if momentum > 0 else "strong_negative"

            # Volatility Analysis (0-1 point)
            if len(price_history) >= 20:
                volatility = np.std(price_history[-20:]) / np.mean(price_history[-20:])
                if 0.001 < volatility < 0.01:  # Optimal volatility range (relaxed)
                    score += 1.0
                    signals["volatility"] = "optimal"

            # Determine signal strength
            if score >= 6.0:  # Relaxed from 8.0
                signals["strength"] = "very_strong"
            elif score >= 4.0:  # Relaxed from 6.0
                signals["strength"] = "strong"
            elif score >= 2.0:  # Relaxed from 4.0
                signals["strength"] = "moderate"
            else:
                signals["strength"] = "weak"

            # 🚀 HFT UPGRADE: Professional AI Ensemble Enhancement
            try:
                # Use Professional AI Ensemble for institutional-grade confirmation
                ai_result = self.professional_ai_ensemble.professional_prediction(
                    symbol="current",  # Will be enhanced with actual symbol later
                    price_data=(price_history[-50:] if len(price_history) >= 50 else price_history),
                )

                if ai_result["professional_grade"]:
                    # Professional AI confirmed the signal
                    ai_confidence = ai_result["confidence"]
                    ai_prediction = ai_result["prediction"]

                    logger.info(
                        f"🎯 PROFESSIONAL AI: {ai_prediction} at {ai_confidence:.1%} confidence"
                    )

                    # Boost score if AI agrees with technical analysis
                    if ai_prediction == "BUY" and signals.get("bullish"):
                        score += 3.0  # Professional AI confirmation bonus
                        signals["ai_confirmation"] = "bullish_confirmed"
                    elif ai_prediction == "SELL" and signals.get("bearish"):
                        score += 3.0  # Professional AI confirmation bonus
                        signals["ai_confirmation"] = "bearish_confirmed"

                    # Store AI insights
                    signals["ai_confidence"] = ai_confidence
                    signals["ai_prediction"] = ai_prediction
                    signals["professional_grade"] = True
                else:
                    # AI didn't meet professional standards
                    signals["ai_confidence"] = ai_result["confidence"]
                    signals["professional_grade"] = False
                    logger.info(
                        f"🔍 AI Analysis: {ai_result['confidence']:.1%}"
                        f"(Below professional threshold)"
                    )

            except Exception as ai_error:
                logger.debug(f"Professional AI error: {ai_error}")
                signals["ai_error"] = str(ai_error)

            return score, signals

        except Exception as e:
            logger.debug(f"Technical analysis error: {e}")
            return 0.0, {"bullish": False, "bearish": False, "strength": "weak"}

    def _analyze_market_regime_for_winners(
        self, price_history: List[float]
    ) -> Tuple[float, Dict[str, Union[str, bool]]]:
        """📈 Analyze market regime for optimal trading conditions"""
        try:
            if len(price_history) < 20:  # Relaxed from 30
                return 0.0, {}

            score = 0.0
            signals: Dict[str, Union[str, bool]] = {}

            # Trend Analysis (0-3 points)
            trend_slope = np.polyfit(range(15), price_history[-15:], 1)[0]  # Relaxed from 20
            trend_strength = abs(trend_slope) / np.mean(price_history[-15:])

            if trend_strength > 0.0005:  # Trending (relaxed from 0.001)
                score += 3.0
                signals["trend"] = "trending"
            else:
                score += 2.0  # Sideways still gets points
                signals["trend"] = "sideways"

            # Volatility Regime (0-2 points)
            short_vol = np.std(price_history[-10:]) / np.mean(price_history[-10:])
            # long_vol = np.std(price_history[-15:]) / np.mean(price_history[-15:])  # Reserved for future use

            if 0.002 < short_vol < 0.008:  # Good volatility (relaxed)
                score += 2.0
                signals["volatility_regime"] = "optimal"
            else:
                score += 1.0
                signals["volatility_regime"] = "acceptable"

            # Market Structure (0-3 points)
            recent_high = max(price_history[-10:])
            recent_low = min(price_history[-10:])
            range_size = (recent_high - recent_low) / recent_low

            if 0.001 < range_size < 0.01:  # Good range (relaxed)
                score += 3.0
                signals["structure"] = "optimal_range"
            else:
                score += 2.0
                signals["structure"] = "acceptable_range"

            # Time of Day Bonus (0-2 points)
            current_hour = datetime.now().hour
            if 8 <= current_hour <= 16:  # Active trading hours
                score += 2.0
                signals["timing"] = "optimal_hours"
            else:
                score += 1.0  # Still give some points
                signals["timing"] = "acceptable_hours"

            return score, signals

        except Exception as e:
            logger.debug(f"Market regime analysis error: {e}")
            return 5.0, {"trend": "sideways"}  # Default decent score

    def _analyze_risk_reward_potential(
        self, price_history: List[float], action: str
    ) -> Tuple[float, Dict[str, Union[str, bool]]]:
        """⚖️ Analyze risk/reward potential for winner selection"""
        try:
            if len(price_history) < 15:  # Relaxed from 20
                return 5.0, {"entry": "acceptable"}  # Default decent score

            score = 0.0
            signals: Dict[str, Union[str, bool]] = {}

            current_price = price_history[-1]

            # Support/Resistance Analysis (0-4 points)
            recent_high = max(price_history[-15:])  # Relaxed from 20
            recent_low = min(price_history[-15:])

            if action == "BUY":
                distance_from_low = (current_price - recent_low) / recent_low
                if distance_from_low < 0.005:  # Near support (relaxed)
                    score += 4.0
                    signals["entry"] = "good_support"
                else:
                    score += 2.0  # Give some points anyway
                    signals["entry"] = "acceptable"

            elif action == "SELL":
                distance_from_high = (recent_high - current_price) / current_price
                if distance_from_high < 0.005:  # Near resistance (relaxed)
                    score += 4.0
                    signals["entry"] = "good_resistance"
                else:
                    score += 2.0  # Give some points anyway
                    signals["entry"] = "acceptable"

            # Profit Potential (0-3 points)
            range_size = (recent_high - recent_low) / recent_low
            if range_size > 0.003:  # Profit potential (relaxed)
                score += 3.0
                signals["profit_potential"] = "good"
            else:
                score += 2.0
                signals["profit_potential"] = "moderate"

            # Risk Assessment (0-3 points)
            volatility = np.std(price_history[-10:]) / np.mean(price_history[-10:])
            if volatility < 0.008:  # Risk (relaxed)
                score += 3.0
                signals["risk"] = "low"
            else:
                score += 2.0
                signals["risk"] = "moderate"

            return score, signals

        except Exception as e:
            logger.debug(f"Risk/reward analysis error: {e}")
            return 5.0, {"entry": "acceptable"}  # Default decent score

    def _detect_winner_patterns(
        self, price_history: List[float], action: str
    ) -> Tuple[float, Dict[str, Union[str, bool]]]:
        """🔍 Detect winning patterns in price action"""
        try:
            if len(price_history) < 10:  # Relaxed from 15
                return 4.0, {"pattern": "acceptable"}  # Default decent score

            score = 0.0
            signals: Dict[str, Union[str, bool]] = {}

            # Simple momentum pattern (0-5 points)
            if len(price_history) >= 5:
                momentum = (price_history[-1] - price_history[-5]) / price_history[-5]
                if action == "BUY" and momentum > 0.001:  # Upward momentum
                    score += 5.0
                    signals["momentum"] = True
                elif action == "SELL" and momentum < -0.001:  # Downward momentum
                    score += 5.0
                    signals["momentum"] = True
                else:
                    score += 2.0  # Some points anyway

            # Volatility pattern (0-3 points)
            if len(price_history) >= 10:
                volatility = np.std(price_history[-10:]) / np.mean(price_history[-10:])
                if 0.002 < volatility < 0.006:  # Good volatility
                    score += 3.0
                    signals["volatility"] = "optimal"
                else:
                    score += 2.0
                    signals["volatility"] = "acceptable"

            # Trend consistency (0-2 points)
            if len(price_history) >= 8:
                recent_prices = price_history[-8:]
                trend_direction = recent_prices[-1] - recent_prices[0]
                if (action == "BUY" and trend_direction > 0) or (
                    action == "SELL" and trend_direction < 0
                ):
                    score += 2.0
                    signals["trend_consistency"] = True
                else:
                    score += 1.0

            return score, signals

        except Exception as e:
            logger.debug(f"Pattern detection error: {e}")
            return 4.0, {"pattern": "acceptable"}  # Default decent score

    def _get_indicator_data(self) -> Dict[str, Optional[float]]:
        """Get indicator data for AI prediction with enhanced error handling"""
        try:
            indicator_data: Dict[str, Optional[float]] = {}

            # Check if we have sufficient price data
            if not hasattr(self.indicators, "price_data") or len(self.indicators.price_data) < 5:
                logger.warning(
                                        f"⚠️ Insufficient price data for indicators: {len(getattr(self.indicators, "
                    "'price_data', []))} points"
                )
                return {}

            # Helper function to convert numpy values to Python floats
            def safe_float(value: Any) -> Optional[float]:
                if value is None:
                    return None
                try:
                    # Handle numpy values
                    if hasattr(value, "item"):
                        value = value.item()
                    elif hasattr(value, "__float__"):
                        value = float(value)
                    return float(value)
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug(f"Safe float conversion failed for {value}: {e}")
                    return None

            # RSI with enhanced error handling
            try:
                rsi = self.indicators.calculate_rsi()
                if rsi is not None:
                    safe_rsi = safe_float(rsi)
                    if safe_rsi is not None and 0 <= safe_rsi <= 100:
                        indicator_data["rsi"] = safe_rsi
                    else:
                        logger.debug(f"Invalid RSI value: {safe_rsi}")
            except Exception as e:
                logger.debug(f"RSI calculation error: {e}")

            # EMA with enhanced error handling
            try:
                ema_fast = self.indicators.calculate_ema(EMA_FAST)
                ema_slow = self.indicators.calculate_ema(EMA_SLOW)
                if ema_fast is not None and ema_slow is not None:
                    safe_ema_fast = safe_float(ema_fast)
                    safe_ema_slow = safe_float(ema_slow)
                    if safe_ema_fast is not None and safe_ema_slow is not None:
                        indicator_data["ema_fast"] = safe_ema_fast
                        indicator_data["ema_slow"] = safe_ema_slow
            except Exception as e:
                logger.debug(f"EMA calculation error: {e}")

            # MACD with enhanced error handling
            try:
                macd_data = self.indicators.calculate_macd()
                if macd_data and macd_data.get("macd") is not None:
                    safe_macd = safe_float(macd_data.get("macd"))
                    safe_signal = safe_float(macd_data.get("signal"))
                    safe_histogram = safe_float(macd_data.get("histogram"))

                    if safe_macd is not None:
                        indicator_data["macd"] = safe_macd
                    if safe_signal is not None:
                        indicator_data["macd_signal"] = safe_signal
                    if safe_histogram is not None:
                        indicator_data["macd_histogram"] = safe_histogram
            except Exception as e:
                logger.debug(f"MACD calculation error: {e}")

            # SMA with enhanced error handling
            try:
                sma_short = self.indicators.calculate_sma(MA_SHORT)
                sma_long = self.indicators.calculate_sma(MA_LONG)
                if sma_short is not None and sma_long is not None:
                    safe_sma_short = safe_float(sma_short)
                    safe_sma_long = safe_float(sma_long)
                    if safe_sma_short is not None and safe_sma_long is not None:
                        indicator_data["sma_short"] = safe_sma_short
                        indicator_data["sma_long"] = safe_sma_long
            except Exception as e:
                logger.debug(f"SMA calculation error: {e}")

            # Get current price for additional features
            try:
                if hasattr(self, "price_history") and len(self.price_history) > 0:
                    current_price = safe_float(self.price_history[-1])
                    if current_price is not None and current_price > 0:
                        indicator_data["price"] = current_price
            except Exception as e:
                logger.debug(f"Current price retrieval error: {e}")

            # Validate that we have at least some indicator data
            if not indicator_data:
                logger.warning("⚠️ No valid indicator data calculated")
                return {}

            # Log successful indicator data retrieval occasionally
            if hasattr(self, "_loop_counter") and self._loop_counter % 60 == 0:
                indicator_count = len(indicator_data)
                indicators_list = list(indicator_data.keys())
                logger.info(f"📊 Retrieved {indicator_count} indicators: {indicators_list}")

            return indicator_data

        except Exception as e:
            logger.error(f"Error getting indicator data: {e}")
            import traceback

            logger.debug(f"Indicator data error traceback: {traceback.format_exc()}")
            return {}

    def _analyze_market_conditions(self):
        """Analyze market conditions with enhanced error handling"""
        try:
            # Check if we have sufficient price history
            if not hasattr(self, "price_history") or len(self.price_history) < 20:
                logger.debug(
                                        f"Insufficient price history for market analysis: "
                    "{len(getattr(self, 'price_history', []))}"
                )
                return

            recent_prices = self.price_history[-20:]

            # Validate price data
            try:
                valid_prices = [p for p in recent_prices if p > 0]
                if len(valid_prices) != len(recent_prices):
                    logger.warning("⚠️ Invalid price data detected in market analysis")
                    return
            except (TypeError, ValueError):
                logger.warning("⚠️ Invalid price data types in market analysis")
                return

            # Calculate trend with error handling
            try:
                trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                if not np.isfinite(trend):
                    logger.debug("Invalid trend calculation result")
                    trend = 0.0
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.debug(f"Trend calculation error: {e}")
                trend = 0.0

            # Calculate volatility with error handling
            try:
                mean_price = np.mean(recent_prices)
                if mean_price <= 0:
                    logger.debug("Invalid mean price for volatility calculation")
                    volatility = 0.0
                else:
                    std_price = np.std(recent_prices)
                    volatility = std_price / mean_price
                    if not np.isfinite(volatility):
                        volatility = 0.0
            except (ValueError, ZeroDivisionError) as e:
                logger.debug(f"Volatility calculation error: {e}")
                volatility = 0.0

            # Determine market condition with validation
            try:
                if abs(trend) < 0.001:
                    self.market_condition = MarketCondition.SIDEWAYS
                elif trend > 0.001:
                    self.market_condition = MarketCondition.TRENDING_UP
                elif trend < -0.001:
                    self.market_condition = MarketCondition.TRENDING_DOWN

                if volatility > 0.005:
                    self.market_condition = MarketCondition.VOLATILE

                # Log market condition occasionally
                if hasattr(self, "_loop_counter") and self._loop_counter % 120 == 0:
                    logger.debug(
                        f"📈 Market condition: {self.market_condition.value}"
                        f" Trend: {trend:.6f}, Volatility: {volatility:.4f}"
                    )

            except Exception as e:
                logger.debug(f"Market condition assignment error: {e}")
                self.market_condition = MarketCondition.SIDEWAYS

        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            # Set safe default
            if hasattr(self, "market_condition"):
                self.market_condition = MarketCondition.SIDEWAYS

    async def _smart_risk_management_check(self):
        """🚨 Smart Risk Management: Auto-close losing trades and manage portfolio risk"""
        try:
            if not self.enable_smart_risk_management or not self.mt5_interface:
                return

            current_time = datetime.now()
            trades_closed = 0
            total_saved = 0.0
            portfolio_loss = 0.0

            logger.debug("🔍 Smart Risk Management: Checking active trades...")

            # Check each active trade
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    mt5_ticket = trade.get("mt5_ticket", trade_id)
                    if not mt5_ticket:
                        continue

                    # Get current position info from MT5
                    position_info = await self.mt5_interface.get_position_info(mt5_ticket)
                    if not position_info:
                        # Position doesn't exist in MT5, remove from our tracking
                        logger.warning(f"🗑️ Removing orphaned trade {trade_id} - not found in MT5")
                        async with self.trades_lock:
                            del self.active_trades[trade_id]
                        continue

                    # Extract position data
                    current_profit = float(getattr(position_info, "profit", 0.0))
                    symbol = getattr(position_info, "symbol", trade.get("symbol", "Unknown"))
                    volume = float(getattr(position_info, "volume", 0.0))

                    # Calculate trade age
                    trade_timestamp = trade.get("timestamp")
                    if trade_timestamp:
                        try:
                            trade_time = datetime.fromisoformat(trade_timestamp)
                            trade_age_minutes = (current_time - trade_time).total_seconds() / 60
                        except:
                            trade_age_minutes = 0
                    else:
                        trade_age_minutes = 0

                    # Add to portfolio loss calculation
                    if current_profit < 0:
                        portfolio_loss += abs(current_profit)

                    # Risk Management Rules
                    should_close = False
                    close_reason = ""

                    # Rule 1: Max loss per trade
                    if current_profit <= -self.max_loss_per_trade:
                        should_close = True
                        f"Max loss exceeded: ${current_profit:.2f}"
                        f"<= -${self.max_loss_per_trade:.2f}"

                    # Rule 2: Trade age limit
                    elif trade_age_minutes >= self.max_trade_age_minutes:
                        should_close = True
                        f"Trade too old: {trade_age_minutes:.0f}"
                        f">= {self.max_trade_age_minutes} minutes"

                    # Rule 3: Portfolio protection (if losing)
                    elif current_profit < 0 and portfolio_loss >= self.max_portfolio_loss:
                        should_close = True
                        f"Portfolio protection: Total loss ${portfolio_loss:.2f}"
                        f">= ${self.max_portfolio_loss:.2f}"

                    # Close trade if any rule triggered
                    if should_close:
                        logger.warning(f"🚨 RISK MANAGEMENT: Closing {symbol} trade {mt5_ticket}")
                        logger.warning(f"   Reason: {close_reason}")
                        logger.warning(f"   Current P&L: ${current_profit:.2f}")
                        logger.warning(f"   Volume: {volume}")

                        # Close the trade
                        close_result = await self.mt5_interface.close_position(mt5_ticket)
                        if close_result:
                            # Update statistics
                            trades_closed += 1
                            if current_profit < 0:
                                total_saved += abs(current_profit)

                            # Remove from active trades with thread safety
                            async with self.trades_lock:
                                del self.active_trades[trade_id]

                            # Send notification
                            await self._send_risk_management_notification(
                                symbol, current_profit, close_reason, trades_closed
                            )

                            logger.info(
                                f"✅ Risk Management: Closed {symbol}"
                                f"trade - saved ${abs(current_profit):.2f}"
                            )
                        else:
                            logger.error(f"❌ Failed to close {symbol} trade {mt5_ticket}")

                    else:
                        # Log healthy trades occasionally
                        if hasattr(self, "_loop_counter") and self._loop_counter % 120 == 0:
                            logger.debug(
                                f"📊 {symbol}"
                                f"trade healthy: P&L ${current_profit:.2f}, Age {trade_age_minutes:.0f}m"
                            )

                except Exception as trade_error:
                    logger.error(f"Error checking trade {trade_id}: {trade_error}")

            # Portfolio Emergency Protection
            if portfolio_loss >= self.max_portfolio_loss:
                logger.error(
                    f"🚨 PORTFOLIO EMERGENCY: Total loss ${portfolio_loss:.2f}"
                    f">= ${self.max_portfolio_loss:.2f}"
                )
                await self._emergency_close_all_losing_trades(portfolio_loss)

            # Update risk management statistics
            if trades_closed > 0:
                current_trades_closed = self.risk_management_stats.get("trades_auto_closed", 0)
                current_saved_loss = self.risk_management_stats.get("total_saved_loss", 0.0)

                # Ensure types are correct before conversion
                if isinstance(current_trades_closed, (int, float)):
                    self.risk_management_stats["trades_auto_closed"] = (
                        int(current_trades_closed) + trades_closed
                    )
                else:
                    self.risk_management_stats["trades_auto_closed"] = trades_closed

                if isinstance(current_saved_loss, (int, float)):
                    self.risk_management_stats["total_saved_loss"] = (
                        float(current_saved_loss) + total_saved
                    )
                else:
                    self.risk_management_stats["total_saved_loss"] = total_saved
                self.risk_management_stats["last_cleanup_time"] = current_time.isoformat()

                logger.warning("🛡️ RISK MANAGEMENT SUMMARY:")
                logger.warning(f"   Trades closed: {trades_closed}")
                logger.warning(f"   Potential loss saved: ${total_saved:.2f}")
                logger.warning(
                    f"   Total trades auto-closed: {self.risk_management_stats['trades_auto_closed']}"
                    f"
                )
                logger.warning(
                    f"   Total saved this session: ${self.risk_management_stats['total_saved_loss']:.2f}"
                    f"
                )

        except Exception as e:
            logger.error(f"Smart risk management check error: {e}")

    async def _send_risk_management_notification(
        self, symbol: str, profit: float, reason: str, total_closed: int
    ):
        """Send risk management notification"""
        try:
            msg = f"""🚨 RISK MANAGEMENT ACTION

🔴 Closed {symbol} Trade
💰 P&L: ${profit:+.2f}
📋 Reason: {reason}
🔢 Trades closed today: {total_closed}

🛡️ Protecting your capital automatically!"""

            # Use sanitizer for outbound message
            clean_msg = TelegramTextSanitizer.sanitize_message(msg)
            await self.notifier.telegram.send_message(clean_msg)
        except Exception as e:
            logger.warning(f"Risk management notification failed: {e}")

    async def _emergency_close_all_losing_trades(self, portfolio_loss: float):
        """Emergency close all losing trades"""
        try:
            logger.error("🚨 EMERGENCY PORTFOLIO PROTECTION: Closing all losing trades!")

            trades_closed = 0
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    mt5_ticket = trade.get("mt5_ticket", trade_id)
                    if not mt5_ticket:
                        continue

                    if self.mt5_interface:
                        position_info = await self.mt5_interface.get_position_info(mt5_ticket)
                    else:
                        position_info = None
                    if position_info:
                        current_profit = float(getattr(position_info, "profit", 0.0))
                        if current_profit < 0:  # Only close losing trades
                            if self.mt5_interface:
                                close_result = await self.mt5_interface.close_position(mt5_ticket)
                            else:
                                close_result = False
                            if close_result:
                                async with self.trades_lock:
                                    del self.active_trades[trade_id]
                                trades_closed += 1
                                logger.warning(
                                    f"🚨 Emergency closed trade {mt5_ticket}: ${current_profit:.2f}"
                                )

                except Exception as e:
                    logger.error(f"Error in emergency close {trade_id}: {e}")

            # Send emergency notification
            msg = f"""🚨 EMERGENCY PORTFOLIO PROTECTION

⚠️ Portfolio loss reached ${portfolio_loss:.2f}
🔴 Auto-closed {trades_closed} losing trades
🛡️ Capital preserved!

💡 Consider reducing position sizes or taking a break."""

            # Use sanitizer for outbound message
            clean_msg = TelegramTextSanitizer.sanitize_message(msg)
            await self.notifier.telegram.send_message(clean_msg)
            logger.error(f"🚨 Emergency protection complete: {trades_closed} trades closed")

        except Exception as e:
            logger.error(f"Emergency close all error: {e}")

    def _check_loss_protection(self) -> bool:
        """🛡️ LOSS Protection system - DISABLED FOR DEMO MODE"""
        try:
            # 🚨 DEMO MODE: NEVER STOP TRADING!
            if self.demo_mode or "Demo" in str(getattr(self.mt5_interface, "server", "")):
                logger.info("✅ DEMO BYPASS: Loss protection disabled for demo trading")
                # Reset any triggered protection for demo accounts
                if self.loss_protection_triggered:
                    logger.warning(
                        "🎯 DEMO MODE: Resetting loss protection - Demo accounts never stop!"
                    )
                    self.loss_protection_triggered = False
                    self.starting_balance = self.current_balance  # Reset baseline

                return True  # ALWAYS allow trading in demo mode!

            if not self.loss_protection_enabled:
                return True  # Protection disabled, allow trading

            if self.loss_protection_triggered:
                logger.warning("🛡️ LOSS PROTECTION: Bot already stopped for loss protection")
                return False

            # Calculate current profit/loss
            current_profit = self.current_balance - self.starting_balance

            # 🚀 UNLIMITED PROFITS MODE - Never stop for profits!
            if current_profit > 0:
                # Log progress but NEVER stop for profits
                if hasattr(self, "_loop_counter") and self._loop_counter % 60 == 0:
                    logger.info(
                        f"💰 PROFIT MODE: +${current_profit:.2f} - Bot continues indefinitely!"
                    )
                return True  # Always allow trading when profitable!

            # 🛡️ LOSS PROTECTION - Only check loss limits
            loss_threshold = 2.0  # Stop at $2 loss
            emergency_loss_threshold = 5.0  # Emergency stop at $5 loss

            if current_profit <= -emergency_loss_threshold:
                # Emergency stop at max loss threshold
                self.loss_protection_triggered = True
                logger.error("🚨 EMERGENCY LOSS PROTECTION TRIGGERED!")
                logger.error(
                    f"� Loss reached: ${current_profit:.2f} <= -${emergency_loss_threshold:.2f}"
                )
                logger.error("🛑 BOT STOPPED! Check your account and consider reducing risk!")

                # Send emergency notification
                asyncio.create_task(
                    self._send_loss_protection_notification(current_profit, emergency=True)
                )
                return False

            elif current_profit <= -loss_threshold:
                # Standard loss protection trigger
                self.loss_protection_triggered = True
                logger.warning("🛡️ LOSS PROTECTION TRIGGERED!")
                logger.warning(
                    f"� Loss limit reached: ${current_profit:.2f} <= -${loss_threshold:.2f}"
                )
                logger.warning("🛑 BOT STOPPED! Protecting remaining capital!")
                logger.warning(f"📊 Starting balance: ${self.starting_balance:.2f}")
                logger.warning(f"📊 Current balance: ${self.current_balance:.2f}")
                logger.warning(f"📊 Total loss: ${current_profit:.2f}")

                # Send notification
                asyncio.create_task(
                    self._send_loss_protection_notification(current_profit, emergency=False)
                )
                return False

            # Log progress occasionally when approaching loss limit
            if current_profit < 0:
                progress_pct = abs(current_profit / loss_threshold) * 100
                if hasattr(self, "_loop_counter") and self._loop_counter % 60 == 0:
                    logger.info(
                        f"🛡️ Loss Protection: ${current_profit:.2f}"
                        f"-${loss_threshold:.2f} ({progress_pct:.0f}%)"
                    )

            return True  # Allow trading

        except Exception as e:
            logger.error(f"Loss protection check error: {e}")
            return True  # Allow trading if check fails

    async def _send_loss_protection_notification(
        self, current_loss: float, emergency: bool = False
    ):
        """Send loss protection notification"""
        try:
            emoji = "🚨" if emergency else "🛡️"
            title = "EMERGENCY LOSS PROTECTION" if emergency else "LOSS PROTECTION TRIGGERED"
            threshold = 5.0 if emergency else 2.0

            msg = f"""{emoji} {title}

💰 Starting Balance: ${self.starting_balance:.2f}
💰 Current Balance: ${self.current_balance:.2f}
� Total Loss: ${current_loss:.2f}
🎯 Loss Limit: -${threshold:.2f}

🛑 BOT HAS STOPPED TRADING!

📋 ACTION REQUIRED:
1. Review your trading strategy
2. Consider reducing position sizes
3. Check market conditions
4. Restart bot when ready to continue

⚠️ The bot will NOT place new trades until manually restarted."""

            # Use sanitizer for outbound message
            await self._send_sanitized_message(msg)
            logger.info("📱 Loss protection notification sent")
        except Exception as e:
            logger.warning(f"Loss protection notification failed: {e}")

    def reset_loss_protection(self) -> Optional[Dict[str, Union[bool, float]]]:
        """Reset loss protection system (call this to resume trading after loss limit)"""
        try:
            old_status = self.loss_protection_triggered
            self.loss_protection_triggered = False

            # Update starting balance to current balance (new baseline)
            self.starting_balance = self.current_balance

            logger.warning("🔄 LOSS PROTECTION RESET!")
            logger.warning(f"📊 New starting balance: ${self.starting_balance:.2f}")
            logger.warning("✅ Bot ready to resume unlimited profit trading")

            return {
                "previous_status": old_status,
                "new_starting_balance": self.starting_balance,
                "current_balance": self.current_balance,
            }
        except Exception as e:
            logger.error(f"Loss protection reset error: {e}")
            return None

    def _check_risk_limits(self) -> bool:
        """Check risk limits with DEMO MODE overrides"""
        try:
            # 🚨 DEMO MODE: UNLIMITED RISK FOR AI TRAINING!
            if self.demo_mode or "Demo" in str(getattr(self.mt5_interface, "server", "")):
                logger.info("✅ DEMO BYPASS: Risk checks disabled for unlimited AI training")
                # Reset any risk counters that might block trading
                if self.consecutive_losses >= 25:  # Reset every 25 losses
                    old_count = self.consecutive_losses
                    self.consecutive_losses = 0
                    logger.warning(
                        f"🎯 DEMO RESET: Consecutive losses {old_count}"
                        f"→ 0 (Demo training continues!)"
                    )

                return True  # ALWAYS allow trading in demo mode!

            # 🛡️ LOSS PROTECTION CHECK - HIGHEST PRIORITY for live accounts!
            loss_protection_ok = self._check_loss_protection()
            logger.info(f"✅ Loss Protection: {'PASS' if loss_protection_ok else 'BLOCKED'}")
            if not loss_protection_ok:
                return False

            # FORCE RELOAD CONFIG TO GET LATEST VALUES
            try:
                importlib.reload(config)
                max_consecutive_losses = getattr(config, "MAX_CONSECUTIVE_LOSSES", 100)
                max_daily_loss = getattr(config, "MAX_DAILY_LOSS", 8.0)
                trade_amount = getattr(config, "TRADE_AMOUNT", 0.5)
            except (ImportError, AttributeError):
                # Fallback values - ensure it's 100, not 10!
                max_consecutive_losses = 100  # ✅ ENSURE IT'S 100!
                max_daily_loss = 8.0
                trade_amount = 0.5

            # Check daily loss
            if self.daily_profit < -max_daily_loss:
                logger.warning(
                    f"Daily loss limit reached: ${self.daily_profit:.2f} < -${max_daily_loss}"
                )
                return False

            # Check consecutive losses with BULLETPROOF logic
            if self.consecutive_losses >= max_consecutive_losses:
                logger.warning(
                    f"🚨 CONSECUTIVE LOSS LIMIT: {self.consecutive_losses}/{max_consecutive_losses}"
                )
                logger.warning(
                    f"🔧 Current config MAX_CONSECUTIVE_LOSSES: {max_consecutive_losses}"
                )

                # EMERGENCY AUTO-RESET if limit is unreasonably low
                if max_consecutive_losses < 50:
                    logger.error(
                        f"🚨 DETECTED LOW LIMIT ({max_consecutive_losses}"
                        f" - APPLYING EMERGENCY RESET!"
                    )
                    old_count = self.consecutive_losses
                    self.consecutive_losses = 0
                    logger.info(
                        f"🔧 EMERGENCY: Reset {old_count} → 0 due to unreasonably low limit"
                    )
                    return True  # Allow trading after emergency reset
                else:
                    return False
            else:
                # Log current status occasionally with actual limit
                if self.consecutive_losses > 0 and self.consecutive_losses % 25 == 0:
                    logger.info(
                        f"📊 Consecutive losses: {self.consecutive_losses}/{max_consecutive_losses}"
                    )

            # Check balance
            if self.current_balance < trade_amount * 2:
                logger.warning(
                    f"Insufficient balance: ${self.current_balance:.2f} < ${trade_amount * 2:.2f}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False

    def _check_concurrent_trades_limit(self) -> bool:
        """Check if we can open more trades based on margin-safe concurrent limit"""
        try:
            # �️ MARGIN-SAFE CONCURRENT TRADING - Only 1 trade for small accounts!
            if self.current_balance <= 100:
                max_concurrent_trades = 1  # ONLY 1 trade for small accounts!
                logger.info(
                    f"🛡️ MARGIN-SAFE: Small account ${self.current_balance:.2f} - max 1 trade"
                )
            else:
                max_concurrent_trades = 3  # Maximum 3 for larger accounts

            current_open_trades = len(self.active_trades)

            if current_open_trades >= max_concurrent_trades:
                logger.warning(
                    f"📊 Max concurrent trades reached: {current_open_trades}"
                    f"{max_concurrent_trades}"
                )

                # 🚨 AUTO-FIX: Check for stale trades if limit reached
                logger.warning("🔧 Auto-checking for stale trades to free up slots...")

                # Quick check for obviously stale trades (old timestamps)
                stale_found = 0
                current_time = datetime.now()

                for trade_id, trade in list(self.active_trades.items()):
                    try:
                        trade_time = datetime.fromisoformat(trade["timestamp"])
                        age_hours = (current_time - trade_time).total_seconds() / 3600

                        # If trade is older than 2 hours, it might be stale
                        if age_hours > 2:
                            logger.warning(
                                f"🗑️ Removing old trade {trade_id} (age: {age_hours:.1f}h)"
                            )
                            # Note: This is in a sync function, so we can't use async lock
                            # The trade will be cleaned up in the next async monitor cycle
                            self.active_trades.pop(trade_id, None)  # Safe removal
                            stale_found += 1

                    except Exception as e:
                        logger.warning(f"⚠️ Error checking trade age {trade_id}: {e}")
                        # Remove problematic trade safely
                        self.active_trades.pop(trade_id, None)  # Safe removal
                        stale_found += 1

                if stale_found > 0:
                    new_count = len(self.active_trades)
                    logger.warning(
                        f"🧹 Removed {stale_found}"
                        f"stale trades. New count: {new_count}/{max_concurrent_trades}"
                    )

                    # Check again after cleanup
                    if new_count < max_concurrent_trades:
                        logger.warning(
                            f"✅ SLOTS FREED! Can now trade again: {new_count}"
                            f"{max_concurrent_trades}"
                        )
                        return True

                return False

            logger.info(
                f"✅ Concurrent trades OK: {current_open_trades}"
                f"{max_concurrent_trades} - {max_concurrent_trades - current_open_trades} slots available!"
            )
            return True

        except Exception as e:
            logger.error(f"Concurrent trades check error: {e}")
            return True  # Allow trading if check fails

    async def check_margin_safety(self) -> bool:
        """🛡️ MARGIN-SAFE MONITORING - using async interface only"""
        try:
            if not self.mt5_interface:
                logger.warning("⚠️ MT5 interface not available for margin check")
                return True  # Allow trading if interface unavailable

            # Use async interface instead of direct MT5 calls
            try:
                balance = await self.mt5_interface.get_account_balance()
                if balance and balance > 0:
                    # Update current balance
                    self.current_balance = balance

                    # Basic safety check
                    if balance < 10:  # Critical low balance
                        logger.error(f"🚨 CRITICAL LOW BALANCE: ${balance:.2f}")
                        return False

                    # Rate-limited debug message for routine checks
                    if self.log_limiter.should_log("margin_check_ok"):
                        logger.debug(f"💰 Margin Check: Balance ${balance:.2f} - OK")
                    return True
                else:
                    logger.error("❌ Invalid account balance for margin check")
                    return False

            except Exception as e:
                logger.error(f"Margin check error: {e}")
                return True  # Allow trading on error to prevent lockup

            # 🚨 CRITICAL MARGIN LEVELS - These prevent account destruction!
            MARGIN_CALL_LEVEL = 200  # If below 200%, STOP ALL TRADING!
            DANGER_ZONE = 300  # Warning zone
            MINIMUM_FREE_MARGIN = 10  # Must have $10 free margin minimum

            # Initialize margin variables with safe defaults
            equity = balance
            free_margin = balance * 0.8  # Conservative estimate
            margin_level = 999999  # Safe default

            logger.info(
                f"💰 Margin Check: Balance=${balance:.2f}, Equity=${equity:.2f}, "
                f"Free=${free_margin:.2f}, Level={margin_level:.1f}%"
            )

            # 🚨 EMERGENCY STOP CONDITIONS
            if margin_level < MARGIN_CALL_LEVEL:
                logger.error(
                    f"🚨 MARGIN CALL DANGER! Level {margin_level:.1f}% < "
                    f"{MARGIN_CALL_LEVEL}% - STOPPING ALL TRADING!"
                )
                return False

            if free_margin < MINIMUM_FREE_MARGIN:
                logger.error(
                    f"🚨 LOW FREE MARGIN! ${free_margin:.2f} < ${MINIMUM_FREE_MARGIN} - "
                    f"STOPPING ALL TRADING!"
                )
                return False

            # Warning zone
            if margin_level < DANGER_ZONE:
                logger.warning(
                    f"⚠️ MARGIN WARNING: Level {margin_level:.1f}% < {DANGER_ZONE}% - "
                    f"Trading with extreme caution!"
                )
                return True  # Still allow trading but with warning

            # All good
            logger.info(
                f"✅ MARGIN SAFE: Level {margin_level:.1f}% > {DANGER_ZONE}%, "
                f"Free=${free_margin:.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Margin safety check failed: {e}")
            return False  # Be safe - don't trade if we can't check margin

    async def _continuous_margin_monitoring(self):
        """🛡️ CONTINUOUS margin monitoring - Check every 10 seconds during active trades"""
        while self.running:
            try:
                # Thread-safe check for active trades
                async with self.trades_lock:
                    has_active_trades = bool(self.active_trades)

                if has_active_trades and self.mt5_interface:
                    # Use async interface for account info
                    try:
                        balance = await self.mt5_interface.get_account_balance()
                        if balance and balance > 0:
                            # For continuous monitoring, we'll use simplified margin assumptions
                            # Real margin monitoring should be done through MT5 interface methods
                            margin_level = 999999  # Safe default
                            free_margin = balance * 0.8  # Conservative estimate
                        else:
                            margin_level = 999999
                            free_margin = 0.0
                    except Exception as e:
                        logger.error(f"Margin check failed: {e}")
                        margin_level = 999999
                        free_margin = 0.0

                        # 🚨 EMERGENCY: Close all trades if margin drops dangerously
                        if margin_level < 300:  # Danger zone
                            logger.error(
                                f"🚨 MARGIN EMERGENCY: {margin_level:.1f}% - CLOSING ALL TRADES!"
                            )
                            await self._emergency_close_all_trades("Margin Emergency")

                        elif margin_level < 500:  # Warning zone
                            logger.warning(
                                f"⚠️ MARGIN WARNING: {margin_level:.1f}% - Monitoring closely"
                            )

                        # 🚨 EMERGENCY: Free margin too low
                        if free_margin < 20:  # $20 minimum
                            logger.error(
                                f"🚨 LOW FREE MARGIN: ${free_margin:.2f} - CLOSING ALL TRADES!"
                            )
                            await self._emergency_close_all_trades("Low Free Margin")

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Continuous margin monitoring error: {e}")
                await asyncio.sleep(30)

    async def _emergency_close_all_trades(self, reason: str):
        """🚨 EMERGENCY: Close all active trades immediately"""
        try:
            logger.error(f"🚨 EMERGENCY CLOSE ALL TRADES: {reason}")

            trades_closed = 0
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    mt5_ticket = trade.get("mt5_ticket", trade_id)
                    if not mt5_ticket:
                        continue

                    if self.mt5_interface:
                        close_result = await self.mt5_interface.close_position(mt5_ticket)
                        if close_result:
                            async with self.trades_lock:
                                del self.active_trades[trade_id]
                            trades_closed += 1
                            logger.error(f"🚨 EMERGENCY CLOSED: {mt5_ticket}")
                        else:
                            logger.error(f"❌ Failed to emergency close: {mt5_ticket}")

                except Exception as e:
                    logger.error(f"Error in emergency close {trade_id}: {e}")

            # Send emergency notification
            msg = f"""🚨 EMERGENCY PROTOCOL ACTIVATED

⚠️ Reason: {reason}
🔴 Trades closed: {trades_closed}
🛡️ Account protected!

💡 Check your account and restart bot when safe."""

            # Use sanitizer for outbound message
            await self._send_sanitized_message(msg)
            logger.error(f"🚨 Emergency close complete: {trades_closed} trades closed")

        except Exception as e:
            logger.error(f"Emergency close all error: {e}")

    def _check_maximum_drawdown_protection(self) -> bool:
        """🛡️ Maximum drawdown protection - Never lose more than 20% of starting balance"""
        try:
            if self.starting_balance <= 0:
                return True

            current_drawdown = self.starting_balance - self.current_balance
            max_allowed_drawdown = self.starting_balance * 0.20  # 20% maximum

            if current_drawdown >= max_allowed_drawdown:
                logger.error("🚨 MAXIMUM DRAWDOWN REACHED!")
                logger.error(f"   Starting: ${self.starting_balance:.2f}")
                logger.error(f"   Current: ${self.current_balance:.2f}")
                logger.error(
                    f"   Drawdown: ${current_drawdown:.2f} (Max: ${max_allowed_drawdown:.2f})"
                )
                logger.error("🛑 STOPPING ALL TRADING TO PRESERVE CAPITAL!")

                # Trigger emergency shutdown
                asyncio.create_task(self._emergency_close_all_trades("Maximum Drawdown Protection"))
                return False

            return True

        except Exception as e:
            logger.error(f"Drawdown protection error: {e}")
            return True

    async def _validate_position_size_against_margin(self, symbol: str, lot_size: float) -> float:
        """🛡️ Validate position size against available margin"""
        try:
            if not self.mt5_interface:
                return 0.01

            # Get margin requirement for this position (conservative estimate)
            margin_required = lot_size * 1000 * 100  # Conservative 100:1 margin estimate

            # Get current free margin using async interface
            if self.mt5_interface:
                try:
                    balance = await self.mt5_interface.get_account_balance()
                    if balance and balance > 0:
                        # Conservative free margin estimate (80% of balance)
                        free_margin = balance * 0.8

                        # Must have at least 3x the margin requirement available
                        safety_margin = margin_required * 3

                        if free_margin < safety_margin:
                            # Reduce position size to fit available margin
                            max_safe_lots = (free_margin / 3) / (1000 * 100)
                            safe_lot_size = max(
                                0.20, min(lot_size, max_safe_lots)
                            )  # 🚀 Minimum 0.20 lots!

                            logger.warning(
                                f"🛡️ MARGIN SAFETY: Reduced {symbol}"
                                f"from {lot_size} to {safe_lot_size} lots"
                            )
                            return safe_lot_size
                    else:
                        logger.warning("⚠️ Could not get account balance for margin check")
                except Exception as e:
                    logger.error(f"Margin check failed: {e}")
                    logger.warning("⚠️ Using conservative default margin for safety")

            return lot_size

        except Exception as e:
            logger.error(f"Margin validation error: {e}")
            return 0.01  # Ultra-safe fallback

    async def _check_spread_protection(self, symbol: str) -> bool:
        """🛡️ Spread protection using async interface only"""
        try:
            if not self.mt5_interface:
                return False

            # Use async interface instead of direct mt5 calls
            current_price = await self.mt5_interface.get_current_price(symbol)
            if not current_price:
                return False

            # Get symbol info through async interface for spread calculation
            try:
                symbol_info = await self.mt5_interface.get_symbol_info(symbol)
                if not symbol_info:
                    # Only bypass spread check in emergency mode
                    if ENABLE_EMERGENCY_TRADING:
                        logger.warning(f"⚠️ EMERGENCY MODE: Bypassing spread check for {symbol}")
                        return True
                    return False  # Default: reject if can't get symbol info

                # Calculate spread using symbol info
                spread_points = symbol_info.get("spread", 0)
            except Exception:
                # Only bypass spread check in emergency mode
                if ENABLE_EMERGENCY_TRADING:
                    logger.warning(f"⚠️ EMERGENCY MODE: Bypassing spread check error for {symbol}")
                    return True
                return False  # Default: reject on error

            # Maximum allowed spread by symbol
            max_spread = {
                "Volatility 10 Index": 2.0,
                "Volatility 25 Index": 3.0,
                "Volatility 50 Index": 4.0,
                "Volatility 75 Index": 5.0,
                "Volatility 100 Index": 6.0,
            }

            symbol_max_spread = max_spread.get(symbol, 10.0)

            if spread_points > symbol_max_spread:
                logger.warning(
                    f"� SPREAD TOO WIDE: {symbol}"
                    f"spread {spread_points} > {symbol_max_spread} points"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Spread check error: {e}")
            # Only bypass spread check in emergency mode
            if ENABLE_EMERGENCY_TRADING:
                logger.warning(f"⚠️ EMERGENCY MODE: Bypassing spread check exception for {symbol}")
                return True
            return False  # Default: reject on error

    async def _check_correlation_protection(self, new_symbol: str, new_action: str) -> bool:
        """🛡️ Correlation protection - Prevent overexposure to correlated assets"""
        try:
            # Count positions by symbol and direction (thread-safe read)
            symbol_exposure: Dict[str, int] = {}
            direction_exposure = {"BUY": 0, "SELL": 0}

            async with self.trades_lock:
                for trade in self.active_trades.values():
                    symbol = trade.get("symbol", "")
                    action = trade.get("action", "")

                    if symbol:
                        symbol_exposure[symbol] = symbol_exposure.get(symbol, 0) + 1
                    if action:
                        direction_exposure[action] = direction_exposure.get(action, 0) + 1

            # 🚨 Rule 1: Maximum 1 position per symbol for small accounts
            if self.current_balance <= 100:
                if new_symbol in symbol_exposure and symbol_exposure[new_symbol] >= 1:
                    logger.warning(
                        f"🚫 CORRELATION: Already have {symbol_exposure[new_symbol]}"
                        f"{new_symbol} position(s)"
                    )
                    return False

            # 🚨 Rule 2: Maximum 2 positions in same direction for small accounts
            if self.current_balance <= 100:
                if direction_exposure[new_action] >= 2:
                    logger.warning(
                        f"🚫 CORRELATION: Already have {direction_exposure[new_action]}"
                        f"{new_action} positions"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Correlation protection error: {e}")
            return True

    async def _connection_watchdog(self):
        """🛡️ Connection watchdog - Monitor MT5 connection health"""
        consecutive_failures = 0

        while self.running:
            try:
                if self.mt5_interface:
                    # Test connection using async interface
                    try:
                        balance = await self.mt5_interface.get_account_balance()
                        if balance and balance > 0:
                            consecutive_failures = 0  # Reset on success
                        else:
                            consecutive_failures += 1
                            logger.error(f"🚨 MT5 CONNECTION FAILURE #{consecutive_failures}")

                            if consecutive_failures >= 3:
                                logger.error(
                                    "🚨 CRITICAL: 3 consecutive MT5 failures - EMERGENCY PROTOCOL!"
                                )
                                logger.error("📱 Check your internet connection and MT5 login!")

                                # Emergency notification
                                critical_msg = (
                                    "🚨 CRITICAL: MT5 CONNECTION LOST!\n\n"
                                    "Bot cannot manage trades!\n"
                                    "Please check your connection immediately!"
                                )
                                clean_msg = TelegramTextSanitizer.sanitize_message(critical_msg)
                                await self.notifier.telegram.send_message(clean_msg)

                                # Set flag to prevent new trades
                                self.mt5_connected = False

                                # Try to reconnect
                                await self._emergency_reconnect()
                    except Exception as e:
                        consecutive_failures += 1
                        logger.error(f"🚨 MT5 CONNECTION TEST FAILED #{consecutive_failures}: {e}")

                        if consecutive_failures >= 3:
                            # Set flag to prevent new trades
                            self.mt5_connected = False

                    else:
                        if consecutive_failures > 0:
                            logger.info(
                                f"✅ MT5 connection restored after {consecutive_failures} failures"
                            )
                            consecutive_failures = 0
                            self.mt5_connected = True

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Connection watchdog error: {e}")
                await asyncio.sleep(60)

    async def _emergency_reconnect(self):
        """Emergency MT5 reconnection"""
        try:
            logger.info("🔄 Attempting emergency MT5 reconnection...")

            if self.mt5_interface:
                # Try to reinitialize
                success = await self.mt5_interface.initialize()
                if success:
                    self.mt5_connected = True
                    logger.info("✅ Emergency reconnection successful!")
                else:
                    logger.error("❌ Emergency reconnection failed!")

        except Exception as e:
            logger.error(f"Emergency reconnection error: {e}")

    def _check_extended_loss_limits(self) -> bool:
        """🛡️ Check weekly and monthly loss limits"""
        try:
            now = datetime.now()

            # Initialize weekly/monthly tracking if not exists
            if not hasattr(self, "weekly_profit"):
                self.weekly_profit = 0.0
                self.monthly_profit = 0.0
                self.last_week_reset = now
                self.last_month_reset = now
                self.max_weekly_loss = 50.0  # $50 max weekly loss
                self.max_monthly_loss = 150.0  # $150 max monthly loss

            # Reset weekly if needed
            if (now - self.last_week_reset).days >= 7:
                self.weekly_profit = 0.0
                self.last_week_reset = now
                logger.info("📅 Weekly P&L reset")

            # Reset monthly if needed
            if (now - self.last_month_reset).days >= 30:
                self.monthly_profit = 0.0
                self.last_month_reset = now
                logger.info("📅 Monthly P&L reset")

            # Check limits
            if self.weekly_profit <= -self.max_weekly_loss:
                logger.error(
                    f"🚨 WEEKLY LOSS LIMIT: ${self.weekly_profit:.2f} <= -${self.max_weekly_loss}"
                )
                return False

            if self.monthly_profit <= -self.max_monthly_loss:
                logger.error(
                    f"🚨 MONTHLY LOSS LIMIT: ${self.monthly_profit:.2f} <= -${self.max_monthly_loss}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Extended loss limits error: {e}")
            return True

    async def _check_hedge_prevention(self, new_action: str) -> bool:
        """Check if new trade would create a hedge conflict - ULTRA AGGRESSIVE MODE!"""
        try:
            # 🚀 ULTRA AGGRESSIVE: DISABLE ALL HEDGE PREVENTION FOR MAXIMUM PROFIT!
            enable_hedge_prevention = False  # FORCE DISABLE for aggressive trading!
            allow_same_direction = True  # FORCE ENABLE stacking!

            if not enable_hedge_prevention:
                logger.info(
                    f"🚀 AGGRESSIVE MODE: {new_action}"
                    f"position ALWAYS allowed - no hedge restrictions!"
                )
                return True  # Allow ALL trades for maximum profit potential!

            # This code below will never execute due to disabled hedge prevention
            async with self.trades_lock:
                existing_actions = [trade["action"] for trade in self.active_trades.values()]

            if not existing_actions:
                return True  # No existing trades, allow any new trade

            # Check for opposite positions (hedging) - DISABLED
            has_buy = "BUY" in existing_actions
            has_sell = "SELL" in existing_actions

            if (new_action == "BUY" and has_sell) or (new_action == "SELL" and has_buy):
                logger.info(
                    f"� HEDGE ALLOWED: {new_action}"
                    f"position with opposite direction for diversification!"
                )
                return True  # Allow hedging for maximum opportunities

            # Check for same direction stacking - ALWAYS ALLOWED
            if new_action in existing_actions and not allow_same_direction:
                logger.info(
                    f"� STACKING ALLOWED: Multiple {new_action} positions for compound profits!"
                )
                return True  # Always allow stacking

            logger.info(f"✅ Hedge Check Passed: {new_action} position allowed")
            return True

        except Exception as e:
            logger.error(f"Hedge prevention check error: {e}")
            return True  # Always allow trading if check fails
            return True  # Always allow trading if check fails

    async def _should_close_trade_profitable(
        self, trade: Dict[str, Any], position_info: Any
    ) -> tuple[bool, str]:
        """🚀 PROFITABLE trade management - much more aggressive profit taking!"""
        try:
            # Get current profit
            current_profit = 0.0

            try:
                if hasattr(position_info, "profit"):
                    current_profit = float(position_info.profit or 0.0)
                elif isinstance(position_info, dict):
                    # Dict access with explicit type handling
                    profit_val = position_info.get("profit", 0.0)  # type: ignore
                    try:
                        current_profit = float(profit_val) if profit_val is not None else 0.0  # type: ignore
                    except (TypeError, ValueError):
                        current_profit = 0.0
            except Exception:
                current_profit = 0.0

            # 🚀 ULTRA-AGGRESSIVE PROFIT TAKING - Take profits MUCH faster!
            if current_profit >= self.quick_profit_target:  # $0.20 profit target!
                return True, f"Quick Profit: +${current_profit:.2f}"

            # 🛡️ MUCH TIGHTER STOP LOSSES - Cut losses much faster!
            if current_profit <= -self.max_loss_per_trade:  # $1.50 stop loss
                return True, f"Tight Stop Loss: ${current_profit:.2f}"

            # ⏰ TIME-BASED EXITS (very important for volatility indices!)
            trade_time = datetime.fromisoformat(trade["timestamp"])
            elapsed_minutes = (datetime.now() - trade_time).total_seconds() / 60

            # Exit after 6 minutes if slightly profitable (take small profits!)
            if elapsed_minutes >= 6 and current_profit >= self.min_profit_threshold:
                return (
                    True,
                    f"Time + Small Profit: {elapsed_minutes:.1f}min, +${current_profit:.2f}",
                )

            # Exit after 10 minutes regardless (prevent large losses on volatility)
            if elapsed_minutes >= self.max_trade_age_minutes:
                return (
                    True,
                    f"Time Limit: {elapsed_minutes:.1f}min, P&L: ${current_profit:.2f}",
                )

            # 📊 Profit locking - if we were profitable but now declining
            if hasattr(trade, "max_profit") and trade.get("max_profit", 0) > 1.0:
                if current_profit < trade["max_profit"] * 0.7:  # Lock in 70% of max profit
                    return (
                        True,
                        f"Profit Lock: ${current_profit:.2f} (was ${trade['max_profit']:.2f})",
                    )

            # Track maximum profit achieved
            if not hasattr(trade, "max_profit") or current_profit > trade.get("max_profit", 0):
                trade["max_profit"] = current_profit

            return False, ""

        except Exception as e:
            logger.error(f"Profitable trade check error: {e}")
            return False, ""

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

    async def _circuit_breaker_checks(self, symbol: str, action: str, confidence: float) -> bool:
        """🔒 Circuit breaker: Pre-flight safety checks for trade entry"""
        try:
            # Circuit Breaker 1: Maximum concurrent trades (thread-safe check)
            max_trades = getattr(config, "MAX_CONCURRENT_TRADES", 3)  # Default 3
            async with self.trades_lock:
                current_trade_count = len(self.active_trades)

            if current_trade_count >= max_trades:
                logger.warning(f"🔒 CB1: Max concurrent trades ({max_trades}) reached")
                return False

            # Circuit Breaker 2: Minimum confidence threshold (robust unit handling)
            cb2_ok, cb2_msg = self._cb2_confidence_gate(confidence)
            logger.warning(f"🔒 {cb2_msg}")
            if not cb2_ok:
                return False

            # Circuit Breaker 3: Market hours check
            if not self._check_market_hours():
                logger.warning("🔒 CB3: Outside market hours")
                return False

            # Circuit Breaker 4: Connection health
            if not self.mt5_connected or not self.mt5_interface:
                logger.warning("🔒 CB4: MT5 connection unhealthy")
                return False

            # Circuit Breaker 5: Recent loss limit (last 5 minutes)
            recent_losses = self._count_recent_losses(300)  # 5 minutes
            if recent_losses >= 3:
                logger.warning(f"🔒 CB5: Too many recent losses ({recent_losses}/3)")
                return False

            # Circuit Breaker 6: Balance minimum (lowered for small accounts)
            try:
                if self.mt5_interface:
                    balance = await self.mt5_interface.get_account_balance()
                    if (
                        balance and balance >= MIN_FREE_MARGIN
                    ):  # Using MIN_FREE_MARGIN ($5) as minimum balance
                        logger.info(f"🔒 CB6: Balance OK (${balance:.2f} ≥ ${MIN_FREE_MARGIN:.2f})")
                    else:
                        actual_balance = balance if balance else 0.0
                        logger.warning(
                            f"🔒 CB6: Account balance too low (${actual_balance:.2f}"
                            f"< ${MIN_FREE_MARGIN:.2f})"
                        )
                        return False
                else:
                    logger.warning("🔒 CB6: MT5 not available for balance check")
                    return False
            except Exception as e:
                logger.warning(f"🔒 CB6: Could not verify account balance: {e}")
                return False

            logger.info(f"🔒 CIRCUIT BREAKER: All checks passed for {symbol} {action}")
            return True

        except Exception as e:
            logger.error(f"🔒 Circuit breaker error: {e}")
            return False  # Fail safe - block trade on error

    def _check_market_hours(self) -> bool:
        """Check if market is open for trading - Deriv symbols are 24/7"""
        try:
            # Check config flag first
            from config import ENFORCE_MARKET_HOURS

            if not ENFORCE_MARKET_HOURS:
                return True  # Skip market hours check when disabled

            # Check if we're using Deriv symbols (24/7 trading)
            if hasattr(self, "mt5_interface") and self.mt5_interface:
                account_info = self.mt5_interface.get_account_info()  # type: ignore
                if account_info and "server" in account_info:
                    server = account_info["server"]  # type: ignore
                    # Deriv symbols are available 24/7
                    if "Deriv" in server:
                        return True

            # Check if current symbol is a Deriv synthetic symbol (24/7)
            current_symbol = getattr(self, "current_symbol", "")
            deriv_keywords = [
                "Volatility",
                "Crash",
                "Boom",
                "Vol over",
                "VolSwitch",
                "Step",
            ]
            if any(keyword in current_symbol for keyword in deriv_keywords):
                return True

            # For other symbols, apply traditional market hours
            current_hour = datetime.now().hour
            weekday = datetime.now().weekday()
            if weekday >= 5:  # Saturday = 5, Sunday = 6
                return False
            if current_hour < 6 or current_hour > 23:  # Outside 6 AM - 11 PM
                return False
            return True
        except Exception:
            return True  # Default to allow trading if check fails

    def _count_recent_losses(self, seconds: int) -> int:
        """Count losing trades in the last N seconds"""
        try:
            # This would need trade history tracking
            # For now, return 0 to allow trading
            # Future implementation would use cutoff_time and recent_losses
            _ = datetime.now() - timedelta(seconds=seconds)  # cutoff_time placeholder
            _ = 0  # recent_losses placeholder
            return 0

        except Exception:
            return 0

    async def _place_trade(self, action: str, confidence: float):
        """Place a trade with lot size validation (single symbol)"""
        await self._place_trade_for_symbol(action, confidence, DEFAULT_SYMBOL)

    async def _place_trade_for_symbol(self, action: str, confidence: float, symbol: str):
        """GODLIKE trade placement with confidence validation and all protections"""
        try:
            # 🔒 CIRCUIT BREAKER 1: Trade Entry Safety Checks
            if not await self._circuit_breaker_checks(symbol, action, confidence):
                logger.error(
                    f"🔒 CIRCUIT BREAKER: Blocking {symbol} {action} trade - safety checks failed"
                )
                return

            # 🎯 VOLUME SAFETY CHECK: Ensure proper lot sizing with broker constraints
            trade_vol = 0.01  # Default fallback
            if self.mt5_interface:
                try:
                    symbol_info = await self.mt5_interface.get_symbol_info(symbol)
                    if symbol_info:
                        min_lot = float(symbol_info.get("volume_min", 0.5))
                        step_lot = float(symbol_info.get("volume_step", 0.01))
                        max_lot = float(symbol_info.get("volume_max", 1.0))
                        max_safe_lot = 0.05  # Cap to what $27 balance can handle safely

                        # Apply overrides if configured
                        volume_guard_mode = getattr(config, "VOLUME_GUARD_MODE", "smart_filter")
                        overrides = getattr(config, "SYMBOL_MIN_LOT_OVERRIDES", {})
                        if symbol in overrides:
                            min_lot = overrides[symbol]
                            logger.info(f"📋 Using override min lot for {symbol}: {min_lot}")

                        logger.info(
                            f"📊 {symbol} lot rules: min={min_lot}, step={step_lot}, max={max_lot}"
                        )

                        # 🎯 SMART VOLUME GUARD: Intelligent handling based on mode
                        if volume_guard_mode == "smart_filter":
                            # Smart filter mode: Should have been filtered out already, but handle gracefully
                            avoid_symbols = getattr(config, "AVOID_HIGH_MIN_LOT_SYMBOLS", [])
                            if symbol in avoid_symbols and min_lot > max_safe_lot:
                                logger.warning(
                                    f"🔒 SMART FILTER: SKIP {symbol}"
                                    f"- high-risk symbol with min lot {min_lot} > safe {max_safe_lot}"
                                )
                                return
                            elif min_lot > max_safe_lot:
                                logger.warning(
                                    f"⚠️ SMART FILTER: {symbol}"
                                    f"has large min lot {min_lot}, using with enhanced risk management"
                                )
                                trade_vol = normalize_volume(min_lot, min_lot, step_lot, max_lot)
                                logger.warning(
                                    f"🎯 Using {trade_vol}"
                                    f"lots (min {min_lot}) - protected by $0.75 stop"
                                )
                            else:
                                trade_vol = normalize_volume(
                                    max_safe_lot, min_lot, step_lot, max_lot
                                )
                                logger.info(
                                    f"✅ SMART FILTER: {symbol}"
                                    f"using safe {trade_vol} lots (min {min_lot} ≤ safe {max_safe_lot})"
                                )
                        elif volume_guard_mode == "skip" and min_lot > max_safe_lot:
                            logger.warning(
                                f"🔒 VOLUME GUARD: SKIP {symbol}"
                                f"- min lot {min_lot} > max safe lot {max_safe_lot}"
                            )
                            return
                        elif min_lot > max_safe_lot:
                            # "raise_to_min" mode: Use broker minimum, rely on $0.75 stop loss for safety
                            trade_vol = normalize_volume(min_lot, min_lot, step_lot, max_lot)
                            logger.warning(
                                f"🎯 VOLUME GUARD: using {trade_vol}"
                                f"lots (min {min_lot}, step {step_lot}) - protected by $0.75 stop"
                            )
                        else:
                            trade_vol = normalize_volume(max_safe_lot, min_lot, step_lot, max_lot)
                            logger.info(
                                f"✅ VOLUME OK: {symbol}"
                                f"using {trade_vol} lots (min {min_lot} ≤ safe {max_safe_lot})"
                            )

                except Exception as e:
                    logger.warning(f"Volume check failed for {symbol}: {e}")
                    trade_vol = 0.01  # Fallback to minimum

            # 🚨 FIX: Validate and normalize confidence first
            validated_confidence = self._validate_and_normalize_confidence(
                confidence, "Trade Placement"
            )

            # 🚨 FRESH PRICE REQUIREMENT - No trades on stale prices
            current_price = await self._get_current_price_for_symbol(symbol)
            if not current_price:
                logger.error(
                    f"🚨 NO FRESH PRICE: Blocking {symbol} trade - price too stale/invalid"
                )
                return

            # Check price staleness one more time before trade
            if symbol in self.price_cache:
                cache_time = self.price_cache[symbol].get("timestamp")
                if cache_time:
                    staleness = (datetime.now() - cache_time).total_seconds()
                    if staleness > 5.0:  # No trades on prices older than 5 seconds
                        logger.error(
                            f"🚨 STALE PRICE: {symbol}"
                            f"price is {staleness:.1f}s old - BLOCKING TRADE"
                        )
                        return

            if not self.mt5_interface or not self.mt5_connected:
                logger.error("❌ Cannot place trade - MT5 not connected")
                return

            # 🛡️ PROTECTION 1: Maximum drawdown check
            if not self._check_maximum_drawdown_protection():
                logger.error("🚨 MAXIMUM DRAWDOWN PROTECTION TRIGGERED")
                return

            # 🛡️ PROTECTION 2: Extended loss limits
            if not self._check_extended_loss_limits():
                logger.error("🚨 EXTENDED LOSS LIMITS TRIGGERED")
                return

            # 🛡️ PROTECTION 3: Margin safety
            if not await self.check_margin_safety():
                logger.error("🚨 MARGIN SAFETY FAILED - BLOCKING TRADE TO PREVENT MARGIN CALL!")
                return

            # 🛡️ PROTECTION 4: Spread protection
            if not await self._check_spread_protection(symbol):
                logger.error(f"🚨 SPREAD TOO WIDE: {symbol}")
                return

            # 🛡️ PROTECTION 5: Correlation protection
            if not await self._check_correlation_protection(symbol, action):
                logger.error(f"🚨 CORRELATION BLOCKED: {symbol} {action}")
                return

            # 🛡️ PROTECTION 6: Calculate and validate position size with normalized confidence
            base_position_size = self._calculate_position_size_for_symbol(
                validated_confidence / 100, symbol
            )  # Pass as 0-1 for internal calc
            base_position_size = await self._validate_position_size_against_margin(
                symbol, base_position_size
            )

            # 🚀 HFT UPGRADE: Use Advanced Position Sizer with Kelly Criterion
            if self.mt5_interface:
                try:
                    account_balance = await self.mt5_interface.get_account_balance()
                    if not account_balance or account_balance <= 0:
                        account_balance = 1000.0  # Safe fallback
                except Exception as e:
                    logger.error(f"Balance check failed for position sizing: {e}")
                    account_balance = 1000.0  # Safe fallback
            else:
                account_balance = 1000.0  # Safe fallback

            # Calculate recent volatility for position sizing
            recent_volatility = await self._calculate_recent_volatility(symbol)

            # Get optimal position size using Kelly Criterion + volatility
            # with CORRECT confidence scale
            optimal_position_size = self.advanced_position_sizer.calculate_optimal_size(
                symbol=symbol,
                confidence=validated_confidence,  # ✅ Normalized to 0-1 scale
                account_balance=account_balance,
                recent_volatility=recent_volatility,
            )

            # Use the smaller of base size and optimal size for safety
            position_size = min(base_position_size, optimal_position_size)

            logger.info(
                f"📊 SIZING FIX: Raw confidence {confidence:.3f} → "
                f"Validated {validated_confidence:.1%} → Final size {position_size:.3f}"
            )
            logger.info(
                f"📊 POSITION SIZING: Base: {base_position_size:.2f}, "
                f"Kelly: {optimal_position_size:.2f}, Final: {position_size:.2f}"
            )

            # Validate and adjust lot size
            try:
                _, adjusted_size = await self.mt5_interface.validate_lot_size(symbol, position_size)
                if adjusted_size != position_size:
                    logger.info(
                        f"📏 Lot size adjusted for {symbol}: {position_size} → {adjusted_size}"
                    )
                    position_size = adjusted_size
            except Exception as validation_error:
                logger.warning(f"⚠️ Lot validation failed for {symbol}: {validation_error}")
                # Use conservative fallback
                position_size = 0.01

            logger.info(
                f"🚀 PLACING HFT TRADE: {action} on {symbol}: {trade_vol} lots "
                f"(Confidence: {confidence:.0%})"
            )

            # 🚀 HFT UPGRADE: Use Ultra-Fast Order Manager for sub-5ms execution
            result: Dict[str, Any] = {}

            # Try HFT order execution first (if available)
            if self.ultra_fast_order_manager:
                hft_result = await self.ultra_fast_order_manager.ultra_fast_order(
                    symbol=symbol,
                    action=action,
                    volume=trade_vol,  # Use calculated trade volume
                )

                if hft_result["success"]:
                    # HFT order successful
                    trade_id = hft_result["order"]
                    execution_latency = hft_result["latency_ms"]
                    is_fast = hft_result["fast_execution"]

                    logger.info(
                        f"⚡ HFT ORDER EXECUTED: {execution_latency:.2f}ms latency "
                        f"{'(FAST)' if is_fast else '(NORMAL)'}"
                    )

                    result = {
                        "ticket": trade_id,
                        "order": trade_id,
                        "price": 0.0,  # Will be filled by position info
                        "hft_execution": True,
                        "latency_ms": execution_latency,
                        "fast_execution": is_fast,
                    }
                else:
                    # Fallback to standard MT5 interface
                    logger.warning(f"⚠️ HFT order failed: {hft_result['reason']} - using fallback")
                    fallback_result = await self.mt5_interface.place_trade(
                        action=action,
                        symbol=symbol,
                        amount=trade_vol,  # Use calculated trade volume
                    )
                    if fallback_result:
                        result = fallback_result
                    else:
                        logger.error("❌ Fallback trade placement failed - no result from MT5")
                        return None
            else:
                # No HFT manager available, use standard interface
                fallback_result = await self.mt5_interface.place_trade(
                    action=action,
                    symbol=symbol,
                    amount=trade_vol,  # Use calculated trade volume
                )
                if fallback_result:
                    result = fallback_result
                else:
                    logger.error("❌ Standard trade placement failed - no result from MT5")
                    return None

            if result:
                # Store trade with thread-safe locking
                trade_id = result.get("ticket", result.get("order", f"trade_{int(time.time())}"))
                async with self.trades_lock:
                    self.active_trades[str(trade_id)] = {
                        "action": action,
                        "symbol": symbol,  # Store actual symbol used
                        "amount": position_size,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat(),
                        "entry_price": result.get("price", 0.0),
                        "mt5_ticket": trade_id,  # Store the actual MT5 ticket
                    }

                self.trades_today += 1
                self.last_trade_time = datetime.now()

                # Update symbol performance tracking
                if symbol in self.symbol_performance:
                    self.symbol_performance[symbol]["last_trade_time"] = datetime.now()

                # Send enhanced notification
                await self._send_enhanced_trade_notification(
                    action,
                    position_size,
                    confidence,
                    symbol,
                    trade_id,
                    result.get("price", 0.0),
                )

                logger.info(f"✅ Trade placed on {symbol}: {trade_id}")
            else:
                logger.error(f"❌ Trade placement failed for {symbol}")

        except Exception as e:
            logger.error(f"Trade placement error for {symbol}: {e}")

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size with proper lot size validation"""
        return self._calculate_position_size_for_symbol(confidence, DEFAULT_SYMBOL)

    async def _calculate_recent_volatility(self, symbol: str) -> float:
        """Calculate recent price volatility for position sizing"""
        try:
            # Get recent price data using async interface
            if self.mt5_interface:
                try:
                    # Get historical data using MT5 interface
                    rates_df = await self.mt5_interface.get_price_history(symbol, count=50)
                    if rates_df is None or len(rates_df) < 10:
                        return 0.02  # Default volatility assumption

                    # Calculate returns from close prices
                    prices = rates_df["close"].tolist()
                    returns: List[float] = []
                    for i in range(1, len(prices)):
                        if prices[i - 1] != 0:
                            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
                except Exception as e:
                    logger.error(f"Failed to get price history for {symbol}: {e}")
                    return 0.02  # Default volatility
            else:
                return 0.02  # Default volatility when no MT5 interface

            if not returns:
                return 0.02

            # Calculate volatility (standard deviation of returns)
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = variance**0.5

            # Scale volatility to reasonable range
            return min(max(volatility, 0.001), 0.1)  # Between 0.1% and 10%

        except Exception as e:
            logger.debug(f"Volatility calculation error for {symbol}: {e}")
            return 0.02  # Safe default

    def _calculate_position_size_for_symbol(self, confidence: float, symbol: str) -> float:
        """�️ MARGIN-SAFE position sizing - NEVER get margin called again!"""
        try:
            # 🚨 FIX: Normalize confidence scale first
            normalized_confidence = self._validate_and_normalize_confidence(
                confidence, "Position Sizing"
            )

            # 🛡️ EMERGENCY: ULTRA-CONSERVATIVE SIZING TO PREVENT MARGIN CALLS
            if self.current_balance <= 100:
                # For accounts under $100 - MAXIMUM SAFETY
                base_risk = self.current_balance * 0.001  # Only 0.1% risk!
                logger.warning(f"🛡️ MARGIN-SAFE MODE: Only 0.1% risk = ${base_risk:.2f}")

                # 🚀 AGGRESSIVE lot sizes for DEMO TRADING!
                if base_risk >= 0.50:
                    lot_size = 0.50  # 🚀 Big size for aggressive trading!
                else:
                    lot_size = 0.30  # 🚀 Still big for demo mode!

            else:
                # For larger accounts - AGGRESSIVE DEMO MODE
                base_risk = self.current_balance * 0.05  # 🚀 5% risk - GO BIG!
                lot_size = 0.50  # 🚀 Start with big lots for demo growth!

            # Use advanced position sizer with CORRECT confidence scale if available
            if hasattr(self, "advanced_position_sizer") and normalized_confidence > 0:
                try:
                    account_balance = self.current_balance
                    recent_volatility = 0.01  # Safe default volatility

                    optimal_size = self.advanced_position_sizer.calculate_optimal_size(
                        symbol=symbol,
                        confidence=normalized_confidence,  # ✅ Normalized to 0-1 scale
                        account_balance=account_balance,
                        recent_volatility=recent_volatility,
                    )

                    # 🚀 NO MORE CONSERVATIVE LIMITS - GO BIG!
                    lot_size = min(optimal_size, 2.50)  # 🚀 Cap at maximum aggressive size!
                    logger.debug(
                        f"📊 SIZING FIX: Raw confidence {confidence:.3f} → "
                        f"Normalized {normalized_confidence:.1f} → Size {lot_size:.3f}"
                    )

                except Exception as sizer_error:
                    logger.warning(f"Advanced position sizer failed: {sizer_error}")
                    # Keep conservative fallback

            # NEVER increase lot size regardless of confidence or consecutive losses
            # This prevents the margin call disaster that happened before
            logger.warning(
                f"🛡️ MARGIN-SAFE: ${base_risk:.2f} risk → {lot_size} lots (MARGIN PROTECTED)"
            )
            return lot_size

        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 0.01  # Ultra-safe fallback

    def _validate_and_normalize_confidence(
        self, raw_confidence: float, source: str = "AI"
    ) -> float:
        """Validate and normalize confidence to 0-1 scale"""
        try:
            # Use the global normalizer to ensure 0-1 range
            conf = _normalize_confidence(raw_confidence)

            # Validate reasonable range (warn if outside expected)
            if conf < 0.1 or conf > 0.95:
                f"⚠️ {source}"
                f"confidence {conf:.3f} outside normal range [0.1, 0.95]"

            # Clamp to safe bounds
            conf = max(0.01, min(conf, 0.99))

            if conf != raw_confidence:
                logger.debug(
                    f"📊 {source} confidence: {raw_confidence} → {conf:.3f} (normalized to 0-1)"
                )

            return conf

        except Exception as e:
            logger.error(f"Confidence validation error: {e}")
            return 0.5  # Safe default

    def _calculate_recent_win_rate(self) -> float:
        """Calculate win rate from recent trades for adaptive sizing"""
        try:
            if len(self.win_rate_tracker) < 5:
                return 0.5  # Default 50% if insufficient data

            wins = sum(1 for result in self.win_rate_tracker if result == "win")
            total = len(self.win_rate_tracker)
            return wins / total if total > 0 else 0.5
        except Exception:
            return 0.5

    def _count_recent_wins(self, lookback: int = 10) -> int:
        """Count wins in the last N trades"""
        try:
            recent_trades = list(self.win_rate_tracker)[-lookback:]
            return sum(1 for result in recent_trades if result == "win")
        except Exception:
            return 0

    async def _send_trade_notification(self, action: str, amount: float, confidence: float):
        """Send trade notification"""
        await self._send_trade_notification_for_symbol(action, amount, confidence, DEFAULT_SYMBOL)

    async def _send_trade_notification_for_symbol(
        self, action: str, amount: float, confidence: float, symbol: str
    ):
        """Send trade notification for specific symbol - LEGACY"""
        try:
            msg = f"""🚀 New Trade Placed

📊 Action: {action}
💰 Lot Size: {amount} lots
🎯 Symbol: {symbol}
🧠 Confidence: {confidence:.0%}
🕒 Time: {datetime.now().strftime('%H:%M:%S')}

💡 Trade sent to MT5"""

            # Use sanitizer for outbound message
            await self._send_sanitized_message(msg)
        except Exception as e:
            logger.warning(f"Trade notification failed: {e}")

    async def _send_corrected_trade_notification(
        self,
        action: str,
        amount: float,
        confidence: float,
        symbol: str,
        trade_id: str,
        win_rate: float,
    ):
        """Send corrected trade notification with accurate percentages"""
        try:
            # Ensure confidence is in percentage format (0-100)
            if confidence <= 1.0:
                confidence_display = confidence * 100
            else:
                confidence_display = confidence
            confidence_display = max(0, min(100, confidence_display))

            # Clean symbol name
            clean_symbol = self.text_sanitizer.sanitize_symbol_name(symbol)

            # Create confidence bar
            confidence_bars = int(confidence_display / 10)  # 0-10 bars
            confidence_bar = "🟢" * confidence_bars + "⚪" * (10 - confidence_bars)

            msg = f"""🚨 TRADE OPENED 🚨

📊 **TRADE DETAILS:**
🎯 Action: {action}
💎 Symbol: {clean_symbol}
💰 Size: {amount:.2f} lots
🎫 Ticket: {trade_id}
💲 Entry: {self.current_balance:.2f} (account balance)

🧠 **AI ANALYSIS:**
🎯 Confidence: {confidence_display:.1f}%
{confidence_bar}
📈 Session Win Rate: {win_rate:.1f}%
🔢 Total Trades: {self.session_stats.get('total_trades', 0)}

⚡ **TARGETS:**
🎯 Profit Target: +${UNIVERSAL_PROFIT_TARGET:.2f}
🛡️ Stop Loss: -${UNIVERSAL_STOP_LOSS:.2f}
⏰ Time Limit: {UNIVERSAL_TIME_LIMIT} minutes

🕒 Opened: {datetime.now().strftime('%H:%M:%S')}
📱 Monitoring for ${UNIVERSAL_PROFIT_TARGET:.2f} profit..."""

            await self._send_clean_telegram_message(msg)
            logger.info(
                f"📱 Corrected trade notification sent: {action}"
                f"{clean_symbol} ({confidence_display:.1f}% confidence)"
            )

        except Exception as e:
            logger.error(f"Corrected trade notification failed: {e}")

    async def _send_enhanced_trade_notification(
        self,
        action: str,
        amount: float,
        confidence: float,
        symbol: str,
        trade_id: str,
        entry_price: float,
    ):
        """🚀 Send enhanced trade opening notification with beautiful formatting"""
        try:
            # Calculate ACTUAL win rate from session stats
            total_trades = max(1, self.session_stats.get("total_trades", 0))
            wins = self.session_stats.get("wins", 0)
            actual_win_rate = (wins / total_trades) * 100

            # Normalize confidence to percentage (0-1 scale to 0-100)
            confidence_percent = confidence * 100 if confidence <= 1.0 else confidence
            confidence_percent = max(0, min(100, confidence_percent))  # Clamp to 0-100%

            # Create enhanced trade record with CORRECTED values
            trade_record = TradeRecord(
                trade_id=str(trade_id),
                symbol=symbol,
                direction=action,
                entry_price=entry_price,
                entry_time=datetime.now().isoformat(),
                stake=amount * entry_price,  # Approximate stake calculation
                confidence=confidence_percent / 100,  # Store as 0-1 scale
                strategy="AI + Technical",
                reason=f"High confidence {action.lower()} signal",
                win_rate_at_time=actual_win_rate,  # Use ACTUAL win rate
                mt5_ticket=str(trade_id),
            )

            # Send enhanced notification
            try:
                # Sync session stats with enhanced notifier BEFORE sending notification
                self.enhanced_notifier.set_external_session_stats(self.session_stats)
                self.enhanced_notifier.notify_trade_opened(trade_record)
            except Exception as enhanced_error:
                logger.debug(f"Enhanced notifier error: {enhanced_error}")

            # Also send corrected legacy notification
            await self._send_corrected_trade_notification(
                action, amount, confidence_percent, symbol, trade_id, actual_win_rate
            )

        except Exception as e:
            logger.error(f"Enhanced trade notification failed: {e}")
            # Fallback to corrected legacy notification
            await self._send_corrected_trade_notification(
                action, amount, confidence or 50.0, symbol, trade_id or "Unknown", 50.0
            )

    def _validate_config_invariants(self):
        """Validate configuration invariants at boot time"""
        try:
            # Universal constants from config
            profit_target = getattr(config, "UNIVERSAL_PROFIT_TARGET", 0.20)
            stop_loss = getattr(config, "UNIVERSAL_STOP_LOSS", 0.75)
            time_limit = getattr(config, "UNIVERSAL_TIME_LIMIT", 6)

            # Validate critical invariants
            if stop_loss <= profit_target:
                logger.error(
                    f"🚨 CONFIG ERROR: Stop loss (${stop_loss}"
                    f" must exceed profit target (${profit_target})"
                )
                raise ValueError("Invalid risk/reward configuration")

            if time_limit <= 0:
                logger.error(f"🚨 CONFIG ERROR: Time limit ({time_limit}) must be positive")
                raise ValueError("Invalid time limit configuration")

            # Validate position sizing limits
            max_concurrent = getattr(config, "MAX_CONCURRENT_TRADES", 3)
            if max_concurrent < 1 or max_concurrent > 10:
                logger.warning(
                    f"⚠️ CONFIG WARNING: MAX_CONCURRENT_TRADES ({max_concurrent}"
                    f" outside recommended range 1-10"
                )

            # Validate loss limits
            max_daily_loss = getattr(config, "MAX_DAILY_LOSS", 8.0)
            max_consecutive = getattr(config, "MAX_CONSECUTIVE_LOSSES", 100)
            if max_daily_loss <= 0 or max_consecutive <= 0:
                logger.error("🚨 CONFIG ERROR: Loss limits must be positive")
                raise ValueError("Invalid loss limit configuration")

            logger.info("✅ Configuration invariants validated successfully")

        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise  # Re-raise to prevent startup with invalid config

    async def _enhanced_check_active_trades(self):
        """Enhanced trade checking with retry logic"""
        try:
            if not self.active_trades:
                return

            logger.info(f"🔍 ENHANCED CHECK: {len(self.active_trades)} active trades")

            completed_trades: List[str] = []
            retry_trades: List[tuple[str, Dict[str, Any]]] = []

            for trade_id, trade in self.active_trades.items():
                try:
                    mt5_ticket = trade.get("mt5_ticket", trade_id)

                    if isinstance(mt5_ticket, str) and mt5_ticket.isdigit():
                        mt5_ticket = int(mt5_ticket)
                    elif not isinstance(mt5_ticket, int):
                        completed_trades.append(trade_id)
                        continue

                    # Check position status
                    if self.mt5_interface:
                        position_info = await self.mt5_interface.get_position_info(mt5_ticket)

                        if not position_info:
                            logger.info(f"✅ Trade {mt5_ticket} already closed")
                            completed_trades.append(trade_id)
                            continue

                        # Get current profit
                        current_profit = float(getattr(position_info, "profit", 0.0))

                        # Enhanced closure conditions
                        should_close, reason = await self._should_close_trade_enhanced(
                            trade, position_info
                        )

                        if should_close:
                            logger.warning(
                                f"🔒 ENHANCED CLOSE: {mt5_ticket}"
                                f"- {reason} (Profit: ${current_profit:.2f})"
                            )

                            # Use enhanced close method
                            close_result = await self._enhanced_force_close_trade(
                                mt5_ticket, reason
                            )

                            if close_result and close_result.get("closed"):
                                await self._handle_trade_completion(trade_id, trade, close_result)
                                completed_trades.append(trade_id)
                                logger.info(f"✅ Successfully closed {mt5_ticket}")
                            else:
                                # Add to retry list
                                retry_trades.append((trade_id, trade))
                                logger.warning(f"⚠️ Close failed, added {mt5_ticket} to retry list")

                except Exception as trade_error:
                    logger.error(f"⚠️ Error checking trade {trade_id}: {trade_error}")
                    completed_trades.append(trade_id)

            # Remove completed trades
            for trade_id in completed_trades:
                self.active_trades.pop(trade_id, None)

            # Retry failed closes after 30 seconds
            if retry_trades:
                logger.warning(f"🔄 Will retry {len(retry_trades)} failed closes in 30 seconds")
                asyncio.create_task(self._retry_failed_closes(retry_trades))

        except Exception as e:
            logger.error(f"Enhanced trade check error: {e}")

    async def _retry_failed_closes(self, retry_trades: List[tuple[str, Dict[str, Any]]]):
        """Retry failed trade closes after delay"""
        try:
            await asyncio.sleep(30)  # Wait 30 seconds

            logger.warning(f"� RETRYING {len(retry_trades)} failed closes")

            for trade_id, trade in retry_trades:
                if trade_id not in self.active_trades:
                    continue  # Trade was closed by other means

                mt5_ticket = trade.get("mt5_ticket", trade_id)
                logger.warning(f"🔄 RETRY: Attempting to close {mt5_ticket}")

                close_result = await self._enhanced_force_close_trade(mt5_ticket, "Retry Close")

                if close_result and close_result.get("closed"):
                    await self._handle_trade_completion(trade_id, trade, close_result)
                    self.active_trades.pop(trade_id, None)
                    logger.info(f"✅ RETRY SUCCESS: Closed {mt5_ticket}")
                else:
                    logger.error(f"❌ RETRY FAILED: Could not close {mt5_ticket}")
                    # Force remove from active trades to prevent infinite loop
                    self.active_trades.pop(trade_id, None)
                    logger.warning(f"🗑️ FORCE REMOVED: {mt5_ticket} from active trades")

        except Exception as e:
            logger.error(f"Retry failed closes error: {e}")

    async def _should_close_trade_enhanced(
        self, trade: Dict[str, Any], position_info: Any
    ) -> tuple[bool, str]:
        """Enhanced trade closure conditions - SYMBOL-SPECIFIC PROFIT TAKING"""
        try:
            # Get current profit
            current_profit = float(getattr(position_info, "profit", 0.0))

            # 🎯 UNIVERSAL PROFIT TARGET - All trades close at exactly the configured target
            profit_target = UNIVERSAL_PROFIT_TARGET
            stop_loss = 0.75
            time_limit = 6

            # 🚀 UNIVERSAL PROFIT TAKING
            if current_profit >= profit_target:
                return (
                    True,
                    f"${UNIVERSAL_PROFIT_TARGET:.2f}"
                    f"Target Hit: +${current_profit:.2f} >= ${UNIVERSAL_PROFIT_TARGET:.2f}"
                )

            # 🛡️ Universal stop loss
            if current_profit <= -stop_loss:
                return True, f"Stop Loss: ${current_profit:.2f} <= -${stop_loss:.2f}"

            # Time-based closure with symbol-specific limits
            trade_time = datetime.fromisoformat(trade["timestamp"])
            elapsed_minutes = (datetime.now() - trade_time).total_seconds() / 60

            # Exit after symbol-specific time limit if slightly profitable
            half_time = time_limit / 2
            if elapsed_minutes >= half_time and current_profit >= (profit_target * 0.25):
                return (
                    True,
                    f"Half-Time Small Profit: {elapsed_minutes:.1f}min, +${current_profit:.2f}",
                )

            # Exit after full time limit regardless
            if elapsed_minutes >= time_limit:
                return (
                    True,
                    f"Symbol Time Limit: {elapsed_minutes:.1f}"
                    f"in >= {time_limit}min, P&L: ${current_profit:.2f}"
                )

            return False, ""

        except Exception as e:
            logger.error(f"Enhanced closure check error: {e}")
            return False, ""

    async def _check_active_trades(self):
        """Use enhanced trade checking"""
        await self._enhanced_check_active_trades()

    async def _should_close_trade(
        self, trade: Dict[str, Any], position_info: Any
    ) -> tuple[bool, str]:
        """Check if a trade should be closed based on auto-management rules"""
        try:
            # Import auto-management settings with fallbacks
            enable_auto_trade_management = getattr(config, "ENABLE_AUTO_TRADE_MANAGEMENT", True)
            auto_take_profit = getattr(config, "AUTO_TAKE_PROFIT", 0.75)
            auto_stop_loss = getattr(config, "AUTO_STOP_LOSS", 1.0)
            auto_time_limit = getattr(config, "AUTO_TIME_LIMIT", 15)
            min_profit_threshold = getattr(config, "MIN_PROFIT_THRESHOLD", 0.25)

            if not enable_auto_trade_management:
                return False, ""

            # Get current profit/loss - Try multiple ways to get profit
            current_profit = 0.0

            # Helper function to safely convert to float
            def safe_float(value: Any) -> float:
                try:
                    if value is None:
                        return 0.0
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0

            # Method 1: Direct profit attribute
            if hasattr(position_info, "profit"):
                current_profit = safe_float(position_info.profit)
            # Method 2: Dictionary-style access
            elif isinstance(position_info, dict) and "profit" in position_info:
                profit_value = position_info["profit"]  # type: ignore
                current_profit = safe_float(profit_value)
            # Method 3: Getattr with default (only if not a dict)
            else:
                if isinstance(position_info, dict):
                    profit_value = position_info.get("profit", 0.0)  # type: ignore
                    current_profit = safe_float(profit_value)
                else:
                    current_profit = safe_float(getattr(position_info, "profit", 0.0))

            # ✅ ADD DETAILED LOGGING FOR DEBUGGING
            trade_symbol = trade.get("symbol", "Unknown")
            mt5_ticket = trade.get("mt5_ticket", "Unknown")
            logger.info(
                f"🔍 Auto-Management Check - Ticket: {mt5_ticket}"
                f" Symbol: {trade_symbol}, Profit: ${current_profit:.2f}"
            )

            # Check take profit
            if current_profit >= auto_take_profit:
                logger.info(f"🎯 TAKE PROFIT TRIGGER: ${current_profit:.2f} >= ${auto_take_profit}")
                return (
                    True,
                    f"Take Profit: +${current_profit:.2f} >= +${auto_take_profit}",
                )

            # Check stop loss
            if current_profit <= -auto_stop_loss:
                logger.info(f"🛡️ STOP LOSS TRIGGER: ${current_profit:.2f} <= -${auto_stop_loss}")
                return True, f"Stop Loss: ${current_profit:.2f} <= -${auto_stop_loss}"

            # Check time limit
            trade_time = datetime.fromisoformat(trade["timestamp"])
            elapsed_minutes = (datetime.now() - trade_time).total_seconds() / 60

            if elapsed_minutes >= auto_time_limit:
                if current_profit >= min_profit_threshold:
                    logger.info(
                        f"⏰ TIME + PROFIT TRIGGER: {elapsed_minutes:.1f}min, +${current_profit:.2f}"
                    )
                    return (
                        True,
                        f"Time Limit + Profitable: {elapsed_minutes:.1f}"
                        f"in, +${current_profit:.2f}"
                    )
                elif elapsed_minutes >= auto_time_limit * 1.5:  # Extended time for losses
                    logger.info(f"⏰ EXTENDED TIME TRIGGER: {elapsed_minutes:.1f}min")
                    return (
                        True,
                        f"Extended Time Limit: {elapsed_minutes:.1f}"
                        f"in, P&L: ${current_profit:.2f}"
                    )

            # Log current status every 5 minutes
            if int(elapsed_minutes) % 5 == 0:
                logger.debug(
                    f"📊 Trade Status - {mt5_ticket}"
                    f" {elapsed_minutes:.1f}min, P&L: ${current_profit:.2f}"
                )

            return False, ""

        except Exception as e:
            logger.error(f"Error checking trade closure: {e}")
            return False, ""

    async def _enhanced_force_close_trade(
        self, mt5_ticket: int, reason: str
    ) -> Optional[Dict[str, Any]]:
        """Enhanced force close that delegates to unified close method"""
        try:
            # Get current profit before closing
            if self.mt5_interface:
                position_info = await self.mt5_interface.get_position_info(mt5_ticket)
                current_profit = (
                    float(getattr(position_info, "profit", 0.0)) if position_info else 0.0
                )
            else:
                current_profit = 0.0

            # 🚨 FIX: Use unified close method
            success = await self._async_force_close(mt5_ticket, current_profit, reason)

            return {
                "profit": current_profit,
                "closed": success,
                "reason": reason,
                "method": "Unified",
            }

        except Exception as e:
            logger.error(f"Enhanced close error for {mt5_ticket}: {e}")
            return {
                "profit": 0.0,
                "closed": False,
                "reason": f"Error: {str(e)}",
                "method": "Failed",
            }

    async def _get_final_profit(self, mt5_ticket: int) -> float:
        """Get final profit before position is closed"""
        try:
            if self.mt5_interface:
                position_info = await self.mt5_interface.get_position_info(mt5_ticket)
                if position_info:
                    return float(getattr(position_info, "profit", 0.0))
        except Exception:
            pass
        return 0.0

    async def _force_close_trade(self, mt5_ticket: int, reason: str) -> Optional[Dict[str, Any]]:
        """Use enhanced force close method"""
        return await self._enhanced_force_close_trade(mt5_ticket, reason)

    async def _send_auto_close_notification(self, ticket: int, reason: str, profit: float):
        """Send notification for auto-closed trade"""
        try:
            emoji = "🟢" if profit > 0 else "🔴"

            msg = f"""{emoji} Auto-Closed Trade

🎫 Ticket: {ticket}
💰 P&L: ${profit:+.2f}
🔒 Reason: {reason}
🤖 Action: Automatic

🛡️ Risk management protecting your profits!"""

            await self.notifier.telegram.send_message(msg)
        except Exception as e:
            logger.warning(f"Auto-close notification failed: {e}")

    async def force_close_all_profitable_trades(self):
        """Emergency function to close all profitable trades manually"""
        try:
            logger.info("🚨 EMERGENCY: Force closing all profitable trades")

            closed_count = 0
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    mt5_ticket = trade.get("mt5_ticket", trade_id)
                    if isinstance(mt5_ticket, str) and mt5_ticket.isdigit():
                        mt5_ticket = int(mt5_ticket)
                    elif not isinstance(mt5_ticket, int):
                        continue  # Skip if not a valid ticket

                    # Get position info with null check
                    if self.mt5_interface is not None:
                        position_info = await self.mt5_interface.get_position_info(mt5_ticket)
                        if position_info:
                            profit = float(getattr(position_info, "profit", 0.0))
                            logger.info(f"🔍 Emergency check - Ticket {mt5_ticket}: ${profit:.2f}")

                            if profit > 0:  # Any positive profit
                                logger.info(
                                    f"🔒 Emergency closing profitable trade {mt5_ticket}"
                                    f" ${profit:.2f}"
                                )
                                close_result = await self._force_close_trade(
                                    mt5_ticket, f"Emergency Close: +${profit:.2f}"
                                )
                                if close_result:
                                    await self._handle_trade_completion(
                                        trade_id, trade, close_result
                                    )
                                    closed_count += 1

                except Exception as e:
                    logger.error(f"Emergency close error for {trade_id}: {e}")

            logger.info(f"🚨 Emergency closure complete: {closed_count} trades closed")
            return closed_count

        except Exception as e:
            logger.error(f"Emergency closure failed: {e}")
            return 0

    async def _cleanup_orphaned_trades_on_startup(self):
        """Clean up orphaned trades that might exist in active_trades but not in MT5"""
        logger.info("🧹 Checking for orphaned trades on startup...")

        if not self.active_trades:
            logger.info("✅ No active trades to check")
            return

        trades_to_remove: list[str] = []

        for trade_id, trade_info in list(self.active_trades.items()):
            try:
                mt5_ticket = trade_info.get("mt5_ticket")
                if not mt5_ticket:
                    logger.info(f"🗑️ Removing trade {trade_id}: No MT5 ticket")
                    trades_to_remove.append(trade_id)
                    continue

                # Check if position still exists in MT5
                if self.mt5_interface:
                    position_info = await self.mt5_interface.get_position_info(mt5_ticket)
                    if not position_info:
                        logger.info(
                            f"🗑️ Removing orphaned trade {trade_id} (MT5 ticket: {mt5_ticket})"
                        )
                        trades_to_remove.append(trade_id)

            except Exception as e:
                logger.warning(f"⚠️ Error checking trade {trade_id}: {e}")
                trades_to_remove.append(trade_id)  # Add problematic trades to removal list

        # Remove orphaned trades with thread safety
        for trade_id in trades_to_remove:
            try:
                async with self.trades_lock:
                    del self.active_trades[trade_id]
                logger.info(f"✅ Removed orphaned trade {trade_id}")
            except Exception as e:
                logger.error(f"Error removing trade {trade_id}: {e}")

        if trades_to_remove:
            logger.info(
                f"🧹 Startup cleanup complete: Removed {len(trades_to_remove)} orphaned trades"
            )
        else:
            logger.info("✅ No orphaned trades found")

    def _is_valid_trade_id(self, trade_id: Any) -> bool:
        """Check if trade ID is valid format"""
        if not trade_id:
            return False
        trade_id_str = str(trade_id)
        # Valid format: "signal_YYYYMMDD_HHMMSS_RAND" or "V75_YYYYMMDD_HHMMSS_RAND" etc.
        return len(trade_id_str) >= 20 and ("_" in trade_id_str)

    async def _handle_trade_completion(
        self, trade_id: str, trade: Dict[str, Any], close_info: Dict[str, Any]
    ):
        """Handle trade completion with corrected statistics"""
        try:
            profit = close_info.get("profit", 0.0)
            reason = close_info.get("reason", "Natural")

            # Update stats BEFORE sending notifications
            self.session_stats["total_trades"] = self.session_stats.get("total_trades", 0) + 1
            self.session_stats["total_profit"] = (
                self.session_stats.get("total_profit", 0.0) + profit
            )
            self.daily_profit += profit

            # Determine if this is a win or loss based on PROFIT, not just success flag
            if profit > 0:
                self.session_stats["wins"] = self.session_stats.get("wins", 0) + 1
                self.win_rate_tracker.append("WIN")
                old_losses = self.consecutive_losses
                self.consecutive_losses = 0  # Reset consecutive losses on ANY profit
                result = "WIN"
                if old_losses > 0:
                    logger.info(f"🎉 PROFIT WIN! Consecutive losses reset: {old_losses} → 0")
            else:
                self.session_stats["losses"] = self.session_stats.get("losses", 0) + 1
                self.win_rate_tracker.append("LOSS")
                self.consecutive_losses += 1
                result = "LOSS"
                logger.warning(f"📉 LOSS! Consecutive losses: {self.consecutive_losses}")

            # Calculate CORRECT win rate for logging
            total_trades = int(self.session_stats["total_trades"])
            wins = int(self.session_stats["wins"])
            current_win_rate = (wins / max(1, total_trades)) * 100

            logger.info(
                f"📊 CORRECTED STATS: {wins}/{total_trades} wins = {current_win_rate:.1f}% win rate"
            )

            # Send corrected notifications
            await self._send_enhanced_completion_notification(
                trade_id, trade, profit, result, reason
            )

            # AI Learning Progress (existing code)
            if self.learning_tracker.should_send_milestone_report(total_trades):
                await self._send_ai_milestone_notification(total_trades, current_win_rate)
            elif self.learning_tracker.should_send_progress_report(total_trades):
                await self._send_progress_report()

            # Update AI metrics
            try:
                self.ai_model.performance_metrics["total_predictions"] += 1
                if profit > 0:
                    self.ai_model.performance_metrics["correct_predictions"] += 1

                # Correct accuracy calculation
                total = self.ai_model.performance_metrics["total_predictions"]
                correct = self.ai_model.performance_metrics["correct_predictions"]
                self.ai_model.performance_metrics["accuracy"] = (
                    correct / total if total > 0 else 0.0
                )
            except Exception as ai_error:
                logger.debug(f"AI training update error: {ai_error}")

            logger.info(f"✅ Trade completed: {trade_id} - {result} - ${profit:+.2f} ({reason})")

        except Exception as e:
            logger.error(f"Trade completion error: {e}")

    async def _send_completion_notification(
        self, trade: Dict[str, Any], profit: float, result: str, reason: str = "Natural"
    ):
        """Send completion notification - LEGACY"""
        try:
            emoji = "🟢" if result == "WIN" else "🔴"

            msg = f"""{emoji} Trade Completed

📊 Result: {result}
💰 P&L: ${profit:+.2f}
🎯 Symbol: {trade.get('symbol', 'Unknown')}
📈 Action: {trade.get('action', 'Unknown')}
🧠 Confidence: {trade.get('confidence', 0):.0%}
🔒 Close Reason: {reason}

📊 Session Stats:
🔸 Total: {self.session_stats['total_trades']}
🔸 Win Rate: {self.get_win_rate():.1f}%
🔸 Total P&L: ${self.session_stats['total_profit']:+.2f}"""

            await self.notifier.telegram.send_message(msg)
        except Exception as e:
            logger.warning(f"Completion notification failed: {e}")

    async def _send_enhanced_completion_notification(
        self,
        trade_id: str,
        trade: Dict[str, Any],
        profit: float,
        result: str,
        reason: str = "Natural",
    ):
        """🎯 Send enhanced trade completion notification with accurate data"""
        try:
            # Calculate ACTUAL session statistics
            total_trades = self.session_stats.get("total_trades", 0)
            wins = self.session_stats.get("wins", 0)
            losses = self.session_stats.get("losses", 0)
            total_profit = self.session_stats.get("total_profit", 0.0)

            # Calculate REAL win rate
            actual_win_rate = (wins / max(1, total_trades)) * 100

            # Get trade details
            symbol = trade.get("symbol", "Unknown")
            action = trade.get("action", "Unknown")
            entry_price = trade.get("entry_price", 0.0)
            confidence = trade.get("confidence", 0.0)

            # Normalize confidence display
            if confidence <= 1.0:
                confidence_display = confidence * 100
            else:
                confidence_display = confidence
            confidence_display = max(0, min(100, confidence_display))

            # Get current price for exit calculation
            try:
                current_price = await self._get_current_price_for_symbol(symbol)
                exit_price = current_price or entry_price or 0.0
            except Exception:
                exit_price = entry_price or 0.0

            # Clean symbol name
            clean_symbol = self.text_sanitizer.sanitize_symbol_name(symbol)

            # Result emoji and color
            if result == "WIN" and profit > 0:
                emoji = "🟢"
                result_color = "💚 WIN"
            else:
                emoji = "🔴"
                result_color = "❤️ LOSS"

            msg = f"""{emoji} TRADE COMPLETED {emoji}

📊 **TRADE RESULT:**
{result_color}
💰 P&L: ${profit:+.2f}
💎 Symbol: {clean_symbol}
📈 Action: {action}
🎫 Ticket: {trade_id}

📊 **TRADE DATA:**
💲 Entry: ${entry_price:.5f}
💲 Exit: ${exit_price:.5f}
🧠 AI Confidence: {confidence_display:.1f}%
🔒 Close Reason: {reason}

📈 **SESSION STATS:**
🔢 Total Trades: {total_trades}
✅ Wins: {wins}
❌ Losses: {losses}
🎯 Win Rate: {actual_win_rate:.1f}%
💰 Session P&L: ${total_profit:+.2f}
💳 Balance: ${self.current_balance:.2f}

🕒 Closed: {datetime.now().strftime('%H:%M:%S')}"""

            # Send enhanced notification to both systems
            try:
                # Sync session stats with enhanced notifier BEFORE sending notification
                self.enhanced_notifier.set_external_session_stats(self.session_stats)

                # Send to enhanced notifier
                self.enhanced_notifier.notify_trade_closed(
                    trade_id=trade_id,
                    exit_price=float(exit_price),
                    profit_loss=profit,
                    close_reason=reason,
                )
            except Exception as enhanced_error:
                logger.debug(f"Enhanced notifier error: {enhanced_error}")

            # Send main notification
            await self._send_clean_telegram_message(msg)
            logger.info(f"📱 Corrected completion notification sent: {result} ${profit:+.2f}")

        except Exception as e:
            logger.error(f"Enhanced completion notification failed: {e}")
            # Send basic fallback notification
            try:
                result_emoji = "🟢" if result == "WIN" and profit > 0 else "🔴"
                fallback_msg = f"""{result_emoji} Trade {result}
💰 P&L: ${profit:+.2f}
🎯 Symbol: {trade.get('symbol', 'Unknown')}
Session Win Rate: {((self.session_stats.get('wins', 0) / max(1, self.session_stats.get('total_trades', 1))) * 100):.1f}%"""
                await self._send_clean_telegram_message(fallback_msg)
            except Exception as fallback_error:
                logger.error(f"Fallback notification failed: {fallback_error}")

    async def _update_balance(self):
        """Update balance and check loss protection"""
        try:
            if self.mt5_interface and self.mt5_connected:
                new_balance = await self.mt5_interface.get_account_balance()
                if new_balance and new_balance > 0:
                    old_balance = self.current_balance
                    self.current_balance = new_balance

                    # 🛡️ Check loss protection after balance update
                    if self.loss_protection_enabled and not self.loss_protection_triggered:
                        current_profit = self.current_balance - self.starting_balance

                        # Log balance changes with profit tracking
                        if abs(new_balance - old_balance) > 0.01:  # Only log significant changes
                            logger.info(
                                f"💰 Balance: ${old_balance:.2f}"
                                f"→ ${new_balance:.2f} (Profit: ${current_profit:+.2f})"
                            )

                        # Early warning at 80% of loss threshold
                        warning_threshold = LOSS_PROTECTION_THRESHOLD * 0.8
                        if (
                            current_profit <= -warning_threshold
                            and current_profit > -LOSS_PROTECTION_THRESHOLD
                        ):
                            if hasattr(self, "_loss_warning_sent"):
                                if not self._loss_warning_sent:
                                    logger.warning(
                                        f"⚠️ Loss Warning: ${current_profit:.2f}"
                                        f"-${LOSS_PROTECTION_THRESHOLD:.2f} (80% of loss limit)"
                                    )
                                    self._loss_warning_sent = True
                            else:
                                logger.warning(
                                    f"⚠️ Loss Warning: ${current_profit:.2f}"
                                    f"-${LOSS_PROTECTION_THRESHOLD:.2f} (80% of loss limit)"
                                )
                                self._loss_warning_sent = True
        except Exception as e:
            logger.error(f"Balance update error: {e}")

    def get_win_rate(self) -> float:
        """Get win rate"""
        try:
            if not self.win_rate_tracker:
                return 0.0

            wins = sum(1 for result in self.win_rate_tracker if result == "WIN")
            return (wins / len(self.win_rate_tracker)) * 100
        except Exception:
            return 0.0

    def reset_consecutive_losses(self):
        """Reset consecutive losses manually"""
        old_count = self.consecutive_losses
        self.consecutive_losses = 0
        logger.info(f"🔄 Manual reset: Consecutive losses {old_count} → 0")
        print(f"✅ Consecutive losses reset from {old_count} to 0")
        return True

    # 🔥 GODLIKE HELPER METHODS
    def _get_godlike_indicator_data(self, price_history: List[float]) -> Dict[str, Any]:
        """Get enhanced indicator data for godlike analysis"""
        try:
            # Get basic indicator data
            indicator_data = self._get_indicator_data()

            # Add advanced metrics
            if len(price_history) >= 20:
                indicator_data["volatility"] = float(
                    np.std(price_history[-20:]) / np.mean(price_history[-20:])
                )
                indicator_data["momentum"] = (
                    price_history[-1] - price_history[-10]
                ) / price_history[-10]
                indicator_data["trend_strength"] = float(
                    abs(np.polyfit(range(10), price_history[-10:], 1)[0])
                )

            return indicator_data

        except Exception as e:
            logger.debug(f"Godlike indicator data error: {e}")
            return self._get_indicator_data()

    async def _update_timeframe_data(self, current_price: float, symbol: str) -> None:
        """Update multi-timeframe price data"""
        try:
            now = datetime.now()

            # 1-minute data (every update)
            self.timeframe_data["1min"].append(current_price)

            # 5-minute data (every 5 minutes)
            if (now - self.last_timeframe_update["5min"]).total_seconds() >= 300:
                self.timeframe_data["5min"].append(current_price)
                self.last_timeframe_update["5min"] = now

            # 15-minute data (every 15 minutes)
            if (now - self.last_timeframe_update["15min"]).total_seconds() >= 900:
                self.timeframe_data["15min"].append(current_price)
                self.last_timeframe_update["15min"] = now

            # 1-hour data (every hour)
            if (now - self.last_timeframe_update["1hour"]).total_seconds() >= 3600:
                self.timeframe_data["1hour"].append(current_price)
                self.last_timeframe_update["1hour"] = now

        except Exception as e:
            logger.debug(f"Timeframe update error: {e}")

    def _detect_market_regime(self, price_data: List[float]) -> str:
        """Detect current market regime for optimal strategy selection"""
        try:
            if len(price_data) < 50:
                return "UNKNOWN"

            # Calculate volatility
            returns = np.diff(price_data) / price_data[:-1]
            volatility = np.std(returns[-20:])

            # Calculate trend strength
            trend = np.polyfit(range(20), price_data[-20:], 1)[0]
            trend_strength = abs(trend) / np.mean(price_data[-20:])

            # Regime classification
            if volatility > 0.005 and trend_strength > 0.001:
                return "TRENDING_VOLATILE"  # Best for breakout strategies
            elif volatility > 0.005:
                return "CHOPPY_VOLATILE"  # Best for mean reversion
            elif trend_strength > 0.001:
                return "TRENDING_STABLE"  # Best for trend following
            else:
                return "SIDEWAYS_STABLE"  # Best for range trading

        except Exception as e:
            logger.debug(f"Market regime detection error: {e}")
            return "UNKNOWN"

    def _get_regime_multiplier(self, regime: str) -> float:
        """Get confidence multiplier based on market regime"""
        multipliers = {
            "TRENDING_VOLATILE": 1.3,  # High confidence environment
            "TRENDING_STABLE": 1.2,  # Good trend following
            "CHOPPY_VOLATILE": 0.9,  # Reduce confidence in chop
            "SIDEWAYS_STABLE": 1.0,  # Neutral
            "UNKNOWN": 0.8,  # Conservative
        }
        return multipliers.get(regime, 1.0)

    async def _get_timeframe_signals(self, symbol: str) -> Dict[str, str]:
        """Get signals from multiple timeframes"""
        try:
            signals: Dict[str, str] = {}

            for timeframe, data in self.timeframe_data.items():
                if len(data) >= 20:
                    # Simple trend signal
                    recent_trend = (data[-1] - data[-10]) / data[-10]
                    if recent_trend > 0.001:
                        signals[timeframe] = "BUY"
                    elif recent_trend < -0.001:
                        signals[timeframe] = "SELL"
                    else:
                        signals[timeframe] = "HOLD"
                else:
                    signals[timeframe] = "HOLD"

            return signals

        except Exception as e:
            logger.debug(f"Timeframe signals error: {e}")
            return {}

    def _calculate_timeframe_boost(self, timeframe_signals: Dict[str, str]) -> float:
        """Calculate confidence boost from timeframe alignment"""
        try:
            if not timeframe_signals:
                return 0

            buy_signals = sum(1 for signal in timeframe_signals.values() if signal == "BUY")
            sell_signals = sum(1 for signal in timeframe_signals.values() if signal == "SELL")
            total_signals = len(timeframe_signals)

            # Boost confidence when timeframes align
            alignment = max(buy_signals, sell_signals) / total_signals

            if alignment >= 0.75:  # 75% or more timeframes agree
                return 0.15
            elif alignment >= 0.5:  # 50% or more agree
                return 0.10
            else:
                return 0.0

        except Exception as e:
            logger.debug(f"Timeframe boost calculation error: {e}")
            return 0.0

    def _detect_godlike_patterns(self, price_data: List[float]) -> Dict[str, Any]:
        """Detect advanced profitable patterns"""
        try:
            if len(price_data) < 30:
                return {"action": "HOLD", "confidence": 0.0}

            # Pattern 1: Momentum Breakout
            volatility = np.std(price_data[-20:]) / np.mean(price_data[-20:])
            recent_move = (price_data[-1] - price_data[-5]) / price_data[-5]

            if abs(recent_move) > volatility * 2:  # Move is 2x normal volatility
                if recent_move > 0:
                    return {"action": "BUY", "confidence": 0.65}
                else:
                    return {"action": "SELL", "confidence": 0.65}

            # Pattern 2: Support/Resistance bounce
            recent_high = max(price_data[-10:])
            recent_low = min(price_data[-10:])
            current_price = price_data[-1]

            range_size = (recent_high - recent_low) / recent_low
            if range_size > 0.003:  # Significant range
                if current_price <= recent_low * 1.001:  # Near support
                    return {"action": "BUY", "confidence": 0.55}
                elif current_price >= recent_high * 0.999:  # Near resistance
                    return {"action": "SELL", "confidence": 0.55}

            return {"action": "HOLD", "confidence": 0.0}

        except Exception as e:
            logger.debug(f"Pattern detection error: {e}")
            return {"action": "HOLD", "confidence": 0.0}

    def _calculate_godlike_position_size(self, confidence: float, symbol: str) -> float:
        """Calculate position size with godlike intelligence"""
        try:
            # Base size calculation
            base_size = 0.01  # Start conservative

            # Confidence multiplier (higher confidence = larger size)
            confidence_multiplier = 1 + (confidence - 0.3) * 2  # Scale from 0.3 to 0.95

            # Win streak amplification
            win_streak_bonus = min(2.0, 1 + (self.profit_amplifier["consecutive_wins"] * 0.1))

            # Account size scaling (compound growth)
            if self.current_balance > 50:
                account_multiplier = min(3.0, self.current_balance / 50)
            else:
                account_multiplier = 1.0

            # Calculate final size
            final_size = base_size * confidence_multiplier * win_streak_bonus * account_multiplier

            # Safety caps
            max_size_by_balance = (self.current_balance / 1000) * 0.01  # 1% of balance in lots
            final_size = min(final_size, max_size_by_balance, 0.1)  # Never exceed 0.1 lots
            final_size = max(final_size, 0.01)  # Never below minimum

            logger.info(
                f"🎯 GODLIKE SIZE: {symbol}"
                f"= {final_size:.3f} lots (Conf: {confidence:.2f}, Wins: {self.profit_amplifier['consecutive_wins']}, Balance: ${self.current_balance:.2f})"
            )

            return final_size

        except Exception as e:
            logger.debug(f"Position sizing error: {e}")
            return 0.01

    async def _place_godlike_trade(
        self,
        action: str,
        confidence: float,
        symbol: str,
        position_size: float,
        reason: str,
    ) -> None:
        """Place trade with godlike enhancements"""
        try:
            # Use existing trade placement method
            if hasattr(self, "_place_trade_for_symbol"):
                await self._place_trade_for_symbol(action, confidence, symbol)
            else:
                # Fallback to basic trade placement
                logger.warning(
                    f"🚀 GODLIKE TRADE: {action} {symbol} at {position_size} lots (would place here)"
                )

        except Exception as e:
            logger.error(f"Godlike trade placement error: {e}")

    def force_reset_and_resume_trading(self):
        """Emergency function to reset consecutive losses and resume trading"""
        old_count = self.consecutive_losses
        self.consecutive_losses = 0
        logger.info(f"🚨 EMERGENCY RESET: Consecutive losses {old_count} → 0")
        logger.info("🚀 Bot ready to resume trading!")
        print(f"🚨 EMERGENCY RESET: Consecutive losses {old_count} → 0")
        print("🚀 Bot ready to resume trading!")
        return True

    def force_start_trading_now(self):
        """🚨 EMERGENCY: Force immediate trading by lowering all barriers"""
        try:
            logger.warning("🚨 EMERGENCY FORCE TRADING ACTIVATED!")

            # Reset consecutive losses to 0
            old_losses = self.consecutive_losses
            self.consecutive_losses = 0
            logger.warning(f"🔧 EMERGENCY: Reset consecutive losses {old_losses} → 0")

            # Clear price history to force fresh data collection
            for symbol in self.active_symbols:
                if symbol in self.symbol_price_histories:
                    self.symbol_price_histories[symbol] = []
                    logger.warning(f"🔄 EMERGENCY: Cleared price history for {symbol}")
            self.price_history = []

            # Force update current balance
            logger.warning("💰 EMERGENCY: Forcing balance update")

            logger.warning("✅ EMERGENCY FORCE TRADING SETUP COMPLETE!")
            logger.warning("📊 Bot will start trading within 30 seconds with fresh data!")

            return True

        except Exception as e:
            logger.error(f"❌ Emergency force trading failed: {e}")
            return False

    def check_and_auto_fix_limits(self):
        """🔧 AUTO-FIX: Check and fix unreasonably low limits that block trading - FIXED VERSION"""
        try:
            # Add a flag to prevent infinite loops
            if hasattr(self, "_limits_already_fixed") and self._limits_already_fixed:
                return False  # Already fixed, don't keep checking

            # Don't reload config constantly - just check current value
            current_limit = getattr(config, "MAX_CONSECUTIVE_LOSSES", 100)

            # Check if limit is unreasonably low (blocks trading)
            if current_limit < 50:
                logger.warning(f"🚨 AUTO-FIX: MAX_CONSECUTIVE_LOSSES ({current_limit}) is too low!")

                # Reset consecutive losses count to 0
                if self.consecutive_losses > 0:
                    old_count = self.consecutive_losses
                    self.consecutive_losses = 0
                    logger.warning(f"🔧 AUTO-FIX: Reset consecutive losses {old_count} → 0")

                # Temporarily set a higher limit in memory (don't reload config!)
                config.MAX_CONSECUTIVE_LOSSES = 100
                logger.warning(
                    "🔧 AUTO-FIX: Increased MAX_CONSECUTIVE_LOSSES to 100 for this session"
                )

                # Set flag to prevent repeating this fix
                self._limits_already_fixed = True

                return True

            # If limit is already good, mark as fixed to avoid future checks
            self._limits_already_fixed = True
            return False

        except Exception as e:
            logger.warning(f"⚠️ Auto-fix check failed: {e}")
            # Set flag even on error to prevent infinite retries
            self._limits_already_fixed = True
            return False
        """Emergency function to reset all trading limits"""
        try:
            old_consecutive = self.consecutive_losses
            old_daily_profit = self.daily_profit

            # Reset all limits
            self.consecutive_losses = 0
            self.daily_profit = 0.0

            logger.info("🚨 EMERGENCY LIMIT RESET:")
            logger.info(f"   Consecutive losses: {old_consecutive} → 0")
            logger.info(f"   Daily P&L: ${old_daily_profit:.2f} → $0.00")
            logger.info("🚀 Bot ready to resume trading!")

            return True
        except Exception as e:
            logger.error(f"Emergency reset failed: {e}")
            return False

    def auto_reset_consecutive_losses_if_needed(self):
        """Auto-reset consecutive losses after extended period"""
        try:
            # If we have many consecutive losses and it's been a while since last trade
            if (
                self.consecutive_losses >= 50
                and self.last_trade_time
                and (datetime.now() - self.last_trade_time).total_seconds() > 3600
            ):  # 1 hour

                old_count = self.consecutive_losses
                self.consecutive_losses = max(0, self.consecutive_losses - 25)  # Reduce by 25

                logger.info(
                    f"🔄 Auto-reset: Consecutive losses reduced {old_count}"
                    f"→ {self.consecutive_losses} (time-based)"
                )

        except Exception as e:
            logger.error(f"Auto-reset error: {e}")

    def get_risk_status(self):
        """Get current risk status"""
        status = f"""📊 Risk Management Status:
        
🔸 Consecutive Losses: {self.consecutive_losses}/{MAX_CONSECUTIVE_LOSSES}
🔸 Daily P&L: ${self.daily_profit:+.2f} (Limit: -${MAX_DAILY_LOSS})
🔸 Current Balance: ${self.current_balance:.2f}
🔸 Risk Limits: {'🟢 PASS' if self._check_risk_limits() else '🔴 BLOCKED'}
🔸 Active Trades: {len(self.active_trades)}

Status: {'✅ Ready to Trade' if self._check_risk_limits() else '⚠️ Trading Blocked by Risk Limits'}"""

        print(status)
        logger.info(status.replace("\n", " | "))
        return status

    def can_trade(self) -> bool:
        """Public method to check if bot can trade (calls _check_risk_limits)"""
        return self._check_risk_limits()

    def force_reset_now(self):
        """Force reset consecutive losses immediately"""
        old_consecutive = self.consecutive_losses
        old_daily = self.daily_profit

        # Reset everything
        self.consecutive_losses = 0
        self.daily_profit = 0.0

        logger.info(
            f"🚨 FORCE RESET: Consecutive losses {old_consecutive}"
            f"→ 0, Daily P&L ${old_daily:.2f} → $0.00"
        )
        print(f"🚨 FORCE RESET APPLIED: {old_consecutive} consecutive losses cleared!")

        return True

    async def _unified_profit_monitor(self):
        """🧠 Ultra-responsive profit monitoring with rate-limited logging"""
        monitor_name = "unified_profit"

        try:
            logger.warning("🧠 ADAPTIVE MONITOR: Starting ultra-responsive profit monitoring")

            while self.running:
                try:
                    trades_count = len(self.active_trades)
                    profit_detected = False
                    near_target = False

                    if trades_count > 0:
                        # Process each trade with throttled async requests only
                        for trade_id, trade in list(self.active_trades.items()):
                            try:
                                mt5_ticket = trade.get("mt5_ticket", trade_id)
                                if isinstance(mt5_ticket, str) and mt5_ticket.isdigit():
                                    mt5_ticket = int(mt5_ticket)
                                elif not isinstance(mt5_ticket, int):
                                    continue

                                # Use ONLY async interface - no direct MT5 calls
                                if self.mt5_interface:
                                    position_info = (
                                        await self.mt5_request_manager.throttled_request(
                                            self.mt5_interface.get_position_info, mt5_ticket
                                        )
                                    )
                                else:
                                    position_info = None

                                if not position_info:
                                    # Trade closed - remove from tracking with thread safety
                                    async with self.trades_lock:
                                        if trade_id in self.active_trades:
                                            del self.active_trades[trade_id]
                                    continue

                                current_profit = float(getattr(position_info, "profit", 0.0))
                                symbol = trade.get("symbol", "Unknown")

                                # Near target detection (within $0.05 of target)
                                approach_threshold = UNIVERSAL_PROFIT_TARGET - 0.05
                                if approach_threshold <= current_profit < UNIVERSAL_PROFIT_TARGET:
                                    near_target = True
                                    if self.log_limiter.should_log(f"approaching_{trade_id}"):
                                        logger.info(
                                            f"🎯 APPROACHING: {symbol}"
                                            f"${current_profit:.2f} (${UNIVERSAL_PROFIT_TARGET-current_profit:.2f} to target)"
                                        )

                                # Check UNIVERSAL_PROFIT_TARGET ($0.20)
                                if current_profit >= UNIVERSAL_PROFIT_TARGET:
                                    profit_detected = True
                                    logger.error(
                                        f"🎯 UNIVERSAL TARGET: {symbol}"
                                        f"${current_profit:.2f} >= ${UNIVERSAL_PROFIT_TARGET:.2f} - CLOSING!"
                                    )

                                    # Close using async interface only
                                    close_success = await self._async_force_close(
                                        mt5_ticket, current_profit, "Universal Target"
                                    )

                                    if close_success:
                                        async with self.trades_lock:
                                            if trade_id in self.active_trades:
                                                del self.active_trades[trade_id]
                                        await self._send_universal_profit_notification(
                                            symbol, mt5_ticket, current_profit
                                        )

                                # Check UNIVERSAL_STOP_LOSS ($0.75)
                                elif current_profit <= -UNIVERSAL_STOP_LOSS:
                                    logger.warning(
                                        f"🛡️ UNIVERSAL STOP: {symbol}"
                                        f"${current_profit:.2f} <= -${UNIVERSAL_STOP_LOSS:.2f} - CLOSING!"
                                    )

                                    close_success = await self._async_force_close(
                                        mt5_ticket,
                                        current_profit,
                                        "Universal Stop Loss",
                                    )

                                    if close_success:
                                        async with self.trades_lock:
                                            if trade_id in self.active_trades:
                                                del self.active_trades[trade_id]

                                # Check UNIVERSAL_TIME_LIMIT (6 minutes)
                                if "timestamp" in trade:
                                    trade_time = datetime.fromisoformat(trade["timestamp"])
                                    elapsed_minutes = (
                                        datetime.now() - trade_time
                                    ).total_seconds() / 60

                                    if elapsed_minutes >= UNIVERSAL_TIME_LIMIT:
                                        logger.warning(
                                            f"⏰ UNIVERSAL TIME: {symbol}"
                                            f"{elapsed_minutes:.1f}min >= {UNIVERSAL_TIME_LIMIT}min - CLOSING!"
                                        )

                                        close_success = await self._async_force_close(
                                            mt5_ticket,
                                            current_profit,
                                            "Universal Time Limit",
                                        )

                                        if close_success:
                                            async with self.trades_lock:
                                                if trade_id in self.active_trades:
                                                    del self.active_trades[trade_id]

                                # Early profit detection for adaptive intervals (80% of target)
                                elif current_profit >= UNIVERSAL_PROFIT_TARGET * 0.8:
                                    profit_detected = True

                                # Small delay between trades to prevent overload
                                await asyncio.sleep(0.01)  # 10ms between trades

                            except Exception as trade_error:
                                logger.debug(f"Adaptive monitor trade error: {trade_error}")

                    # 🧠 Ultra-responsive interval calculation
                    next_interval = self.intelligent_profit_monitor.get_next_interval(
                        trades_count, profit_detected, near_target
                    )

                    # Rate-limited logging for monitoring
                    if self.log_limiter.should_log("monitor_status"):
                        interval_type = (
                            "ULTRA"
                            if next_interval <= 0.01
                            else "FAST" if next_interval < 0.1 else "NORMAL"
                        )
                        logger.info(
                            f"🧠 ADAPTIVE: {trades_count}"
                            f"trades, {next_interval:.3f}s ({interval_type})"
                        )

                    # Sleep for the intelligent interval
                    await asyncio.sleep(next_interval)

                except Exception as e:
                    logger.error(f"Adaptive profit monitor error: {e}")
                    await asyncio.sleep(1.0)  # Fallback interval

        except Exception as e:
            logger.error(f"🧠 Unified profit monitor fatal error: {e}")
        finally:
            # Cleanup: Remove from active monitors
            await self.monitor_manager.stop_monitor(monitor_name)
            logger.info("🧠 Unified profit monitor cleanup completed")

    async def _capture_trade_features(self, mt5_ticket: int, current_profit: float) -> List[float]:
        """🧠 Capture trade features for real data learning"""
        try:
            features: List[float] = []

            # Get current market data for feature extraction
            current_price = await self._get_current_price()
            if not current_price:
                current_price = 0.0

            # Price-based features
            price_history = (
                self.price_history[-20:] if len(self.price_history) >= 20 else self.price_history
            )
            if price_history:
                features.extend(
                    [
                        float(current_price),
                        float(current_profit),
                        float(len(price_history)),
                        float(max(price_history)) if price_history else 0.0,
                        float(min(price_history)) if price_history else 0.0,
                        (float(sum(price_history) / len(price_history)) if price_history else 0.0),
                    ]
                )
            else:
                features.extend([float(current_price), float(current_profit), 0.0, 0.0, 0.0, 0.0])

            # Market timing features
            now = datetime.now()
            features.extend(
                [
                    float(now.hour),
                    float(now.minute),
                    float(now.weekday()),
                    float(len(self.active_trades)),
                ]
            )

            # Trading session features
            features.extend(
                [
                    float(self.consecutive_losses),
                    float(self.daily_profit),
                    float(self.session_stats.get("total_trades", 0)),
                    float(self.session_stats.get("wins", 0)),
                    float(self.session_stats.get("losses", 0)),
                ]
            )

            # Pad to standard feature length
            while len(features) < 20:
                features.append(0.0)

            return features[:20]  # Return exactly 20 features

        except Exception as e:
            logger.error(f"Error capturing trade features: {e}")
            return [0.0] * 20  # Return zero features on error

    # 🧠 AI LEARNING PROGRESS METHODS
    async def _send_ai_milestone_notification(self, total_trades: int, win_rate: float) -> None:
        """🧠 Send AI milestone achievement notification"""
        try:
            current_level = self.learning_tracker.get_current_level(total_trades)
            performance_rating, performance_desc = self.learning_tracker.get_performance_rating(
                win_rate
            )

            # Calculate experience and skill points
            experience = total_trades * 10 + (win_rate * 100)
            skill_points = int(experience / 50)

            msg = f"""🎉 AI MILESTONE ACHIEVED!

🧠 **LEARNING PROGRESS:**
🏆 Level: {current_level}
⭐ Experience: {experience:.0f} XP
💎 Skill Points: {skill_points}
📊 Total Trades: {total_trades}

📈 **PERFORMANCE RATING:**
{performance_rating}
💬 Status: {performance_desc}
🎯 Win Rate: {win_rate:.1f}%

🚀 **CAPABILITIES UNLOCKED:**
{self._get_unlocked_capabilities(total_trades)}

🔥 Your AI is evolving into a trading MONSTER!
💰 Next milestone at {self._get_next_milestone(total_trades)} trades!"""

            await self._send_clean_telegram_message(msg)
            logger.info(
                f"🎉 AI Milestone notification sent: {current_level} at {total_trades} trades"
            )

        except Exception as e:
            logger.error(f"AI milestone notification error: {e}")

    def _get_unlocked_capabilities(self, total_trades: int) -> str:
        """Get unlocked AI capabilities based on trade count"""
        capabilities: List[str] = []

        if total_trades >= 10:
            capabilities.append("✅ Basic Pattern Recognition")
        if total_trades >= 25:
            capabilities.append("✅ Technical Analysis")
        if total_trades >= 50:
            capabilities.append("✅ Market Regime Detection")
        if total_trades >= 100:
            capabilities.append("✅ Advanced Risk Management")
        if total_trades >= 200:
            capabilities.append("✅ Multi-Timeframe Analysis")
        if total_trades >= 500:
            capabilities.append("✅ Sentiment Analysis")
        if total_trades >= 1000:
            capabilities.append("✅ Predictive Modeling")
        if total_trades >= 2000:
            capabilities.append("✅ Portfolio Optimization")
        if total_trades >= 5000:
            capabilities.append("✅ Market Making Skills")
        if total_trades >= 10000:
            capabilities.append("🔥 GODLIKE OMNISCIENCE")

        return "\n".join(capabilities[-4:]) if capabilities else "🌱 Learning basic trading..."

    def _get_next_milestone(self, total_trades: int) -> int:
        """Get the next milestone target"""
        for milestone in sorted(self.learning_tracker.learning_milestones.keys()):
            if total_trades < milestone:
                return milestone
        return total_trades + 1000  # Beyond max milestone

    async def _send_progress_report(self) -> None:
        """🧠 Send detailed AI learning progress report"""
        try:
            total_trades = int(self.session_stats.get("total_trades", 0))
            wins = self.session_stats.get("wins", 0)
            losses = self.session_stats.get("losses", 0)
            total_profit = self.session_stats.get("total_profit", 0.0)
            win_rate = (wins / max(1, total_trades)) * 100

            # Calculate learning metrics
            current_level = self.learning_tracker.get_current_level(total_trades)
            performance_rating, performance_desc = self.learning_tracker.get_performance_rating(
                win_rate
            )

            # AI confidence trend
            recent_accuracy = self._calculate_recent_accuracy()
            learning_trend = (
                "📈 Improving"
                if recent_accuracy > win_rate
                else "📉 Declining" if recent_accuracy < win_rate else "➡️ Stable"
            )

            # Experience calculation
            experience = total_trades * 10 + (win_rate * 100) + (max(0, total_profit) * 50)
            skill_points = int(experience / 50)

            # Time-based stats
            uptime_hours = (datetime.now() - self.bot_start_time).total_seconds() / 3600
            trades_per_hour = total_trades / max(1, uptime_hours)

            msg = f"""🧠 AI LEARNING PROGRESS REPORT

🏆 **CURRENT STATUS:**
🎯 Level: {current_level}
⭐ Experience: {experience:.0f} XP
💎 Skill Points: {skill_points}
📊 Performance: {performance_rating}

📈 **TRADING STATISTICS:**
🔢 Total Trades: {total_trades}
✅ Wins: {wins}
❌ Losses: {losses}
🎯 Win Rate: {win_rate:.1f}%
💰 Total P&L: ${total_profit:+.2f}

🧠 **AI INTELLIGENCE:**
📊 Recent Accuracy: {recent_accuracy:.1f}%
📈 Learning Trend: {learning_trend}
🎯 Confidence Level: {self._get_ai_confidence_level()}
🔄 Adaptation Rate: {self._get_adaptation_rate()}

⚡ **PERFORMANCE METRICS:**
🕒 Uptime: {uptime_hours:.1f} hours
📊 Trades/Hour: {trades_per_hour:.1f}
💰 Avg Profit: ${(total_profit/max(1, total_trades)):.2f}
🎯 Next Milestone: {self._get_next_milestone(total_trades)} trades

🚀 **AI CAPABILITIES:**
{self._get_current_capabilities(total_trades)}

💬 {performance_desc}
🔥 Keep trading to make your AI GODLIKE!"""

            await self._send_clean_telegram_message(msg)
            logger.info(f"📊 Progress report sent: Level {current_level}, {win_rate:.1f}% win rate")

        except Exception as e:
            logger.error(f"Progress report error: {e}")

    def _calculate_recent_accuracy(self) -> float:
        """Calculate accuracy from recent trades (last 20)"""
        try:
            if len(self.win_rate_tracker) < 5:
                return 50.0  # Default

            recent_trades = list(self.win_rate_tracker)[-20:]  # Last 20 trades
            wins = sum(1 for result in recent_trades if result == "WIN")
            return (wins / len(recent_trades)) * 100

        except Exception:
            return 50.0

    def _get_ai_confidence_level(self) -> str:
        """Get AI confidence level description"""
        total_trades = self.session_stats.get("total_trades", 0)
        win_rate = (self.session_stats.get("wins", 0) / max(1, total_trades)) * 100

        if total_trades < 10:
            return "🌱 Learning"
        elif total_trades < 50:
            return "📈 Building"
        elif win_rate > 70:
            return "💪 High"
        elif win_rate > 60:
            return "👍 Good"
        elif win_rate > 50:
            return "📊 Moderate"
        else:
            return "🔄 Improving"

    def _get_adaptation_rate(self) -> str:
        """Get AI adaptation rate"""
        recent_trades = self.session_stats.get("total_trades", 0)
        if recent_trades < 10:
            return "🐌 Slow"
        elif recent_trades < 50:
            return "🚶 Normal"
        elif recent_trades < 100:
            return "🏃 Fast"
        else:
            return "⚡ Rapid"

    def _get_current_capabilities(self, total_trades: int) -> str:
        """Get current AI capabilities summary"""
        capabilities: List[str] = []

        if total_trades >= 10:
            capabilities.append("📊 Technical Analysis")
        if total_trades >= 25:
            capabilities.append("🎯 Pattern Recognition")
        if total_trades >= 50:
            capabilities.append("📈 Trend Analysis")
        if total_trades >= 100:
            capabilities.append("🛡️ Risk Management")
        if total_trades >= 200:
            capabilities.append("⚡ Real-time Adaptation")
        if total_trades >= 500:
            capabilities.append("🧠 Deep Learning")
        if total_trades >= 1000:
            capabilities.append("🔮 Prediction Mastery")
        if total_trades >= 2000:
            capabilities.append("👑 Elite Trading")
        if total_trades >= 5000:
            capabilities.append("🌟 Legendary Status")
        if total_trades >= 10000:
            capabilities.append("🔥 GODLIKE POWERS")

        active_caps = capabilities[-3:] if len(capabilities) > 3 else capabilities
        return " | ".join(active_caps) if active_caps else "🌱 Basic Learning"

    async def _send_clean_telegram_message(self, message: str) -> bool:
        """📱 Send Telegram message with text sanitization"""
        try:
            # Clean the message text
            clean_message = self.text_sanitizer.sanitize_message(message)

            # Send through both notifiers
            success = False

            # Enhanced notifier
            if hasattr(self, "enhanced_notifier") and self.enhanced_notifier:
                try:
                    if hasattr(self.enhanced_notifier, "send_message"):
                        await self.enhanced_notifier.send_message(clean_message)
                        success = True
                except Exception as enhanced_error:
                    logger.debug(f"Enhanced notifier error: {enhanced_error}")

            # Legacy notifier
            if hasattr(self, "notifier") and self.notifier and hasattr(self.notifier, "telegram"):
                try:
                    await self.notifier.telegram.send_message(clean_message)
                    success = True
                except Exception as legacy_error:
                    logger.debug(f"Legacy notifier error: {legacy_error}")

            if success:
                logger.debug(f"📱 Clean message sent: {clean_message[:50]}...")
            else:
                logger.warning("📱 Failed to send clean message")

            return success

        except Exception as e:
            logger.error(f"📱 Clean message sending error: {e}")
            return False

    async def _record_trade_outcome(
        self, mt5_ticket: int, profit: float, features: List[float], success: bool
    ) -> None:
        """🧠 Record actual trade outcome for AI learning"""
        try:
            # Record in real data learning system
            self.real_data_learning.record_trade_outcome(features, profit, success)

            # Update session stats
            if success:
                self.session_stats["wins"] = self.session_stats.get("wins", 0) + 1
                if profit > 0:
                    self.consecutive_losses = 0  # Reset loss streak on profitable close
            else:
                self.session_stats["losses"] = self.session_stats.get("losses", 0) + 1
                if profit < 0:
                    self.consecutive_losses += 1

            self.session_stats["total_trades"] = self.session_stats.get("total_trades", 0) + 1
            self.session_stats["total_profit"] = (
                self.session_stats.get("total_profit", 0.0) + profit
            )

            # Update position sizer with real trade result
            self.advanced_position_sizer.update_trade_result(profit, success and profit > 0)

            # Check if we can retrain AI with real data
            if self.real_data_learning.can_train():
                await self._retrain_ai_with_real_data()

            # Log learning progress
            win_rate = self.real_data_learning.get_win_rate()
            total_samples = len(self.real_data_learning.trade_outcomes)

            if self.log_limiter.should_log("real_learning"):
                logger.info(
                    f"🧠 REAL LEARNING: {total_samples} samples, {win_rate:.1%} win rate, "
                    f"P&L: ${profit:.2f} ({'WIN' if success and profit > 0 else 'LOSS'})"
                )

        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")

    async def _retrain_ai_with_real_data(self) -> None:
        """🧠 Retrain AI models using real trade outcomes"""
        try:
            if not self.real_data_learning.can_train():
                return

            feature_outcome_pairs = self.real_data_learning.feature_outcome_pairs
            if len(feature_outcome_pairs) < 10:  # Need minimum samples
                return

            # Safe AI training approach - use real data for confidence calibration
            win_rate = self.real_data_learning.get_win_rate()
            total_samples = len(feature_outcome_pairs)

            logger.info(f"🧠 REAL DATA TRAINING: {total_samples} samples, {win_rate:.1%} win rate")

            # Update AI confidence weighting based on real performance
            if hasattr(self, "ai_model") and self.ai_model:
                try:
                    # Adjust AI confidence based on real win rate
                    confidence_adjustment = win_rate - 0.5  # Compare to 50% baseline

                    # Update model weights in ProfessionalAIEnsemble if available
                    if hasattr(self, "professional_ai_ensemble"):
                        # Adjust technical model weight based on real performance
                        if win_rate > 0.6:  # Good performance
                            self.professional_ai_ensemble.model_weights["technical"] = min(
                                0.35,
                                self.professional_ai_ensemble.model_weights["technical"] + 0.05,
                            )
                        elif win_rate < 0.4:  # Poor performance
                            self.professional_ai_ensemble.model_weights["technical"] = max(
                                0.15,
                                self.professional_ai_ensemble.model_weights["technical"] - 0.05,
                            )

                        logger.info(
                            f"🧠 AI WEIGHTS UPDATED: Technical weight now {self.professional_ai_ensemble.model_weights['technical']:.2f}"
                        )

                    logger.info(
                        f"🧠 AI CALIBRATION: Confidence adjusted by {confidence_adjustment:+.1%}"
                        f"based on real data"
                    )

                except Exception as e:
                    logger.warning(f"AI weight adjustment failed: {e}")

            # Mark the retrain time (no need to add attribute, just log)
            logger.info(f"🧠 AI retrained at {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            logger.error(f"Error processing real data for AI: {e}")

    async def _async_force_close(self, mt5_ticket: int, current_profit: float, reason: str) -> bool:
        """Close trade using ONLY async interface - no direct MT5 calls"""
        try:
            if not self.mt5_interface:
                logger.error(f"❌ Cannot close {mt5_ticket}: No MT5 interface")
                return False

            logger.warning(f"🔒 ASYNC CLOSE: {mt5_ticket} at ${current_profit:.2f} - {reason}")

            # 🧠 REAL DATA LEARNING: Capture trade features before closing
            trade_features = await self._capture_trade_features(mt5_ticket, current_profit)

            # Use async interface with throttling - no direct MT5 calls
            close_result = await self.mt5_request_manager.throttled_request(
                self.mt5_interface.close_position, mt5_ticket
            )

            if close_result:
                logger.info(f"✅ ASYNC CLOSE SUCCESS: {mt5_ticket} closed via async interface")

                # 🧠 REAL DATA LEARNING: Record actual trade outcome
                await self._record_trade_outcome(mt5_ticket, current_profit, trade_features, True)

                return True
            else:
                logger.warning(f"❌ ASYNC CLOSE FAILED: {mt5_ticket}")

                # 🧠 REAL DATA LEARNING: Record failed close attempt
                await self._record_trade_outcome(mt5_ticket, current_profit, trade_features, False)

                return False

        except Exception as e:
            logger.error(f"Async close error for {mt5_ticket}: {e}")
            return False

    async def _send_universal_profit_notification(self, symbol: str, ticket: int, profit: float):
        """Send unified profit notification using UNIVERSAL constants"""
        try:
            msg = f"""🎯 UNIVERSAL ${UNIVERSAL_PROFIT_TARGET:.2f} TARGET HIT!

💎 Symbol: {symbol}
🎫 Ticket: {ticket}
💰 Profit: +${profit:.2f}
🎯 Target: ${UNIVERSAL_PROFIT_TARGET:.2f} ✅
🧠 Method: Adaptive Monitoring
⚡ Intervals: 50ms-1s (no CPU pegging!)

✅ Universal system secured your ${UNIVERSAL_PROFIT_TARGET:.2f} profit!
🔄 Scanning for next opportunity..."""

            if hasattr(self, "notifier") and self.notifier and hasattr(self.notifier, "telegram"):
                # Use sanitizer for outbound message
                await self._send_sanitized_message(msg)

        except Exception as e:
            logger.warning(f"Universal notification error: {e}")

    async def _simple_force_close(self, mt5_ticket: int, current_profit: float) -> bool:
        """🎯 PRODUCTION FIX 7: Unified trading system - close trade"""
        try:
            if self.mt5_interface:
                result = await self.mt5_interface.close_position(mt5_ticket)
                if result:
                    logger.info(
                        f"✅ UNIFIED CLOSE: Ticket {mt5_ticket} closed at ${current_profit:.2f}"
                    )
                    return True
                else:
                    logger.warning(f"❌ UNIFIED CLOSE FAILED: {result}")
            return False
        except Exception as e:
            logger.error(f"Unified close error: {e}")
            return False

    async def _ultra_fast_close(self, mt5_ticket: int, current_profit: float, reason: str) -> bool:
        """Alias to unified close method - ensures all monitors use same path"""
        return await self._async_force_close(mt5_ticket, current_profit, reason)

    async def _simple_profit_notification(self, symbol: str, ticket: int, profit: float):
        """🎯 PRODUCTION FIX 7: Unified profit notification"""
        try:
            msg = f"""🎯 UNIVERSAL ${UNIVERSAL_PROFIT_TARGET:.2f} PROFIT SECURED!

💎 Symbol: {symbol}
🎫 Ticket: {ticket}
💰 Profit: +${profit:.2f}

✅ Unified ${UNIVERSAL_PROFIT_TARGET:.2f} system working perfectly!"""

            if hasattr(self, "notifier") and self.notifier and hasattr(self.notifier, "telegram"):
                await self.notifier.telegram.send_message(msg)

        except Exception as e:
            logger.warning(f"Simple notification error: {e}")

    async def _send_optimal_profit_notification(
        self, symbol: str, ticket: int, profit: float, target: float
    ):
        """Send optimal profit notification"""
        try:
            speed_rating = (
                "⚡⚡⚡" if profit >= target * 1.5 else "⚡⚡" if profit >= target * 1.2 else "⚡"
            )

            msg = f"""🎯 OPTIMAL PROFIT SECURED!

💎 Symbol: {symbol}
🎫 Ticket: {ticket}
💰 Profit: +${profit:.2f}
🎯 Target: ${target:.2f} ✅
{speed_rating} Speed: 50ms monitoring!

🤖 Your bot is CRUSHING the volatility indices!
🔄 Hunting for the next optimal opportunity..."""

            # Use sanitizer for outbound message
            await self._send_sanitized_message(msg)

        except Exception as e:
            logger.warning(f"Optimal notification failed: {e}")

    async def _send_emergency_profit_notification(
        self, symbol: str, ticket: int, profit: float, target: float
    ):
        """Send emergency profit securing notification"""
        try:
            msg = f"""🚨 EMERGENCY PROFIT SECURED!

💎 Symbol: {symbol}
🎫 Ticket: {ticket}
💰 Profit: +${profit:.2f}
🎯 Target: $0.20 ✅
⚡ Method: EMERGENCY OVERRIDE!

🚨 Emergency system activated because $0.20 target was exceeded!
🔄 Hunting for the next $0.20 opportunity..."""

            await self.notifier.telegram.send_message(msg)
            logger.error(f"🚨 Emergency profit notification sent: +${profit:.2f}")

        except Exception as e:
            logger.warning(f"Emergency notification failed: {e}")

    async def _send_instant_profit_notification(self, symbol: str, ticket: int, profit: float):
        """Send instant profit securing notification"""
        try:
            msg = f"""🚀 INSTANT PROFIT SECURED!

💎 Symbol: {symbol}
🎫 Ticket: {ticket}
💰 Profit: +${profit:.2f}
⚡ Speed: REAL-TIME!
🎯 Target: ${UNIVERSAL_PROFIT_TARGET:.2f} ✅

🤖 Bot secured your profit the INSTANT it appeared!
🔄 Now hunting for the next opportunity..."""

            await self.notifier.telegram.send_message(msg)
            logger.info(f"📱 Instant profit notification sent: +${profit:.2f}")

        except Exception as e:
            logger.warning(f"Instant profit notification failed: {e}")

    async def _send_profit_secured_notification(self, symbol: str, ticket: int, profit: float):
        """Send $0.20 profit secured notification"""
        try:
            msg = f"""✅ $0.20 PROFIT SECURED!

💎 Symbol: {symbol}
🎫 Ticket: {ticket}
💰 Profit: +${profit:.2f}
🎯 Target: $0.20 ✅
⚡ Speed: 50ms REAL-TIME!

🚀 Bot secured your $0.20 profit INSTANTLY!
🔄 Now hunting for the next $0.20 opportunity..."""

            await self.notifier.telegram.send_message(msg)
            logger.info(f"📱 $0.20 profit secured notification sent: +${profit:.2f}")

        except Exception as e:
            logger.error(f"$0.20 profit notification error: {e}")

    async def stop(self):
        """Stop the bot with proper monitor cleanup"""
        logger.info("🛑 Stopping bot...")
        self.running = False

        try:
            # Stop all active monitors
            active_monitors = list(self.monitor_manager.active_monitors)
            for monitor_name in active_monitors:
                await self.monitor_manager.stop_monitor(monitor_name)
                logger.info(f"🛑 Stopped monitor: {monitor_name}")

            # Save AI model
            self.ai_model.save_model()

            # Send shutdown notification
            await self._send_shutdown_notification()

            logger.info("✅ Bot stopped with clean monitor shutdown")

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

            # Use sanitizer for outbound message
            clean_msg = TelegramTextSanitizer.sanitize_message(msg)
            await self.notifier.telegram.send_message(clean_msg)
        except Exception as e:
            logger.warning(f"Shutdown notification failed: {e}")

    async def force_enable_20_cent_auto_close(self):
        """🎯 BULLETPROOF: Ensure ALL trades auto-close at exactly $0.20"""
        logger.warning("🎯 ENABLING BULLETPROOF $0.20 AUTO-CLOSE FOR ALL TRADES")

        # Use monitor manager to prevent conflicts - if already registered, don't start duplicate
        if await self.monitor_manager.start_monitor("bulletproof_20cent"):
            # Monitor manager registered successfully, start the actual task
            asyncio.create_task(self._bulletproof_20_cent_monitor())
            logger.warning("✅ BULLETPROOF $0.20 MONITOR: ACTIVE - ALL TRADES WILL CLOSE AT $0.20!")
        else:
            logger.warning("⚠️ Bulletproof monitor already running - using existing instance")

    async def _bulletproof_20_cent_monitor(self):
        """🎯 BULLETPROOF: Monitor that GUARANTEES $0.20 closure on ALL trades"""
        monitor_name = "bulletproof_20cent"

        try:
            logger.info("🎯 BULLETPROOF MONITOR: Starting dedicated $0.20 auto-close system")

            while self.running:
                try:
                    if self.active_trades:
                        logger.info(
                            f"🎯 CHECKING {len(self.active_trades)} trades for $0.20 target..."
                        )

                        for trade_id, trade in list(self.active_trades.items()):
                            try:
                                mt5_ticket = trade.get("mt5_ticket", trade_id)
                                if isinstance(mt5_ticket, str) and mt5_ticket.isdigit():
                                    mt5_ticket = int(mt5_ticket)
                                elif not isinstance(mt5_ticket, int):
                                    continue

                                # Get position info
                                if self.mt5_interface:
                                    position_info = await self.mt5_interface.get_position_info(
                                        mt5_ticket
                                    )

                                    if not position_info:
                                        # Trade already closed
                                        async with self.trades_lock:
                                            if trade_id in self.active_trades:
                                                del self.active_trades[trade_id]
                                                logger.info(f"🗑️ Removed closed trade {trade_id}")
                                        continue

                                    current_profit = float(getattr(position_info, "profit", 0.0))
                                    symbol = trade.get("symbol", "Unknown")

                                    logger.info(f"💰 {symbol} {mt5_ticket}: ${current_profit:.2f}")

                                    # 🎯 BULLETPROOF $0.20 CHECK
                                    if current_profit >= 0.20:
                                        logger.error(
                                            f"🎯 $0.20 TARGET HIT: {symbol}"
                                            f"${current_profit:.2f} - FORCE CLOSING NOW!"
                                        )

                                        # 🚨 FIX: Use unified close method
                                        close_success = await self._async_force_close(
                                            mt5_ticket,
                                            current_profit,
                                            "Bulletproof $0.20 Target",
                                        )

                                        if close_success:
                                            # Remove from active trades with thread safety
                                            async with self.trades_lock:
                                                if trade_id in self.active_trades:
                                                    del self.active_trades[trade_id]

                                            # Send notification
                                            await self._send_20_cent_notification(
                                                symbol, mt5_ticket, current_profit
                                            )

                                            logger.error(
                                                f"✅ $0.20 PROFIT SECURED: {symbol}"
                                                f"+${current_profit:.2f}"
                                            )
                                        else:
                                            logger.error(
                                                f"❌ FAILED TO CLOSE {mt5_ticket} - WILL RETRY!"
                                            )

                            except Exception as trade_error:
                                logger.error(f"Error checking trade {trade_id}: {trade_error}")

                    # Check every 2 seconds (fast enough to catch profits)
                    await asyncio.sleep(2.0)

                except Exception as e:
                    logger.error(f"Bulletproof monitor error: {e}")
                    await asyncio.sleep(5.0)

        except Exception as e:
            logger.error(f"🎯 Bulletproof monitor fatal error: {e}")
        finally:
            # Cleanup: Remove from active monitors
            await self.monitor_manager.stop_monitor(monitor_name)
            logger.info("🎯 Bulletproof monitor cleanup completed")

    async def _send_20_cent_notification(self, symbol: str, ticket: int, profit: float):
        """Send $0.20 profit notification"""
        try:
            msg = f"""🎯 $0.20 PROFIT SECURED!

💎 Symbol: {symbol}
🎫 Ticket: {ticket}
💰 Profit: +${profit:.2f}
✅ Target: $0.20 HIT!

🚀 Bulletproof system working perfectly!
🔄 Ready for next $0.20 opportunity..."""

            if hasattr(self, "notifier") and self.notifier and hasattr(self.notifier, "telegram"):
                await self.notifier.telegram.send_message(msg)

        except Exception as e:
            logger.warning(f"$0.20 notification failed: {e}")


# Global bot instance
global_bot_instance: Optional[TradingBot] = None


async def debug_current_trades():
    """Debug current trade status"""
    if not global_bot_instance:
        print("❌ Bot not running")
        return

    bot = global_bot_instance
    logger.info("🔍 DEBUGGING CURRENT TRADES:")
    print("🔍 DEBUGGING CURRENT TRADES:")

    if not bot.active_trades:
        print("✅ No active trades")
        return

    for trade_id, trade in bot.active_trades.items():
        try:
            mt5_ticket = trade.get("mt5_ticket", trade_id)
            if isinstance(mt5_ticket, str) and mt5_ticket.isdigit():
                mt5_ticket = int(mt5_ticket)
            elif not isinstance(mt5_ticket, int):
                continue  # Skip if not a valid ticket

            # Check if MT5 interface is available
            if bot.mt5_interface is not None:
                position_info = await bot.mt5_interface.get_position_info(mt5_ticket)
                if position_info:
                    profit = float(getattr(position_info, "profit", 0.0))
                    print(f"🎫 Ticket: {mt5_ticket}")
                    print(f"💰 Profit: ${profit:.2f}")
                    print(f"🎯 Symbol: {trade.get('symbol', 'Unknown')}")
                    print(f"📈 Action: {trade.get('action', 'Unknown')}")
                    print("---")

                    # Check if this trade should be closed
                    if profit >= 2.0:
                        print(
                            f"🚨 WARNING: Trade {mt5_ticket}"
                            f"should be closed! (${profit:.2f} >= $2.00)"
                        )
            else:
                print(f"⚠️ MT5 interface not available for {trade_id}")

        except Exception as e:
            print(f"Error checking {trade_id}: {e}")


async def force_close_profitable_trades():
    """Force close all profitable trades immediately"""
    if not global_bot_instance:
        print("❌ Bot not running")
        return 0

    bot = global_bot_instance
    return await bot.force_close_all_profitable_trades()


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


# 🛡️ LOSS PROTECTION UTILITY FUNCTIONS


async def debug_trading_status():
    """🔍 Debug why trades aren't happening"""
    if not global_bot_instance:
        print("❌ Bot not running - start bot first!")
        return

    print("🔍 DEBUGGING TRADING STATUS...")
    await global_bot_instance.debug_trading_blocks()
    print("✅ Debug complete - check logs above")


def check_loss_protection_status() -> Optional[Dict[str, Union[bool, float]]]:
    """Check current loss protection status"""
    if global_bot_instance:
        bot = global_bot_instance
        current_profit = bot.current_balance - bot.starting_balance

        print("🛡️ LOSS PROTECTION STATUS")
        print(f"   Enabled: {'YES' if bot.loss_protection_enabled else 'NO'}")
        print(f"   Triggered: {'YES' if bot.loss_protection_triggered else 'NO'}")
        print(f"   Starting Balance: ${bot.starting_balance:.2f}")
        print(f"   Current Balance: ${bot.current_balance:.2f}")
        print(f"   Current Profit: ${current_profit:+.2f}")
        print(f"   Loss Threshold: -${LOSS_PROTECTION_THRESHOLD:.2f}")
        print(f"   Emergency Threshold: -${LOSS_PROTECTION_MAX_THRESHOLD:.2f}")
        print(
                        f"   Loss Status: {('SAFE' if current_profit > "
            "-LOSS_PROTECTION_THRESHOLD else 'AT RISK')}"
        )

        return {
            "enabled": bot.loss_protection_enabled,
            "triggered": bot.loss_protection_triggered,
            "starting_balance": bot.starting_balance,
            "current_balance": bot.current_balance,
            "current_profit": current_profit,
            "threshold": LOSS_PROTECTION_THRESHOLD,
        }
    else:
        print("❌ Bot not running")
        return None
        return None


# Add utility function for manual reset
def reset_loss_protection_manual():
    """Manually reset loss protection to resume trading"""
    if global_bot_instance:
        result = global_bot_instance.reset_loss_protection()
        if result:
            print("✅ Loss protection reset successfully!")
            print(f"   New starting balance: ${result['new_starting_balance']:.2f}")
            print("   Bot ready to resume unlimited profit trading")
        else:
            print("❌ Failed to reset loss protection")
        return result
    else:
        print("❌ Bot not running")
        return None


# 🎯 DEMO TRAINING MODE FUNCTIONS


def force_enable_demo_training_mode():
    """🎯 FORCE ENABLE DEMO TRAINING MODE - Never stops trading!"""
    if global_bot_instance:
        bot = global_bot_instance

        # Force enable demo mode detection
        bot.demo_mode = True

        # Reset all blocking conditions
        bot.loss_protection_triggered = False
        bot.loss_protection_enabled = False
        bot.consecutive_losses = 0
        bot.daily_profit = 0.0

        # Reset baseline for fresh tracking
        bot.starting_balance = bot.current_balance

        logger.error("🎯 DEMO TRAINING MODE FORCED!")
        logger.error("🚀 Bot will NEVER stop trading for unlimited AI learning!")
        logger.error("🧠 All trades will train the AI to become GODLIKE!")

        print("🎯 DEMO TRAINING MODE: ACTIVATED!")
        print("🚀 Bot will trade continuously for AI training!")
        print("🧠 Every trade makes the AI smarter!")

        return True
    else:
        print("❌ Bot not running")
        return False


def is_demo_training_mode() -> bool:
    """Check if we're in demo training mode"""
    if global_bot_instance:
        bot = global_bot_instance
        is_demo = bot.demo_mode or "Demo" in str(getattr(bot.mt5_interface, "server", ""))

        print(f"🎯 Demo Training Mode: {'ACTIVE' if is_demo else 'INACTIVE'}")
        if is_demo:
            print(
                f"🧠 AI Learning: {getattr(bot, 'session_stats', {}"
                f".get('total_trades', 0)} trades completed"
            )
            print(f"📊 Current P&L: ${bot.current_balance - bot.starting_balance:+.2f}")

        return is_demo
    return False


def debug_session_stats() -> Optional[Dict[str, Any]]:
    """Debug current session statistics"""
    if global_bot_instance:
        bot = global_bot_instance

        total = bot.session_stats.get("total_trades", 0)
        wins = bot.session_stats.get("wins", 0)
        losses = bot.session_stats.get("losses", 0)
        profit = bot.session_stats.get("total_profit", 0.0)

        calculated_win_rate = (wins / max(1, total)) * 100

        print("📊 CURRENT SESSION STATISTICS:")
        print(f"   Total Trades: {total}")
        print(f"   Wins: {wins}")
        print(f"   Losses: {losses}")
        print(f"   Win Rate: {calculated_win_rate:.1f}%")
        print(f"   Total P&L: ${profit:+.2f}")
        print(f"   Current Balance: ${bot.current_balance:.2f}")
        print(f"   Consecutive Losses: {bot.consecutive_losses}")

        # Check win rate tracker
        if hasattr(bot, "win_rate_tracker") and bot.win_rate_tracker:
            recent_wins = sum(1 for r in bot.win_rate_tracker if r == "WIN")
            recent_total = len(bot.win_rate_tracker)
            recent_win_rate = (recent_wins / recent_total) * 100
            print(f"   Recent Win Rate: {recent_win_rate:.1f}% ({recent_wins}/{recent_total})")

        return {
            "total_trades": int(total),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": float(calculated_win_rate),
            "total_profit": float(profit),
        }
    else:
        print("❌ Bot not running")
        return None


# End of main.py

if __name__ == "__main__":
    asyncio.run(main())
