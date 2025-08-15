"""
Technical Indicators Module for Deriv Trading Bot
Contains all technical analysis indicators and signal generation logic
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ta

from config import *

# Fallback for PRICE_BUFFER_SIZE if not defined in config
try:
    PRICE_BUFFER_SIZE
except NameError:
    PRICE_BUFFER_SIZE = 100


class TechnicalIndicators:
    """Class containing all technical indicator calculations"""

    def __init__(self):
        self.price_data: List[Dict[str, Any]] = []
        self.df = pd.DataFrame()

    def update_price_data(self, price: float, timestamp: Optional[str] = None) -> None:
        """Update price data with new tick"""
        self.price_data.append(
            {
                "timestamp": timestamp or pd.Timestamp.now(),
                "close": price,
                "price": price,
            }
        )

        # Keep only last N prices for efficiency
        if len(self.price_data) > PRICE_BUFFER_SIZE * 2:
            self.price_data = self.price_data[-PRICE_BUFFER_SIZE:]

        # Update DataFrame
        if len(self.price_data) >= 20:  # Minimum data points for indicators
            self.df = pd.DataFrame(self.price_data)
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.df.set_index("timestamp", inplace=True)

    def calculate_rsi(self, period: int = RSI_PERIOD) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(self.df) < period + 1:
            return None

        try:
            rsi_indicator = ta.momentum.RSIIndicator(
                close=self.df["close"], window=period
            )
            rsi_values = rsi_indicator.rsi()
            return rsi_values.iloc[-1] if not rsi_values.empty else None
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return None

    def calculate_ema(self, period: int = EMA_FAST) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(self.df) < period:
            return None

        try:
            ema_indicator = ta.trend.EMAIndicator(close=self.df["close"], window=period)
            ema_values = ema_indicator.ema_indicator()
            return ema_values.iloc[-1] if not ema_values.empty else None
        except Exception as e:
            print(f"Error calculating EMA: {e}")
            return None

    def calculate_sma(self, period: int = MA_SHORT) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(self.df) < period:
            return None

        try:
            sma_indicator = ta.trend.SMAIndicator(close=self.df["close"], window=period)
            sma_values = sma_indicator.sma_indicator()
            return sma_values.iloc[-1] if not sma_values.empty else None
        except Exception as e:
            print(f"Error calculating SMA: {e}")
            return None

    def calculate_macd(self) -> Dict[str, Optional[float]]:
        """Calculate MACD indicator"""
        if len(self.df) < EMA_SLOW + EMA_SIGNAL:
            return {"macd": None, "signal": None, "histogram": None}

        try:
            macd_indicator = ta.trend.MACD(
                close=self.df["close"],
                window_fast=EMA_FAST,
                window_slow=EMA_SLOW,
                window_sign=EMA_SIGNAL,
            )

            macd_line = macd_indicator.macd()
            signal_line = macd_indicator.macd_signal()
            histogram = macd_indicator.macd_diff()

            return {
                "macd": macd_line.iloc[-1] if not macd_line.empty else None,
                "signal": signal_line.iloc[-1] if not signal_line.empty else None,
                "histogram": histogram.iloc[-1] if not histogram.empty else None,
            }
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return {"macd": None, "signal": None, "histogram": None}

    def calculate_bollinger_bands(self) -> Dict[str, Optional[float]]:
        """Calculate Bollinger Bands"""
        if len(self.df) < BOLLINGER_PERIOD:
            return {"upper": None, "middle": None, "lower": None}

        try:
            bb_indicator = ta.volatility.BollingerBands(
                close=self.df["close"],
                window=BOLLINGER_PERIOD,
                window_dev=BOLLINGER_STD,
            )

            upper = bb_indicator.bollinger_hband()
            middle = bb_indicator.bollinger_mavg()
            lower = bb_indicator.bollinger_lband()

            return {
                "upper": upper.iloc[-1] if not upper.empty else None,
                "middle": middle.iloc[-1] if not middle.empty else None,
                "lower": lower.iloc[-1] if not lower.empty else None,
            }
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            return {"upper": None, "middle": None, "lower": None}

    def calculate_stochastic(
        self, k_period: int = 14, d_period: int = 3
    ) -> Dict[str, Optional[float]]:
        """Calculate Stochastic Oscillator"""
        if len(self.df) < k_period + d_period:
            return {"%K": None, "%D": None}

        try:
            # For synthetic indices, we use close price as high/low approximation
            high = self.df["close"].rolling(window=3).max()
            low = self.df["close"].rolling(window=3).min()

            stoch_indicator = ta.momentum.StochasticOscillator(
                high=high,
                low=low,
                close=self.df["close"],
                window=k_period,
                smooth_window=d_period,
            )

            k_values = stoch_indicator.stoch()
            d_values = stoch_indicator.stoch_signal()

            return {
                "%K": k_values.iloc[-1] if not k_values.empty else None,
                "%D": d_values.iloc[-1] if not d_values.empty else None,
            }
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")
            return {"%K": None, "%D": None}

    def detect_price_patterns(self) -> Dict[str, bool]:
        """Detect common price patterns"""
        if len(self.df) < 10:
            return {}

        patterns = {}
        prices = self.df["close"].values[-10:]  # Last 10 prices

        try:
            # Trend detection
            if len(prices) >= 5:
                recent_trend = np.polyfit(range(5), prices[-5:], 1)[0]
                patterns["uptrend"] = recent_trend > 0
                patterns["downtrend"] = recent_trend < 0

            # Support/Resistance levels
            current_price = prices[-1]
            recent_high = np.max(prices[-5:])
            recent_low = np.min(prices[-5:])

            patterns["near_resistance"] = current_price >= recent_high * 0.99
            patterns["near_support"] = current_price <= recent_low * 1.01

            # Breakout detection
            price_range = recent_high - recent_low
            patterns["breakout_up"] = current_price > recent_high
            patterns["breakout_down"] = current_price < recent_low

            # Consolidation
            patterns["consolidating"] = price_range < (
                current_price * 0.01
            )  # Less than 1% range

        except Exception as e:
            print(f"Error detecting patterns: {e}")

        return patterns

    def generate_signals(self) -> Dict[str, str]:
        """Generate trading signals based on all indicators"""
        signals = {
            "rsi_signal": "HOLD",
            "ema_signal": "HOLD",
            "macd_signal": "HOLD",
            "bb_signal": "HOLD",
            "stoch_signal": "HOLD",
            "pattern_signal": "HOLD",
            "overall_signal": "HOLD",
        }

        if len(self.df) < 30:  # Need enough data
            return signals

        try:
            current_price = self.df["close"].iloc[-1]

            # RSI Signal
            rsi = self.calculate_rsi()
            if rsi:
                if rsi < RSI_OVERSOLD:
                    signals["rsi_signal"] = "BUY"
                elif rsi > RSI_OVERBOUGHT:
                    signals["rsi_signal"] = "SELL"

            # EMA Crossover Signal
            ema_fast = self.calculate_ema(EMA_FAST)
            ema_slow = self.calculate_ema(EMA_SLOW)
            if ema_fast and ema_slow:
                if ema_fast > ema_slow and current_price > ema_fast:
                    signals["ema_signal"] = "BUY"
                elif ema_fast < ema_slow and current_price < ema_fast:
                    signals["ema_signal"] = "SELL"

            # MACD Signal
            macd_data = self.calculate_macd()
            if all(macd_data.values()):
                if (
                    macd_data["macd"] > macd_data["signal"]
                    and macd_data["histogram"] > 0
                ):
                    signals["macd_signal"] = "BUY"
                elif (
                    macd_data["macd"] < macd_data["signal"]
                    and macd_data["histogram"] < 0
                ):
                    signals["macd_signal"] = "SELL"

            # Bollinger Bands Signal
            bb_data = self.calculate_bollinger_bands()
            if all(bb_data.values()):
                if current_price <= bb_data["lower"]:
                    signals["bb_signal"] = "BUY"
                elif current_price >= bb_data["upper"]:
                    signals["bb_signal"] = "SELL"

            # Stochastic Signal
            stoch_data = self.calculate_stochastic()
            if all(stoch_data.values()):
                if stoch_data["%K"] < 20 and stoch_data["%D"] < 20:
                    signals["stoch_signal"] = "BUY"
                elif stoch_data["%K"] > 80 and stoch_data["%D"] > 80:
                    signals["stoch_signal"] = "SELL"

            # Pattern Signal
            patterns = self.detect_price_patterns()
            if patterns.get("breakout_up") and patterns.get("uptrend"):
                signals["pattern_signal"] = "BUY"
            elif patterns.get("breakout_down") and patterns.get("downtrend"):
                signals["pattern_signal"] = "SELL"

            # Overall Signal (majority vote)
            buy_signals = sum(1 for signal in signals.values() if signal == "BUY")
            sell_signals = sum(1 for signal in signals.values() if signal == "SELL")

            if buy_signals >= 3:  # At least 3 indicators agree
                signals["overall_signal"] = "BUY"
            elif sell_signals >= 3:
                signals["overall_signal"] = "SELL"

        except Exception as e:
            print(f"Error generating signals: {e}")

        return signals

    def get_market_strength(self) -> float:
        """Calculate market strength/volatility"""
        if len(self.df) < 20:
            return 0.5  # Neutral

        try:
            # Calculate price volatility
            returns = self.df["close"].pct_change().dropna()
            volatility = returns.std()

            # Normalize to 0-1 scale
            strength = min(1.0, max(0.0, volatility * 100))
            return strength

        except Exception as e:
            print(f"Error calculating market strength: {e}")
            return 0.5

    def get_indicator_summary(self) -> Dict:
        """Get a summary of all current indicators"""
        summary = {
            "price": self.df["close"].iloc[-1] if not self.df.empty else None,
            "rsi": self.calculate_rsi(),
            "ema_fast": self.calculate_ema(EMA_FAST),
            "ema_slow": self.calculate_ema(EMA_SLOW),
            "macd": self.calculate_macd(),
            "bollinger": self.calculate_bollinger_bands(),
            "stochastic": self.calculate_stochastic(),
            "patterns": self.detect_price_patterns(),
            "signals": self.generate_signals(),
            "market_strength": self.get_market_strength(),
            "data_points": len(self.df),
        }

        return summary


# Utility functions
def calculate_simple_signals(prices: List[float]) -> str:
    """Simple signal calculation for quick decisions"""
    if len(prices) < 20:
        return "HOLD"

    recent_prices = prices[-10:]
    avg_price = sum(recent_prices) / len(recent_prices)
    current_price = prices[-1]

    # Simple trend following
    if current_price > avg_price * 1.001:  # 0.1% above average
        return "BUY"
    elif current_price < avg_price * 0.999:  # 0.1% below average
        return "SELL"
    else:
        return "HOLD"


def validate_signal_strength(signals: Dict[str, str]) -> float:
    """Calculate signal strength based on indicator agreement"""
    if not signals:
        return 0.0

    total_signals = len([s for s in signals.values() if s in ["BUY", "SELL"]])
    if total_signals == 0:
        return 0.0

    # Count agreeing signals
    buy_count = sum(1 for s in signals.values() if s == "BUY")
    sell_count = sum(1 for s in signals.values() if s == "SELL")

    max_agreement = max(buy_count, sell_count)
    strength = max_agreement / len(signals)

    return strength


if __name__ == "__main__":
    # Test the indicators
    print("🔧 Testing Technical Indicators...")

    indicators = TechnicalIndicators()

    # Simulate some price data
    import random

    base_price = 100.0
    for i in range(50):
        price = base_price + random.uniform(-2, 2)
        indicators.update_price_data(price)
        base_price = price

    summary = indicators.get_indicator_summary()
    print(f"✅ Test completed. Data points: {summary['data_points']}")
    print(f"📊 Current signals: {summary['signals']['overall_signal']}")
    print(f"📈 Market strength: {summary['market_strength']:.2f}")
