"""
AI Model Module for Deriv Trading Bot
Contains machine learning models and prediction logic
"""

import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Safe config import with fallback
try:
    import config
except ImportError:
    config = None


def _as01(x, default=0.35):
    """Convert confidence threshold to 0..1 scale safely"""
    if x is None:
        return default
    return (x / 100.0) if x > 1 else float(x)


CONFIDENCE_THRESHOLD_01 = _as01(
    getattr(config, "MIN_CONFIDENCE", None) if config else None,
    default=_as01(getattr(config, "AI_CONFIDENCE_THRESHOLD", 0.35) if config else 0.35),
)

# RSI constants with fallbacks
RSI_OVERSOLD = getattr(config, "RSI_OVERSOLD", 30) if config else 30
RSI_OVERBOUGHT = getattr(config, "RSI_OVERBOUGHT", 70) if config else 70


class TradingAI:
    """AI Trading Decision Engine"""

    def __init__(self):
        # Use safe config loading with fallbacks
        self.model_type = (
            getattr(config, "AI_MODEL_TYPE", "randomforest")
            if config
            else "randomforest"
        )
        self.confidence_threshold = CONFIDENCE_THRESHOLD_01  # Use 0..1 scale everywhere
        self.prediction_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "last_updated": datetime.now(),
        }
        self.feature_weights = {
            "rsi": 0.2,
            "ema_trend": 0.25,
            "macd": 0.2,
            "bollinger": 0.15,
            "pattern": 0.2,
        }

    def extract_features(self, indicator_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from technical indicators for ML prediction"""
        features = {}

        try:
            # RSI features
            rsi = indicator_data.get("rsi")
            if rsi:
                features["rsi_normalized"] = (rsi - 50) / 50  # Normalize around 0
                features["rsi_oversold"] = 1.0 if rsi < RSI_OVERSOLD else 0.0
                features["rsi_overbought"] = 1.0 if rsi > RSI_OVERBOUGHT else 0.0

            # EMA trend features
            ema_fast = indicator_data.get("ema_fast")
            ema_slow = indicator_data.get("ema_slow")
            current_price = indicator_data.get("price")

            if all(
                [ema_fast is not None, ema_slow is not None, current_price is not None]
            ):
                features["ema_trend"] = 1.0 if float(ema_fast) > float(ema_slow) else -1.0  # type: ignore
                features["price_above_ema"] = 1.0 if float(current_price) > float(ema_fast) else 0.0  # type: ignore
                features["ema_divergence"] = (float(ema_fast) - float(ema_slow)) / float(current_price)  # type: ignore

            # MACD features
            macd_line = indicator_data.get("macd")
            signal_line = indicator_data.get("macd_signal")
            histogram = indicator_data.get("macd_histogram")

            if all(x is not None for x in [macd_line, signal_line, histogram]):
                features["macd_signal"] = 1.0 if float(macd_line) > float(signal_line) else -1.0  # type: ignore
                hist_val = float(histogram)  # type: ignore
                features["macd_histogram"] = (
                    hist_val / abs(hist_val) if hist_val != 0 else 0.0
                )
                features["macd_strength"] = abs(hist_val)

            # Bollinger Bands features (if available)
            bb_upper = indicator_data.get("bb_upper")
            bb_middle = indicator_data.get("bb_middle")
            bb_lower = indicator_data.get("bb_lower")

            if all(
                x is not None for x in [bb_upper, bb_middle, bb_lower, current_price]
            ):
                bb_width = float(bb_upper) - float(bb_lower)  # type: ignore
                features["bb_position"] = (float(current_price) - float(bb_lower)) / bb_width if bb_width > 0 else 0.5  # type: ignore
                features["bb_squeeze"] = 1.0 if bb_width < (float(bb_middle) * 0.02) else 0.0  # type: ignore
                features["bb_breakout"] = 1.0 if float(current_price) > float(bb_upper) or float(current_price) < float(bb_lower) else 0.0  # type: ignore

            # Pattern features
            patterns = indicator_data.get("patterns", {})
            features["trend_up"] = 1.0 if patterns.get("uptrend") else 0.0
            features["trend_down"] = 1.0 if patterns.get("downtrend") else 0.0
            features["near_support"] = 1.0 if patterns.get("near_support") else 0.0
            features["near_resistance"] = (
                1.0 if patterns.get("near_resistance") else 0.0
            )
            features["breakout"] = (
                1.0
                if patterns.get("breakout_up") or patterns.get("breakout_down")
                else 0.0
            )

            # Market strength
            features["market_strength"] = indicator_data.get("market_strength", 0.5)

            # Time-based features (optional)
            now = datetime.now()
            features["hour_of_day"] = now.hour / 24.0
            features["day_of_week"] = now.weekday() / 7.0

        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default features
            features = dict.fromkeys(self.feature_weights.keys(), 0.0)

        return features

    def simple_ensemble_predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Ultra-aggressive ensemble prediction based on weighted features"""
        try:
            # Calculate weighted scores for buy/sell
            buy_score = 0.0
            sell_score = 0.0

            # ULTRA-AGGRESSIVE RSI signals (48-52 range)
            rsi = features.get("rsi", 50)
            if rsi < 52:  # More sensitive than 30!
                buy_score += self.feature_weights["rsi"] * 2  # Double weight
            if rsi > 48:  # More sensitive than 70!
                sell_score += self.feature_weights["rsi"] * 2  # Double weight

            # AGGRESSIVE EMA trend signals - any trend counts
            if "ema_trend" in features:
                if features["ema_trend"] > 0:  # Any upward momentum
                    buy_score += self.feature_weights["ema_trend"] * 1.5
                elif features["ema_trend"] < 0:  # Any downward momentum
                    sell_score += self.feature_weights["ema_trend"] * 1.5

            # AGGRESSIVE MACD signals - any signal counts
            if "macd_signal" in features:
                macd_strength = (
                    abs(features.get("macd_histogram", 0)) * 10
                )  # 10x more sensitive
                if features["macd_signal"] > 0:
                    buy_score += self.feature_weights["macd"] * (1 + macd_strength)
                else:
                    sell_score += self.feature_weights["macd"] * (1 + macd_strength)

            # ULTRA-AGGRESSIVE Bollinger Bands signals (40-60 range instead of 20-80)
            bb_pos = features.get("bb_position", 0.5)
            if bb_pos < 0.6:  # More sensitive - was 0.2
                buy_score += self.feature_weights["bollinger"] * 1.5
            elif bb_pos > 0.4:  # More sensitive - was 0.8
                sell_score += self.feature_weights["bollinger"] * 1.5

            # AGGRESSIVE Pattern signals - any hint of movement
            if features.get("trend_up", 0) > 0.2:  # Lower threshold
                buy_score += self.feature_weights["pattern"] * 1.5
            elif features.get("trend_down", 0) > 0.2:  # Lower threshold
                sell_score += self.feature_weights["pattern"] * 1.5

            # FORCE TRADING: Add base score if no signals
            if buy_score == 0 and sell_score == 0:
                # Force trade based on RSI
                if rsi >= 50:
                    sell_score = 0.3  # Minimum trade score
                else:
                    buy_score = 0.3  # Minimum trade score

            # Determine prediction and confidence
            max_score = max(buy_score, sell_score)
            total_possible = (
                sum(self.feature_weights.values()) * 2
            )  # Account for double weights
            confidence = min(1.0, max_score / total_possible)

            # AGGRESSIVE: Never return confidence below 0.22
            confidence = max(confidence, 0.22)

            # FORCE TRADING: Lower confidence threshold
            ultra_low_threshold = 0.15  # Much lower than normal 0.4

            if buy_score > sell_score and confidence >= ultra_low_threshold:
                return "BUY", confidence
            elif sell_score > buy_score and confidence >= ultra_low_threshold:
                return "SELL", confidence
            else:
                # FORCE: Never return HOLD - pick direction based on RSI
                if rsi >= 50:
                    return "SELL", 0.22  # Force minimum confidence
                else:
                    return "BUY", 0.22  # Force minimum confidence

        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            # NEVER return 0.0 confidence - force a trade
            return "BUY", 0.22

    def advanced_ml_predict(
        self, features: Dict[str, float], price_history: List[float]
    ) -> Tuple[str, float]:
        """Advanced ML prediction (placeholder for future LSTM/Neural Network)"""
        # For now, use a more sophisticated rule-based approach
        try:
            if len(price_history) < 10:
                return "HOLD", 0.0

            # Calculate price momentum
            recent_prices = price_history[-10:]
            momentum = np.mean(np.diff(recent_prices))
            volatility = np.std(recent_prices) / np.mean(recent_prices)

            # Combine technical and momentum features
            signal_strength = 0.0
            prediction = "HOLD"

            # Momentum factor
            momentum_factor = momentum / abs(momentum) if momentum != 0 else 0

            # Technical factor from ensemble
            tech_prediction, tech_confidence = self.simple_ensemble_predict(features)
            tech_factor = (
                1
                if tech_prediction == "BUY"
                else -1 if tech_prediction == "SELL" else 0
            )

            # Volatility adjustment
            vol_factor = min(
                1.0, float(volatility * 100)
            )  # Higher volatility = higher risk

            # Combined score
            combined_score = (tech_factor * tech_confidence * 0.7) + (
                momentum_factor * 0.3
            )
            adjusted_confidence = abs(combined_score) * (
                1 - vol_factor * 0.3
            )  # Reduce confidence in high volatility

            if (
                combined_score > 0.3
                and adjusted_confidence >= self.confidence_threshold
            ):
                prediction = "BUY"
                signal_strength = adjusted_confidence
            elif (
                combined_score < -0.3
                and adjusted_confidence >= self.confidence_threshold
            ):
                prediction = "SELL"
                signal_strength = adjusted_confidence
            else:
                prediction = "HOLD"
                signal_strength = adjusted_confidence

            return prediction, float(signal_strength)

        except Exception as e:
            print(f"Error in advanced ML prediction: {e}")
            return self.simple_ensemble_predict(features)

    def predict_trade(
        self,
        indicator_data: Dict[str, Any],
        price_history: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Main prediction function with aggressive fallback"""
        if not indicator_data:
            # AGGRESSIVE FALLBACK: Generate a trade anyway based on price history
            if price_history and len(price_history) >= 2:
                recent_change = (
                    (price_history[-1] - price_history[-2]) / price_history[-2] * 100
                )
                if recent_change > 0.001:  # Tiny upward movement
                    return {
                        "prediction": "BUY",
                        "confidence": 0.25,  # Above threshold!
                        "reason": "Aggressive Fallback: Tiny upward momentum detected",
                    }
                else:
                    return {
                        "prediction": "SELL",
                        "confidence": 0.25,  # Above threshold!
                        "reason": "Aggressive Fallback: Tiny downward momentum detected",
                    }
            else:
                # Last resort - generate random trade
                import random

                return {
                    "prediction": "BUY" if random.random() > 0.5 else "SELL",
                    "confidence": 0.21,  # Just above threshold
                    "reason": "Aggressive Fallback: Random trade for activity",
                }

        # Extract features
        features = self.extract_features(indicator_data)

        # Choose prediction method based on model type
        if self.model_type == "simple":
            prediction, confidence = self.simple_ensemble_predict(features)
            method = "Simple Ensemble"
        elif self.model_type == "ensemble" or self.model_type == "advanced":
            prediction, confidence = self.advanced_ml_predict(
                features, price_history or []
            )
            method = "Advanced ML"
        else:
            prediction, confidence = self.simple_ensemble_predict(features)
            method = "Default Ensemble"

        # AGGRESSIVE FIX: If confidence is 0, boost it artificially
        if confidence <= 0.0:
            # Force minimum confidence based on any movement
            if price_history and len(price_history) >= 2:
                recent_change = (
                    abs(price_history[-1] - price_history[-2]) / price_history[-2] * 100
                )
                confidence = max(0.25, min(0.4, recent_change * 100))  # Boost to 25-40%

                # Ensure we have a real prediction
                if prediction == "HOLD":
                    if price_history[-1] > price_history[-2]:
                        prediction = "BUY"
                    else:
                        prediction = "SELL"
            else:
                confidence = 0.25  # Force minimum viable confidence
                prediction = "BUY" if prediction == "HOLD" else prediction

        # Generate reasoning
        reason = self.generate_prediction_reason(features, prediction, confidence)

        # Store prediction for performance tracking
        prediction_record: Dict[str, Any] = {
            "timestamp": datetime.now(),
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
            "method": method,
        }
        self.prediction_history.append(prediction_record)

        # Keep only recent predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]

        self.performance_metrics["total_predictions"] += 1

        return {
            "prediction": prediction,
            "confidence": confidence,
            "reason": reason,
            "method": method,
            "features_used": list(features.keys()),
            "timestamp": datetime.now().isoformat(),
        }

    def generate_prediction_reason(
        self, features: Dict[str, float], prediction: str, confidence: float
    ) -> str:
        """Generate human-readable reason for the prediction"""
        reasons: List[str] = []

        try:
            # RSI reasons
            if features.get("rsi_oversold", 0) > 0.5:
                reasons.append("RSI oversold condition")
            elif features.get("rsi_overbought", 0) > 0.5:
                reasons.append("RSI overbought condition")

            # Trend reasons
            if features.get("ema_trend", 0) > 0:
                reasons.append("EMA uptrend")
            elif features.get("ema_trend", 0) < 0:
                reasons.append("EMA downtrend")

            # MACD reasons
            if features.get("macd_signal", 0) > 0:
                reasons.append("MACD bullish signal")
            elif features.get("macd_signal", 0) < 0:
                reasons.append("MACD bearish signal")

            # Pattern reasons
            if features.get("breakout", 0) > 0.5:
                reasons.append("Price breakout detected")
            if features.get("trend_up", 0) > 0.5:
                reasons.append("Upward trend pattern")
            elif features.get("trend_down", 0) > 0.5:
                reasons.append("Downward trend pattern")

            # Confidence reason
            confidence_desc = (
                "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
            )

            if reasons:
                reason_text = f"{prediction} signal with {confidence_desc} confidence ({confidence:.2f}). "
                reason_text += (
                    f"Based on: {', '.join(reasons[:3])}"  # Limit to top 3 reasons
                )
                if len(reasons) > 3:
                    reason_text += f" and {len(reasons) - 3} other factors"
            else:
                reason_text = f"{prediction} with {confidence_desc} confidence - no strong signals detected"

            return reason_text

        except Exception:
            return f"{prediction} signal (confidence: {confidence:.2f})"

    def update_performance(self, prediction_id: str, actual_result: str) -> None:
        """Update performance metrics based on actual trade results"""
        try:
            # Find the prediction
            if len(self.prediction_history) > 0:
                last_prediction = self.prediction_history[-1]

                # Simple performance tracking
                if actual_result == "WIN":
                    if last_prediction["prediction"] in ["BUY", "SELL"]:
                        self.performance_metrics["correct_predictions"] += 1
                elif actual_result == "LOSS":
                    # Prediction was wrong or we shouldn't have traded
                    pass

                # Update accuracy
                if self.performance_metrics["total_predictions"] > 0:
                    self.performance_metrics["accuracy"] = (
                        self.performance_metrics["correct_predictions"]
                        / self.performance_metrics["total_predictions"]
                    )

                self.performance_metrics["last_updated"] = datetime.now()

                # Adjust model weights based on performance
                self.adjust_model_weights()

        except Exception as e:
            print(f"Error updating performance: {e}")

    def adjust_model_weights(self) -> None:
        """Adjust feature weights based on performance"""
        try:
            accuracy = self.performance_metrics["accuracy"]

            # If accuracy is low, slightly randomize weights
            if accuracy < 0.4 and self.performance_metrics["total_predictions"] > 20:
                adjustment = 0.05  # 5% adjustment
                for feature in self.feature_weights:
                    # Small random adjustment
                    change = np.random.uniform(-adjustment, adjustment)
                    self.feature_weights[feature] = max(
                        0.05, min(0.4, self.feature_weights[feature] + change)
                    )

                # Normalize weights
                total_weight = sum(self.feature_weights.values())
                for feature in self.feature_weights:
                    self.feature_weights[feature] /= total_weight

                print(f"🔧 Adjusted model weights. New accuracy target: {accuracy:.2f}")

        except Exception as e:
            print(f"Error adjusting model weights: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        report: Dict[str, Any] = {
            "metrics": self.performance_metrics.copy(),
            "feature_weights": self.feature_weights.copy(),
            "recent_predictions": len(self.prediction_history),
            "model_type": self.model_type,
            "confidence_threshold": self.confidence_threshold,
        }

        # Calculate recent performance (last 50 predictions)
        if len(self.prediction_history) >= 10:
            recent_predictions = self.prediction_history[-50:]
            recent_buy_sell = [
                p for p in recent_predictions if p["prediction"] in ["BUY", "SELL"]
            ]

            report["recent_activity"] = {
                "total_signals": len(recent_buy_sell),
                "buy_signals": len(
                    [p for p in recent_buy_sell if p["prediction"] == "BUY"]
                ),
                "sell_signals": len(
                    [p for p in recent_buy_sell if p["prediction"] == "SELL"]
                ),
                "avg_confidence": (
                    np.mean([p["confidence"] for p in recent_buy_sell])
                    if recent_buy_sell
                    else 0.0
                ),
            }

        return report

    def save_model(self, filepath: str = "ai_model_state.pkl") -> bool:
        """Save model state to file"""
        try:
            model_state: Dict[str, Any] = {
                "feature_weights": self.feature_weights,
                "performance_metrics": self.performance_metrics,
                "model_type": self.model_type,
                "confidence_threshold": self.confidence_threshold,
                "prediction_history": self.prediction_history[
                    -100:
                ],  # Save last 100 predictions
            }

            with open(filepath, "wb") as f:
                pickle.dump(model_state, f)

            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str = "ai_model_state.pkl") -> bool:
        """Load model state from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    model_state = pickle.load(f)

                self.feature_weights = model_state.get(
                    "feature_weights", self.feature_weights
                )
                self.performance_metrics = model_state.get(
                    "performance_metrics", self.performance_metrics
                )
                self.model_type = model_state.get("model_type", self.model_type)
                self.confidence_threshold = model_state.get(
                    "confidence_threshold", self.confidence_threshold
                )
                self.prediction_history = model_state.get("prediction_history", [])

                print(f"✅ Model loaded from {filepath}")
                return True
            else:
                print(f"⚠️ Model file {filepath} not found, using default settings")
                return False

        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# Utility functions
def calculate_prediction_accuracy(
    predictions: List[Dict[str, Any]], results: List[str]
) -> float:
    """Calculate prediction accuracy"""
    if len(predictions) != len(results) or len(predictions) == 0:
        return 0.0

    correct = 0
    total = 0

    for pred, result in zip(predictions, results):
        if pred["prediction"] in ["BUY", "SELL"]:
            total += 1
            if result == "WIN":
                correct += 1

    return correct / total if total > 0 else 0.0


def optimize_confidence_threshold(
    prediction_history: List[Dict[str, Any]], results: List[str]
) -> float:
    """Find optimal confidence threshold"""
    if len(prediction_history) < 20:
        return CONFIDENCE_THRESHOLD_01

    best_threshold = CONFIDENCE_THRESHOLD_01
    best_accuracy = 0.0

    # Test different thresholds
    for threshold in np.arange(0.3, 0.9, 0.1):
        filtered_predictions = [
            p for p in prediction_history if p["confidence"] >= threshold
        ]
        if len(filtered_predictions) >= 10:  # Need minimum samples
            accuracy = calculate_prediction_accuracy(
                filtered_predictions, results[-len(filtered_predictions) :]
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

    return best_threshold


if __name__ == "__main__":
    # Test the AI model
    print("🧠 Testing AI Trading Model...")

    ai = TradingAI()

    # Test prediction with sample data
    sample_indicators: Dict[str, Any] = {
        "rsi": 35,
        "ema_fast": 100.5,
        "ema_slow": 100.0,
        "price": 100.8,
        "macd": {"macd": 0.2, "signal": 0.1, "histogram": 0.1},
        "bollinger": {"upper": 102, "middle": 100, "lower": 98},
        "patterns": {"uptrend": True, "breakout_up": True},
        "market_strength": 0.6,
    }

    prediction = ai.predict_trade(sample_indicators)
    print(
        f"✅ Test prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.2f})"
    )
    print(f"📝 Reason: {prediction['reason']}")

    # Test performance report
    report = ai.get_performance_report()
    print(f"📊 Model accuracy: {report['metrics']['accuracy']:.2f}")
    print(f"🎯 Active features: {len(report['feature_weights'])}")
