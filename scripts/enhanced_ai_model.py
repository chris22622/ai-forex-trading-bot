"""
Enhanced AI Model with Reinforcement Learning, Confidence-Based Sizing, and Pattern Recognition
"""

import json
import logging
import os
import pickle
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

try:
    from reinforcement_learning import ReinforcementLearningManager
    HAS_RL = True
except ImportError:
    HAS_RL = False
    print("⚠️ Reinforcement Learning module not available")

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ Scikit-learn not available. ML models disabled.")

class PatternRecognition:
    """LSTM-like pattern recognition using sliding windows"""

    def __init__(self, sequence_length: int = 50, pattern_memory: int = 1000):
        self.sequence_length = sequence_length
        self.pattern_memory = pattern_memory
        self.price_sequences: deque = deque(maxlen=pattern_memory)
        self.outcome_sequences: deque = deque(maxlen=pattern_memory)
        self.pattern_db: Dict[str, List[str]] = {}

    def extract_features(self, price_sequence: List[float]) -> List[float]:
        """Extract features from price sequence"""
        if len(price_sequence) < 5:
            return [0] * 10

        prices = np.array(price_sequence[-self.sequence_length:])

        features = []

        # Ensure we have valid prices
        if len(prices) < 2:
            return [0.0] * 10  # Return default features

        # Price movement features with safe division
        returns = np.diff(prices) / np.maximum(prices[:-1], 1e-8)  # Avoid division by zero
        features.extend([
            np.mean(returns),  # Average return
            np.std(returns),   # Volatility
            np.min(returns),   # Max loss
            np.max(returns),   # Max gain
        ])

        # Trend features with safe division
        if prices[0] != 0:
            total_return = (prices[-1] - prices[0]) / prices[0]
        else:
            total_return = 0.0

        features.extend([
            total_return,  # Total return
            np.sum(returns > 0) / max(len(returns), 1),    # Win rate
        ])

        # Pattern features with safe calculations
        recent_momentum = 1.0
        if len(prices) >= 10:
            recent_mean = np.mean(prices[-5:])
            older_mean = np.mean(prices[-10:])
            if older_mean != 0:
                recent_momentum = recent_mean / older_mean

                up_moves = len(
            [i for i in range(1, len(prices)) if prices[i] > prices[i-1]]) / max(len(prices),
            1
        )
        recent_volatility = np.mean(np.abs(returns[-5:])) if len(returns) >= 5 else 0
        new_high = 1 if len(prices) > 1 and prices[-1] > np.max(prices[:-1]) else 0

        features.extend([
            recent_momentum,  # Recent momentum
            up_moves,  # Up moves ratio
            recent_volatility,  # Recent volatility
            new_high,  # New high indicator
        ])

        return features

        def find_similar_patterns(
        self,
        current_sequence: List[float],
        threshold: float = 0.1
    )
        """Find historically similar patterns"""
        if len(self.price_sequences) < 10:
            return []

        current_features = self.extract_features(current_sequence)
        similar_outcomes = []

        for i, historical_sequence in enumerate(list(self.price_sequences)[-100:]):  # Check last 100
            historical_features = self.extract_features(historical_sequence)

            # Calculate similarity (Euclidean distance)
            distance = np.sqrt(np.sum((np.array(current_features) - np.array(historical_features)) ** 2))

            if distance < threshold:
                outcome_idx = len(self.price_sequences) - 100 + i
                if outcome_idx < len(self.outcome_sequences):
                    similar_outcomes.append(self.outcome_sequences[outcome_idx])

        return similar_outcomes

    def predict_pattern(self, price_sequence: List[float]) -> Dict[str, Any]:
        """Predict based on pattern recognition"""
        similar_outcomes = self.find_similar_patterns(price_sequence)

        if not similar_outcomes:
            return {
                "prediction": "HOLD",
                "confidence": 0.3,
                "method": "PATTERN_NO_MATCH",
                "similar_count": 0
            }

        # Count outcomes
        buy_count = similar_outcomes.count("BUY")
        sell_count = similar_outcomes.count("SELL")
        hold_count = similar_outcomes.count("HOLD")
        total = len(similar_outcomes)
        # Safely calculate confidence - avoid division by None
        total = max(buy_count + sell_count + hold_count, 1)  # Ensure total is never 0

        # Determine prediction
        if buy_count > sell_count and buy_count > hold_count:
            prediction = "BUY"
            confidence = buy_count / total if total > 0 else 0.5
        elif sell_count > buy_count and sell_count > hold_count:
            prediction = "SELL"
            confidence = sell_count / total if total > 0 else 0.5
        else:
            prediction = "HOLD"
            confidence = max(buy_count, sell_count, hold_count) / total if total > 0 else 0.5

        return {
            "prediction": prediction,
            "confidence": confidence,
            "method": "PATTERN_RECOGNITION",
            "similar_count": total,
            "distribution": {"BUY": buy_count, "SELL": sell_count, "HOLD": hold_count}
        }

    def update_pattern_database(self, price_sequence: List[float], outcome: str):
        """Update pattern database with new data"""
        if len(price_sequence) >= self.sequence_length:
            self.price_sequences.append(price_sequence[-self.sequence_length:])
            self.outcome_sequences.append(outcome)

class ConfidenceBasedPositionSizer:
    """Dynamic position sizing based on prediction confidence"""

    def __init__(self, base_amount: float = 1.0, max_multiplier: float = 3.0,
                 min_confidence: float = 0.6):
        self.base_amount = base_amount
        self.max_multiplier = max_multiplier
        self.min_confidence = min_confidence
        self.position_history: List[Dict[str, Any]] = []

    def calculate_position_size(self, confidence: float, balance: float,
                               recent_performance: Dict[str, Any]) -> float:
        """Calculate position size based on confidence and performance"""

        # Base confidence scaling
        if confidence < self.min_confidence:
            return 0  # No trade

        # Confidence-based multiplier (linear scaling)
        confidence_multiplier = 1 + (confidence - self.min_confidence) * (self.max_multiplier - 1) / (1 - self.min_confidence)

        # Performance-based adjustment
        performance_multiplier = 1.0
        if recent_performance:
            win_rate = recent_performance.get('win_rate', 0.5)
            recent_profit = recent_performance.get('recent_profit', 0)

            # Reduce size if recent performance is poor
            if win_rate < 0.4:
                performance_multiplier *= 0.7
            elif win_rate > 0.7:
                performance_multiplier *= 1.2

            # Reduce size if recent losses
            if recent_profit < -50:
                performance_multiplier *= 0.5

        # Risk management - never risk more than 5% of balance
        max_risk_amount = balance * 0.05

        # Calculate final position size
        raw_size = self.base_amount * confidence_multiplier * performance_multiplier
        final_size = min(raw_size, max_risk_amount)

        # Store position decision
        self.position_history.append({
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'confidence_multiplier': confidence_multiplier,
            'performance_multiplier': performance_multiplier,
            'raw_size': raw_size,
            'final_size': final_size,
            'balance': balance
        })

        return round(final_size, 2)

    def get_position_stats(self) -> Dict[str, Any]:
        """Get position sizing statistics"""
        if not self.position_history:
            return {}

        recent_positions = self.position_history[-50:]  # Last 50 positions

        return {
            'total_positions': len(self.position_history),
            'average_size': np.mean([p['final_size'] for p in recent_positions]),
            'max_size': np.max([p['final_size'] for p in recent_positions]),
            'min_size': np.min([p['final_size'] for p in recent_positions]),
            'average_confidence': np.mean([p['confidence'] for p in recent_positions]),
            'size_utilization': np.mean([p['final_size'] / p['raw_size'] for p in recent_positions if p['raw_size'] > 0])
        }

class EnsembleAIModel:
    """Ensemble AI model combining multiple approaches"""

    def __init__(self, base_trade_amount: float = 1.0):
        # Components
        self.pattern_recognition = PatternRecognition()
        self.position_sizer = ConfidenceBasedPositionSizer(base_trade_amount)

        # RL component (if available)
        self.rl_manager = None
        if HAS_RL:
            try:
                self.rl_manager = ReinforcementLearningManager()
                print("✅ Reinforcement Learning enabled")
            except Exception as e:
                print(f"⚠️ RL initialization failed: {e}")

        # ML Models (if sklearn available)
        self.ml_models = {}
        self.scalers = {}
        if HAS_SKLEARN:
            self._initialize_ml_models()

        # Training data storage
        self.training_data: List[Dict[str, Any]] = []
        self.model_performance: Dict[str, List[float]] = {
            'rule_based': [],
            'pattern_recognition': [],
            'reinforcement_learning': [],
            'ensemble': []
        }

        # Configuration
        self.ensemble_weights = {
            'rule_based': 0.3,
            'pattern_recognition': 0.3,
            'reinforcement_learning': 0.4 if HAS_RL else 0.0,
            'ml_model': 0.0  # Will be activated after training
        }

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('EnhancedAI')
        logger.setLevel(logging.INFO)
        return logger

    def _initialize_ml_models(self):
        """Initialize ML models for ensemble"""
        self.ml_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'neural_net': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        }

        for model_name in self.ml_models:
            self.scalers[model_name] = StandardScaler()

    def extract_ml_features(self, indicators: Dict[str, Any],
                           price_history: List[float]) -> List[float]:
        """Extract features for ML models"""
        features = []

        # Price-based features with safe calculations
        if len(price_history) >= 20:
            prices = np.array(price_history[-20:])
            returns = np.diff(prices) / np.maximum(prices[:-1], 1e-8)  # Avoid division by zero

            # Safe total return calculation
            total_return = 0.0
            if prices[0] != 0:
                total_return = (prices[-1] - prices[0]) / prices[0]

            # Safe MA ratio calculation
            ma_ratio = 1.0
            if len(prices) >= 10:
                short_ma = np.mean(prices[-5:])
                long_ma = np.mean(prices[-10:])
                if long_ma != 0:
                    ma_ratio = short_ma / long_ma

            features.extend([
                np.mean(returns),           # Mean return
                np.std(returns),            # Volatility
                np.min(returns),            # Max loss
                np.max(returns),            # Max gain
                total_return,               # Total return
                ma_ratio,                   # Short/long MA ratio
            ])
        else:
            features.extend([0] * 6)

        # Technical indicators with safe defaults
        current_price = max(indicators.get('price', 1), 1e-8)  # Avoid division by zero
        ema_fast = indicators.get('ema_fast')
        ema_slow = indicators.get('ema_slow')

        # Safe ratio calculations
        ema_fast_ratio = 0.0
        if ema_fast is not None and current_price > 0:
            ema_fast_ratio = ema_fast / current_price

        ema_slow_ratio = 0.0
        if ema_slow is not None and current_price > 0:
            ema_slow_ratio = ema_slow / current_price

        features.extend([
            indicators.get('rsi', 50) / 100 if indicators.get('rsi') is not None else 0.5,  # Normalized RSI
            ema_fast_ratio,  # EMA fast ratio
            ema_slow_ratio,  # EMA slow ratio
            1 if indicators.get('signals', {}).get('rsi_signal') == 'BUY' else 0,
            1 if indicators.get('signals', {}).get('ema_signal') == 'BUY' else 0,
        ])

        # Time-based features
        now = datetime.now()
        features.extend([
            now.hour / 24,              # Hour of day
            now.weekday() / 7,          # Day of week
            (now.minute % 60) / 60,     # Minute in hour
        ])

        return features

    def get_rule_based_prediction(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Get traditional rule-based prediction"""
        signals = indicators.get('signals', {})
        overall_signal = signals.get('overall_signal', 'HOLD')

        # Calculate confidence based on signal strength
        signal_count = 0
        total_signals = 0

        for signal_name, signal_value in signals.items():
            if signal_name != 'overall_signal':
                total_signals += 1
                if signal_value == overall_signal:
                    signal_count += 1

        confidence = signal_count / max(total_signals, 1) if total_signals > 0 else 0.5

        # Boost confidence for strong RSI signals
        rsi = indicators.get('rsi', 50)
        if rsi < 30 and overall_signal == 'BUY':
            confidence += 0.2
        elif rsi > 70 and overall_signal == 'SELL':
            confidence += 0.2

        confidence = min(confidence, 1.0)

        return {
            "prediction": overall_signal,
            "confidence": confidence,
            "method": "RULE_BASED",
            "signal_strength": signal_count / max(total_signals, 1)
        }

    def get_ml_prediction(self, indicators: Dict[str, Any],
                         price_history: List[float]) -> Dict[str, Any]:
        """Get ML model predictions"""
        if not HAS_SKLEARN or not self.ml_models:
            return {
                "prediction": "HOLD",
                "confidence": 0.3,
                "method": "ML_UNAVAILABLE"
            }

        try:
            features = self.extract_ml_features(indicators, price_history)
            features_array = np.array(features).reshape(1, -1)

            predictions = []
            confidences = []

            for model_name, model in self.ml_models.items():
                if hasattr(model, 'predict'):  # Model is trained
                    try:
                        # Check if scaler is fitted before using it
                        scaler = self.scalers[model_name]
                        if not hasattr(scaler, 'scale_') or scaler.scale_ is None:
                            # Scaler not fitted, skip this model silently
                            continue

                        # Scale features
                        scaled_features = scaler.transform(features_array)

                        # Get prediction
                        pred = model.predict(scaled_features)[0]

                        # Get confidence (if available)
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(scaled_features)[0]
                            confidence = np.max(proba)
                        else:
                            confidence = 0.7  # Default confidence

                        predictions.append(pred)
                        confidences.append(confidence)

                    except Exception as e:
                        # Don't spam logs with unfitted model errors
                        if "not fitted" not in str(e).lower():
                            self.logger.warning(f"ML model {model_name} prediction failed: {e}")

            if not predictions:
                return {
                    "prediction": "HOLD",
                    "confidence": 0.3,
                    "method": "ML_NO_PREDICTIONS"
                }

            # Ensemble ML predictions
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            pred_counts = {0: 0, 1: 0, 2: 0}

            for pred in predictions:
                pred_counts[pred] += 1

            final_pred = max(pred_counts, key=pred_counts.get)
            final_confidence = np.mean(confidences)

            return {
                "prediction": action_map[final_pred],
                "confidence": final_confidence,
                "method": "ML_ENSEMBLE",
                "model_count": len(predictions),
                "raw_predictions": predictions
            }

        except Exception as e:
            # Only log significant errors, not unfitted model errors
            if "not fitted" not in str(e).lower():
                self.logger.error(f"ML prediction error: {e}")
            return {
                "prediction": "HOLD",
                "confidence": 0.3,
                "method": "ML_ERROR"
            }

    def predict_trade(self, indicators: Dict[str, Any],
                     price_history: List[float]) -> Dict[str, Any]:
        """Main prediction method using ensemble approach"""
        try:
            # Get predictions from all methods
            predictions = {}

            # 1. Rule-based prediction
            predictions['rule_based'] = self.get_rule_based_prediction(indicators)

            # 2. Pattern recognition
            predictions['pattern_recognition'] = self.pattern_recognition.predict_pattern(price_history)

            # 3. Reinforcement learning (if available)
            if self.rl_manager:
                predictions['reinforcement_learning'] = self.rl_manager.get_action_recommendation(
                    indicators, price_history
                )

            # 4. ML models (if trained)
            predictions['ml_model'] = self.get_ml_prediction(indicators, price_history)

            # Ensemble the predictions
            final_prediction = self._ensemble_predictions(predictions)

            # Calculate position size based on confidence
            if final_prediction['prediction'] != 'HOLD':
                balance = indicators.get('balance', 1000)  # Default balance
                recent_performance = self._get_recent_performance()

                position_size = self.position_sizer.calculate_position_size(
                    final_prediction['confidence'], balance, recent_performance
                )

                final_prediction['position_size'] = position_size
                final_prediction['base_amount'] = self.position_sizer.base_amount

                # Don't trade if position size is too small
                if position_size < 0.1:
                    final_prediction['prediction'] = 'HOLD'
                    final_prediction['reason'] += ' (Position size too small)'

            return final_prediction

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {
                "prediction": "HOLD",
                "confidence": 0.3,
                "method": "ERROR",
                "reason": f"Prediction failed: {str(e)}",
                "position_size": 0
            }

    def get_ensemble_prediction(self, indicators: Dict[str, Any],
                               price_history: List[float]) -> Dict[str, Any]:
        """Alias for predict_trade method - for backward compatibility"""
        return self.predict_trade(indicators, price_history)

    def _ensemble_predictions(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine predictions using weighted ensemble"""

        # Weighted voting
        action_scores = {"BUY": 0, "SELL": 0, "HOLD": 0}
        total_weight = 0
        reasons = []
        methods_used = []

        for method, weight in self.ensemble_weights.items():
            if method in predictions and weight > 0:
                pred = predictions[method]
                action = pred['prediction']
                confidence = pred['confidence']

                # Weight by both ensemble weight and prediction confidence
                effective_weight = weight * confidence
                action_scores[action] += effective_weight
                total_weight += effective_weight

                methods_used.append(f"{method}({action}, {confidence:.2f})")

                # Collect reasons
                if 'reason' in pred:
                    reasons.append(f"{method}: {pred['reason']}")

        # Determine final action
        if total_weight == 0:
            final_action = "HOLD"
            final_confidence = 0.3
        else:
            # Normalize scores
            for action in action_scores:
                action_scores[action] /= total_weight

            final_action = max(action_scores, key=action_scores.get)
            final_confidence = action_scores[final_action]

        # Apply minimum confidence threshold
        if final_confidence < 0.6:
            final_action = "HOLD"

        return {
            "prediction": final_action,
            "confidence": final_confidence,
            "method": "ENSEMBLE",
            "reason": f"Ensemble of {', '.join(methods_used)}",
            "detailed_scores": action_scores,
            "component_predictions": predictions,
            "features_used": len(methods_used)
        }

    def update_performance(self, trade_id: str, result: str, profit_loss: float = 0,
                          indicators: Dict[str, Any] = None, price_history: List[float] = None):
        """Update model performance with trade results"""
        try:
            # Update pattern recognition
            if price_history and result in ["BUY", "SELL", "HOLD"]:
                outcome = "BUY" if result == "WIN" and profit_loss > 0 else "SELL" if result == "WIN" and profit_loss < 0 else "HOLD"
                self.pattern_recognition.update_pattern_database(price_history, outcome)

            # Update RL model
            if self.rl_manager and indicators and price_history:
                confidence = 0.7  # Default, should come from original prediction
                                self.rl_manager.update_with_result(
                    result, profit_loss, confidence, indicators, price_history)

            # Store training data for ML models
            if indicators and price_history:
                features = self.extract_ml_features(indicators, price_history)
                label = 1 if result == "WIN" else 0  # Binary classification

                self.training_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'features': features,
                    'label': label,
                    'trade_id': trade_id,
                    'result': result,
                    'profit_loss': profit_loss
                })

            self.logger.info(f"Updated performance for trade {trade_id}: {result}")

        except Exception as e:
            self.logger.error(f"Performance update error: {e}")

    def retrain_models(self, min_samples: int = 100):
        """Retrain ML models with accumulated data"""
        if not HAS_SKLEARN or len(self.training_data) < min_samples:
            self.logger.info(f"Not enough training data: {len(self.training_data)}/{min_samples}")
            return False

        try:
            # Prepare training data
            features = []
            labels = []

            for data_point in self.training_data[-1000:]:  # Use last 1000 samples
                features.append(data_point['features'])
                labels.append(data_point['label'])

            X = np.array(features)
            y = np.array(labels)

            # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            # Train each model
            trained_models = 0
            for model_name, model in self.ml_models.items():
                try:
                    # Scale features
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_test_scaled = self.scalers[model_name].transform(X_test)

                    # Train model
                    model.fit(X_train_scaled, y_train)

                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)

                    self.logger.info(f"Model {model_name} trained with accuracy: {accuracy:.3f}")
                    trained_models += 1

                except Exception as e:
                    self.logger.error(f"Failed to train {model_name}: {e}")

            if trained_models > 0:
                # Enable ML model in ensemble
                self.ensemble_weights['ml_model'] = 0.2
                # Rebalance other weights
                self.ensemble_weights['rule_based'] = 0.25
                self.ensemble_weights['pattern_recognition'] = 0.25
                self.ensemble_weights['reinforcement_learning'] = 0.3 if HAS_RL else 0.0

                self.logger.info(f"Retrained {trained_models} ML models successfully")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Model retraining error: {e}")
            return False

    def _get_recent_performance(self) -> Dict[str, Any]:
        """Get recent performance metrics"""
        if len(self.training_data) < 10:
            return {}

        recent_data = self.training_data[-50:]  # Last 50 trades
        wins = sum(1 for d in recent_data if d['label'] == 1)
        total = len(recent_data)
        recent_profit = sum(d.get('profit_loss', 0) for d in recent_data)

        return {
            'win_rate': wins / total,
            'total_trades': total,
            'recent_profit': recent_profit
        }

    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        stats = {
            'training_data_size': len(self.training_data),
            'ensemble_weights': self.ensemble_weights,
            'recent_performance': self._get_recent_performance(),
            'position_stats': self.position_sizer.get_position_stats()
        }

        # Add RL stats if available
        if self.rl_manager:
            stats['rl_performance'] = self.rl_manager.get_performance_metrics()

        # Add pattern recognition stats
        stats['pattern_recognition'] = {
            'pattern_count': len(self.pattern_recognition.price_sequences),
            'sequence_length': self.pattern_recognition.sequence_length
        }

        return stats

    def save_model(self, filepath: str = "models/enhanced_ai_model"):
        """Save the enhanced AI model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save main model data
            model_data = {
                'ensemble_weights': self.ensemble_weights,
                'training_data': self.training_data[-1000:],  # Save last 1000 samples
                'position_sizer_history': self.position_sizer.position_history[-100:],
                'pattern_sequences': list(self.pattern_recognition.price_sequences)[-100:],
                'pattern_outcomes': list(self.pattern_recognition.outcome_sequences)[-100:],
                'model_performance': self.model_performance
            }

            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(model_data, f)

            # Save ML models if available
            if HAS_SKLEARN:
                for model_name, model in self.ml_models.items():
                    if hasattr(model, 'predict'):  # Model is trained
                        with open(f"{filepath}_{model_name}.pkl", 'wb') as f:
                            pickle.dump({
                                'model': model,
                                'scaler': self.scalers[model_name]
                            }, f)

            # Save RL model
            if self.rl_manager:
                self.rl_manager.save_model(f"{filepath}_rl")

            self.logger.info(f"Enhanced AI model saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, filepath: str = "models/enhanced_ai_model"):
        """Load the enhanced AI model"""
        try:
            # Load main model data
            with open(f"{filepath}.pkl", 'rb') as f:
                model_data = pickle.load(f)

            self.ensemble_weights = model_data.get('ensemble_weights', self.ensemble_weights)
            self.training_data = model_data.get('training_data', [])
            self.position_sizer.position_history = model_data.get('position_sizer_history', [])
            self.model_performance = model_data.get('model_performance', self.model_performance)

            # Restore pattern recognition data
            pattern_sequences = model_data.get('pattern_sequences', [])
            pattern_outcomes = model_data.get('pattern_outcomes', [])

            for seq, outcome in zip(pattern_sequences, pattern_outcomes):
                self.pattern_recognition.price_sequences.append(seq)
                self.pattern_recognition.outcome_sequences.append(outcome)

            # Load ML models if available
            if HAS_SKLEARN:
                for model_name in self.ml_models.keys():
                    try:
                        with open(f"{filepath}_{model_name}.pkl", 'rb') as f:
                            ml_data = pickle.load(f)
                        self.ml_models[model_name] = ml_data['model']
                        self.scalers[model_name] = ml_data['scaler']
                    except FileNotFoundError:
                        pass  # Model not trained yet

            # Load RL model
            if self.rl_manager:
                self.rl_manager.load_model(f"{filepath}_rl")

            self.logger.info(f"Enhanced AI model loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Test enhanced AI model
    enhanced_ai = EnsembleAIModel(base_trade_amount=2.0)

    # Simulate some data
    dummy_indicators = {
        "rsi": 35,
        "ema_fast": 100.5,
        "ema_slow": 100.2,
        "price": 100.3,
        "signals": {
            "rsi_signal": "BUY",
            "ema_signal": "BUY",
            "overall_signal": "BUY"
        },
        "balance": 1000
    }

    dummy_prices = [99.8, 100.0, 100.1, 100.3, 100.5]

    # Get prediction
    prediction = enhanced_ai.predict_trade(dummy_indicators, dummy_prices)
    print(f"Enhanced AI Prediction: {prediction}")

    # Update with result
    enhanced_ai.update_performance("test_trade_1", "WIN", 5.0, dummy_indicators, dummy_prices)

    # Get stats
    stats = enhanced_ai.get_model_stats()
    print(f"Model Stats: {json.dumps(stats, indent=2)}")
