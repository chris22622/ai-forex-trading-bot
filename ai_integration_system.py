"""
AI Model Integration System
Integrates all AI components for comprehensive trading decisions
"""

from typing import Dict, List, Optional, Any, cast
from datetime import datetime
import numpy as np
from safe_logger import get_safe_logger

# Use safe logger for Windows compatibility
logger = get_safe_logger(__name__)

# Import our AI modules with proper type hints
enhanced_ai_available = True
rl_available = True
dynamic_risk_available = True
backtesting_available = True
basic_ai_available = True

# Type stubs for imports
EnsembleAIModel = None
ReinforcementLearningManager = None
MultiStrategyOrchestrator = None
BacktestingEngine = None
MarketDataRecorder = None

try:
    from enhanced_ai_model import EnsembleAIModel
    enhanced_ai_available = True
except ImportError:
    enhanced_ai_available = False
    logger.warning("Enhanced AI model not available")

try:
    from reinforcement_learning import ReinforcementLearningManager
    rl_available = True
except ImportError:
    rl_available = False
    logger.warning("Reinforcement Learning not available")

try:
    from dynamic_risk_management import MultiStrategyOrchestrator
except ImportError:
    dynamic_risk_available = False
    logger.warning("Dynamic risk management not available")

try:
    from backtesting_analytics import BacktestingEngine, MarketDataRecorder
except ImportError:
    backtesting_available = False
    logger.warning("Backtesting engine not available")

class AIModelManager:
    """Manages all AI models and provides unified predictions"""
    
    def __init__(self, initial_balance: float = 1000.0, enable_features: Optional[Dict[str, bool]] = None):
        self.initial_balance = initial_balance
        
        # Feature flags
        default_features = {
            'enhanced_ai': True,
            'reinforcement_learning': True,
            'dynamic_risk': True,
            'backtesting': True,
            'pattern_recognition': True
        }
        self.features = {**default_features, **(enable_features or {})}
        
        # Initialize components
        self.models: Dict[str, Any] = {}
        self.performance_tracker: Dict[str, List[float]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        self._initialize_models()
        
        logger.info(f"[BOT] AI Model Manager initialized with features: {self.features}")
    
    def _initialize_models(self):
        """Initialize available AI models"""
        
        # Enhanced AI Model (Ensemble)
        if enhanced_ai_available and self.features['enhanced_ai'] and EnsembleAIModel is not None:
            try:
                self.models['enhanced'] = EnsembleAIModel()
                self.performance_tracker['enhanced'] = []
                logger.info("✅ Enhanced AI model loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load enhanced AI: {e}")
        
        # Reinforcement Learning
        if rl_available and self.features['reinforcement_learning'] and ReinforcementLearningManager is not None:
            try:
                self.models['rl'] = ReinforcementLearningManager()
                self.performance_tracker['rl'] = []
                logger.info("✅ Reinforcement learning model loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load RL model: {e}")
        
        # Dynamic Risk Management & Multi-Strategy
        if dynamic_risk_available and self.features['dynamic_risk'] and MultiStrategyOrchestrator is not None:
            try:
                self.models['multi_strategy'] = MultiStrategyOrchestrator(self.initial_balance)
                self.performance_tracker['multi_strategy'] = []
                logger.info("✅ Multi-strategy orchestrator loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load multi-strategy: {e}")
        
        # Backtesting Engine
        if backtesting_available and self.features['backtesting'] and BacktestingEngine is not None and MarketDataRecorder is not None:
            try:
                self.models['backtesting'] = BacktestingEngine()
                self.models['data_recorder'] = MarketDataRecorder()
                logger.info("✅ Backtesting engine loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load backtesting: {e}")
        
        # Create basic fallback if no models available
        if not self.models:
            logger.warning("⚠️ No AI models available - creating basic fallback")
            # Simple fallback implementation
            self.models['fallback'] = {'type': 'basic_fallback'}
            self.performance_tracker['fallback'] = []
    
    def get_trading_prediction(self, indicators: Dict[str, Any], 
                             price_history: List[float], 
                             base_trade_amount: float = 10.0) -> Dict[str, Any]:
        """Get comprehensive trading prediction from all available models"""
        
        predictions = {}
        final_prediction: Dict[str, Any] = {
            'prediction': 'HOLD',
            'confidence': 0.0,
            'position_size': 0.0,
            'method': 'NO_MODELS',
            'models_used': [],
            'ensemble_details': {},
            'risk_assessment': {}
        }
        
        try:
            # Update market data for models that need it
            current_price = indicators.get('price', 0)
            if current_price > 0:
                self._update_market_data(current_price)
            
            # Get predictions from each model
            predictions = self._collect_predictions(indicators, price_history, base_trade_amount)
            
            # Ensemble the predictions
            final_prediction = self._ensemble_predictions(predictions, base_trade_amount)
            
            # Record the prediction for learning
            self._record_prediction(final_prediction)
            
        except Exception as e:
            logger.error(f"❌ Error getting prediction: {e}")
            final_prediction['error'] = str(e)
        
        return final_prediction
    
    def _update_market_data(self, price: float, volume: Optional[float] = None):
        """Update market data for all models that need it"""
        
        # Update multi-strategy orchestrator
        if 'multi_strategy' in self.models:
            try:
                self.models['multi_strategy'].update_market_data(price, volume)
            except Exception as e:
                logger.error(f"Error updating multi-strategy market data: {e}")
        
        # Update data recorder
        if 'data_recorder' in self.models:
            try:
                import time
                timestamp = int(time.time())
                symbol = "R_75"  # Default symbol
                self.models['data_recorder'].record_tick(timestamp, symbol, price, "", 0.00001)
            except Exception as e:
                logger.error(f"Error recording market data: {e}")
    
    def _collect_predictions(self, indicators: Dict[str, Any], 
                           price_history: List[float], 
                           base_trade_amount: float) -> Dict[str, Dict[str, Any]]:
        """Collect predictions from all available models"""
        
        predictions: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced AI Model
        if 'enhanced' in self.models:
            try:
                pred = self.models['enhanced'].get_ensemble_prediction(indicators, price_history)
                predictions['enhanced'] = pred
            except Exception as e:
                logger.error(f"Enhanced AI prediction error: {e}")
        
        # Reinforcement Learning with better error handling
        if 'rl' in self.models:
            try:
                # RL expects indicators as Dict[str, Any], so pass the original indicators dict
                if hasattr(self.models['rl'], 'get_state'):
                    # Use RL's own get_state method with the correct type
                    state = self.models['rl'].get_state(indicators, price_history)
                else:
                    # Fallback: convert indicators to state manually
                    state = self._indicators_to_state(indicators, price_history)
                
                # Safely get action from RL model
                if hasattr(self.models['rl'], 'get_action') and state is not None:
                    # Check if get_action expects price_history as well
                    import inspect
                    sig = inspect.signature(self.models['rl'].get_action)
                    param_count = len(sig.parameters)
                    
                    if param_count > 1:
                        # Method expects both state and price_history
                        action = self.models['rl'].get_action(state, price_history)
                    else:
                        # Method expects only state
                        action = self.models['rl'].get_action(state)
                    
                    # Ensure action is a valid number
                    if action is None:
                        action = 0  # Default to hold
                    elif isinstance(action, dict):
                        # If action is a dict, extract the actual action
                        try:
                            if 'action' in action:
                                action_value = cast(Any, action['action'])
                                # Safe conversion to int
                                if isinstance(action_value, (int, float)):
                                    action = int(action_value)
                                elif isinstance(action_value, str) and action_value.isdigit():
                                    action = int(action_value)
                                else:
                                    action = 0
                            else:
                                action = 0
                        except (ValueError, TypeError, KeyError):
                            action = 0
                    elif not isinstance(action, (int, float)):
                        try:
                            action = int(action) if hasattr(action, '__int__') else 0
                        except (ValueError, TypeError):
                            action = 0
                    else:
                        action = int(action)
                    
                    # Convert RL action to trading prediction
                    if action == 1:  # Buy
                        pred: Dict[str, Any] = {'prediction': 'BUY', 'confidence': 0.7, 'position_size': base_trade_amount}
                    elif action == 2:  # Sell
                        pred = {'prediction': 'SELL', 'confidence': 0.7, 'position_size': base_trade_amount}
                    else:  # Hold
                        pred = {'prediction': 'HOLD', 'confidence': 0.5, 'position_size': 0}
                    
                    predictions['rl'] = pred
                else:
                    # RL model doesn't have get_action method or state is invalid
                    predictions['rl'] = {'prediction': 'HOLD', 'confidence': 0.3, 'position_size': 0}
                    
            except Exception as e:
                # Don't spam logs with RL errors - log only significant issues
                if "unsupported operand" not in str(e).lower() and "float() argument" not in str(e).lower():
                    logger.error(f"RL prediction error: {e}")
                # Use fallback prediction
                predictions['rl'] = {'prediction': 'HOLD', 'confidence': 0.3, 'position_size': 0}
        
        # Multi-Strategy Orchestrator with safe execution
        if 'multi_strategy' in self.models:
            try:
                pred = self.models['multi_strategy'].get_trading_recommendation(
                    indicators, price_history, base_trade_amount
                )
                # Ensure prediction is valid
                if pred and pred.get('prediction'):
                    # Validate confidence is a number
                    confidence = pred.get('confidence', 0.5)
                    if confidence is None or not isinstance(confidence, (int, float)):
                        confidence = 0.5
                    pred['confidence'] = confidence
                    predictions['multi_strategy'] = pred
                else:
                    predictions['multi_strategy'] = {'prediction': 'HOLD', 'confidence': 0.5, 'position_size': 0}
            except Exception as e:
                # Don't spam logs with multi-strategy errors
                if "not supported between instances" not in str(e).lower():
                    logger.error(f"Multi-strategy prediction error: {e}")
                predictions['multi_strategy'] = {'prediction': 'HOLD', 'confidence': 0.5, 'position_size': 0}
        
        # Basic AI Model (fallback)
        if 'basic' in self.models and not predictions:
            try:
                pred = self.models['basic'].predict(indicators)
                pred['position_size'] = base_trade_amount if pred['prediction'] != 'HOLD' else 0
                predictions['basic'] = pred
            except Exception as e:
                logger.error(f"Basic AI prediction error: {e}")
        
        return predictions
    
    def _indicators_to_state(self, indicators: Dict[str, Any], price_history: List[float]) -> List[float]:
        """Convert indicators to state vector for RL"""
        state: List[float] = []
        
        # Safely get current price
        current_price = indicators.get('price', 1.0)
        if current_price is None or current_price <= 0:
            current_price = 1.0  # Fallback to avoid division by zero
        
        # Price indicators with safe defaults and validation
        rsi = indicators.get('rsi', 50.0)
        if rsi is None or not isinstance(rsi, (int, float)):
            rsi = 50.0
        state.append(float(rsi) / 100.0)  # Normalize RSI
        
        macd = indicators.get('macd', 0.0)
        if macd is None or not isinstance(macd, (int, float)):
            macd = 0.0
        state.append(float(macd) / 10.0)   # Normalize MACD
        
        # Moving averages (normalized by current price) with safety checks
        ema_fast = indicators.get('ema_fast', current_price)
        if ema_fast is None or not isinstance(ema_fast, (int, float)):
            ema_fast = current_price
        state.append(float(ema_fast) / float(current_price) - 1)
        
        ema_slow = indicators.get('ema_slow', current_price)
        if ema_slow is None or not isinstance(ema_slow, (int, float)):
            ema_slow = current_price
        state.append(float(ema_slow) / float(current_price) - 1)
        
        # Price momentum (recent change)
        if len(price_history) >= 5:
            try:
                recent_price = float(price_history[-5])
                current_price_f = float(price_history[-1])
                if recent_price > 0:
                    momentum = (current_price_f - recent_price) / recent_price
                    state.append(float(np.clip(momentum * 100, -1, 1)))  # Clip and normalize
                else:
                    state.append(0.0)
            except (ValueError, TypeError, ZeroDivisionError):
                state.append(0.0)
        else:
            state.append(0.0)
        
        # Volatility (standard deviation of recent returns)
        if len(price_history) >= 10:
            try:
                recent_prices = [float(p) for p in price_history[-10:]]
                returns = np.diff(recent_prices) / np.array(recent_prices[:-1])
                volatility = np.std(returns)
                state.append(float(np.clip(volatility * 10, 0, 1)))  # Normalize volatility
            except (ValueError, TypeError, ZeroDivisionError):
                state.append(0.0)
        else:
            state.append(0.0)
        
        return state
    
    def _ensemble_predictions(self, predictions: Dict[str, Dict[str, Any]], 
                            base_trade_amount: float) -> Dict[str, Any]:
        """Ensemble multiple model predictions into final decision"""
        
        if not predictions:
            return {
                'prediction': 'HOLD',
                'confidence': 0.0,
                'position_size': 0.0,
                'method': 'NO_PREDICTIONS',
                'models_used': [],
                'ensemble_details': {},
                'risk_assessment': {}
            }
        
        # Weight models based on historical performance
        model_weights = self._calculate_model_weights(list(predictions.keys()))
        
        # Aggregate predictions
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_confidence = 0.0
        position_sizes: List[float] = []
        
        ensemble_details = {}
        
        for model_name, pred in predictions.items():
            weight = model_weights.get(model_name, 1.0)
            confidence = pred.get('confidence', 0.5)
            weighted_confidence = confidence * weight
            
            if pred['prediction'] == 'BUY':
                buy_score += weighted_confidence
            elif pred['prediction'] == 'SELL':
                sell_score += weighted_confidence
            else:
                hold_score += weighted_confidence
            
            total_confidence += weighted_confidence
            position_sizes.append(pred.get('position_size', 0))
            
            ensemble_details[model_name] = {
                'prediction': pred['prediction'],
                'confidence': confidence,
                'weight': weight,
                'weighted_score': weighted_confidence
            }
        
        # Determine final prediction
        if buy_score > sell_score and buy_score > hold_score and buy_score > 0.3:
            final_pred = 'BUY'
            final_confidence = buy_score / max(total_confidence, 0.1)
        elif sell_score > buy_score and sell_score > hold_score and sell_score > 0.3:
            final_pred = 'SELL'
            final_confidence = sell_score / max(total_confidence, 0.1)
        else:
            final_pred = 'HOLD'
            final_confidence = hold_score / max(total_confidence, 0.1)
        
        # Calculate position size (average of non-zero sizes, with confidence adjustment)
        non_zero_sizes = [size for size in position_sizes if size > 0]
        if non_zero_sizes and final_pred != 'HOLD':
            avg_size = np.mean(non_zero_sizes)
            final_size = avg_size * final_confidence
        else:
            final_size = 0.0
        
        # Risk assessment
        risk_assessment = self._assess_risk(predictions, final_pred, final_confidence)
        
        return {
            'prediction': final_pred,
            'confidence': final_confidence,
            'position_size': round(final_size, 2),
            'method': 'ENSEMBLE_AI',
            'models_used': list(predictions.keys()),
            'ensemble_details': ensemble_details,
            'risk_assessment': risk_assessment,
            'model_weights': model_weights,
            'scores': {
                'buy': buy_score,
                'sell': sell_score,
                'hold': hold_score
            }
        }
    
    def _calculate_model_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Calculate weights for each model based on historical performance"""
        
        weights: Dict[str, float] = {}
        
        for model_name in model_names:
            performance_history = self.performance_tracker.get(model_name, [])
            
            if len(performance_history) >= 10:
                # Calculate win rate and average return
                wins = len([p for p in performance_history[-20:] if p > 0])
                win_rate = wins / len(performance_history[-20:])
                avg_return = np.mean(performance_history[-20:])
                
                # Combine metrics for weight (favor recent performance)
                weight = (win_rate * 0.6 + (avg_return + 1) * 0.4)
                weights[model_name] = max(0.1, min(2.0, float(weight)))  # Clamp between 0.1 and 2.0
            else:
                # Default weight for new models
                weights[model_name] = 1.0
        
        return weights
    
    def _assess_risk(self, predictions: Dict[str, Dict[str, Any]], 
                    final_pred: str, final_confidence: float) -> Dict[str, Any]:
        """Assess risk of the final prediction"""
        
        # Check for model agreement
        pred_counts: Dict[str, int] = {}
        for pred in predictions.values():
            p = pred['prediction']
            pred_counts[p] = pred_counts.get(p, 0) + 1
        
        agreement_ratio = max(pred_counts.values()) / len(predictions) if predictions else 0.0
        
        # Check for high confidence outliers
        confidences = [pred.get('confidence', 0) for pred in predictions.values()]
        confidence_std = np.std(confidences) if len(confidences) > 1 else 0
        
        # Risk factors
        risk_factors: List[str] = []
        risk_score = 0.0
        
        if agreement_ratio < 0.6:
            risk_factors.append("Low model agreement")
            risk_score += 0.3
        
        if confidence_std > 0.3:
            risk_factors.append("High confidence variance")
            risk_score += 0.2
        
        if final_confidence < 0.4:
            risk_factors.append("Low final confidence")
            risk_score += 0.2
        
        if len(predictions) < 2:
            risk_factors.append("Single model prediction")
            risk_score += 0.3
        
        return {
            'risk_score': min(1.0, risk_score),
            'risk_level': 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW',
            'risk_factors': risk_factors,
            'model_agreement': agreement_ratio,
            'confidence_variance': confidence_std
        }
    
    def update_trade_result(self, prediction_result: Dict[str, Any], 
                          actual_result: str, profit_loss: float):
        """Update all models with trade result"""
        
        models_used = prediction_result.get('models_used', [])
        position_size = prediction_result.get('position_size', 0)
        
        # Normalize result for tracking
        normalized_result = 1.0 if profit_loss > 0 else -1.0 if profit_loss < 0 else 0.0
        
        # Update performance tracking
        for model_name in models_used:
            if model_name in self.performance_tracker:
                self.performance_tracker[model_name].append(normalized_result)
                
                # Keep only recent history
                if len(self.performance_tracker[model_name]) > 100:
                    self.performance_tracker[model_name] = self.performance_tracker[model_name][-100:]
        
        # Update individual models
        self._update_individual_models(actual_result, profit_loss, position_size, models_used)
        
        # Record trade history
        self.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_result.get('prediction'),
            'actual_result': actual_result,
            'profit_loss': profit_loss,
            'position_size': position_size,
            'models_used': models_used,
            'confidence': prediction_result.get('confidence'),
            'method': prediction_result.get('method')
        })
        
        logger.info(f"📊 Trade result updated: {actual_result} "
                        f"P&L={profit_loss:.2f} Models={models_used}")
    
    def _update_individual_models(self, result: str, profit_loss: float, 
                                position_size: float, models_used: List[str]):
        """Update individual models with trade results"""
        
        # Update multi-strategy orchestrator
        if 'multi_strategy' in self.models and 'multi_strategy' in models_used:
            try:
                strategy = 'momentum'  # Default strategy, should get from prediction
                self.models['multi_strategy'].update_trade_result(
                    result, profit_loss, position_size, strategy
                )
            except Exception as e:
                logger.error(f"Error updating multi-strategy: {e}")
        
        # Update RL model
        if 'rl' in self.models and 'rl' in models_used:
            try:
                # Convert result to reward
                reward = profit_loss / max(position_size, 1.0) if position_size > 0 else 0
                # Note: Would need state and action from prediction to properly update RL
                logger.info(f"RL reward: {reward}")
            except Exception as e:
                logger.error(f"Error updating RL model: {e}")
    
    def _record_prediction(self, prediction: Dict[str, Any]):
        """Record prediction for analysis"""
        # This could be expanded to store predictions for later analysis
        pass
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all AI models"""
        
        status: Dict[str, Any] = {
            'available_models': list(self.models.keys()),
            'features_enabled': self.features,
            'model_performance': {},
            'total_trades': len(self.trade_history),
            'recent_accuracy': self._calculate_recent_accuracy()
        }
        
        # Get performance for each model
        for model_name, performance in self.performance_tracker.items():
            if performance:
                wins = len([p for p in performance if p > 0])
                status['model_performance'][model_name] = {
                    'total_predictions': len(performance),
                    'win_rate': wins / len(performance),
                    'average_return': np.mean(performance),
                    'recent_performance': np.mean(performance[-10:]) if len(performance) >= 10 else 0
                }
        
        # Get individual model status
        if 'multi_strategy' in self.models:
            try:
                status['multi_strategy_status'] = self.models['multi_strategy'].get_comprehensive_status()
            except Exception as e:
                logger.error(f"Error getting multi-strategy status: {e}")
        
        return status
    
    def _calculate_recent_accuracy(self) -> float:
        """Calculate recent prediction accuracy"""
        recent_trades = self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
        
        if not recent_trades:
            return 0.0
        
        correct_predictions = 0
        for trade in recent_trades:
            predicted = trade['prediction']
            profit_loss = trade['profit_loss']
            
            # Consider prediction correct if direction matches (simplified)
            if (predicted == 'BUY' and profit_loss > 0) or \
               (predicted == 'SELL' and profit_loss > 0) or \
               (predicted == 'HOLD' and abs(profit_loss) < 0.01):
                correct_predictions += 1
        
        return correct_predictions / len(recent_trades)
    
    def run_backtest(self, historical_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run backtest on historical data"""
        
        if 'backtesting' not in self.models:
            return {'error': 'Backtesting engine not available'}
        
        try:
            # Use provided data or generate sample data
            if historical_data is None:
                historical_data = self._generate_sample_data()
            
            backtest_results = self.models['backtesting'].run_backtest(
                historical_data, self.get_trading_prediction
            )
            
            return backtest_results
        
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return {'error': str(e)}
    
    def _generate_sample_data(self) -> List[Dict[str, Any]]:
        """Generate sample historical data for testing"""
        np.random.seed(42)
        data: List[Dict[str, Any]] = []
        price = 100.0
        
        for i in range(1000):
            # Random walk with trend
            change = np.random.normal(0.001, 0.01)
            price *= (1 + change)
            
            # Generate indicators
            data.append({
                'timestamp': i,
                'price': price,
                'rsi': 50 + 30 * np.sin(i / 50) + np.random.normal(0, 5),
                'macd': np.random.normal(0, 0.5),
                'ema_fast': price * (1 + np.random.normal(0, 0.001)),
                'ema_slow': price * (1 + np.random.normal(0, 0.002)),
                'volume': np.random.uniform(0.5, 2.0)
            })
        
        return data

# Example usage
if __name__ == "__main__":
    # Test AI model manager
    manager = AIModelManager(initial_balance=1000)
    
    # Get model status
    status = manager.get_model_status()
    print(f"Available models: {status['available_models']}")
    
    # Test prediction
    test_indicators = {
        'price': 1.1234,
        'rsi': 65.5,
        'macd': 0.0012,
        'ema_fast': 1.1240,
        'ema_slow': 1.1220
    }
    
    prediction = manager.get_trading_prediction(
        test_indicators, 
        [1.12, 1.121, 1.123, 1.1234], 
        base_trade_amount=10.0
    )
    
    print(f"Prediction: {prediction['prediction']}")
    print(f"Confidence: {prediction['confidence']:.2f}")
    print(f"Position Size: {prediction['position_size']}")
    print(f"Models Used: {prediction['models_used']}")
    print(f"Risk Level: {prediction['risk_assessment']['risk_level']}")
