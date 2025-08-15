"""
Dynamic Risk Management and Multi-Strategy System
Includes adaptive position sizing, strategy rotation, and market regime detection
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
import logging
from enum import Enum

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down" 
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class TradingStrategy(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    TREND_FOLLOWING = "trend_following"
    CONTRARIAN = "contrarian"

class MarketRegimeDetector:
    """Detects current market regime for strategy adaptation"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.price_history: deque[float] = deque(maxlen=lookback_period * 2)
        self.volume_history: deque[float] = deque(maxlen=lookback_period)
        self.volatility_history: deque[float] = deque(maxlen=lookback_period)
        
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        self.regime_history: List[Dict[str, Any]] = []
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('RegimeDetector')
        logger.setLevel(logging.INFO)
        return logger
    
    def update_market_data(self, price: float, volume: Optional[float] = None, timestamp: Optional[int] = None):
        """Update market data for regime detection"""
        self.price_history.append(price)
        
        if volume is not None:
            self.volume_history.append(volume)
        
        # Calculate volatility
        if len(self.price_history) >= 20:
            returns = np.diff(list(self.price_history)[-20:]) / np.array(list(self.price_history)[-20:-1])
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            self.volatility_history.append(volatility)
        
        # Detect regime
        self._detect_regime()
    
    def _detect_regime(self) -> MarketRegime:
        """Detect current market regime"""
        if len(self.price_history) < self.lookback_period:
            return self.current_regime
        
        prices = np.array(list(self.price_history))
        
        # Calculate trend indicators
        short_ma = np.mean(prices[-20:])  # 20-period MA
        long_ma = np.mean(prices[-50:])   # 50-period MA
        
        # Calculate volatility metrics
        recent_volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
        historical_volatility = np.mean(list(self.volatility_history)[-20:]) if self.volatility_history else recent_volatility
        
        # Calculate price momentum
        price_momentum = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        # Calculate range metrics (for future use in additional regime detection)
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        
        # Regime detection logic
        regime_scores: Dict[MarketRegime, float] = {
            MarketRegime.TRENDING_UP: 0.0,
            MarketRegime.TRENDING_DOWN: 0.0,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.HIGH_VOLATILITY: 0.0,
            MarketRegime.LOW_VOLATILITY: 0.0,
            MarketRegime.BREAKOUT: 0.0,
            MarketRegime.REVERSAL: 0.0
        }
        
        # Trend detection
        if short_ma > long_ma * 1.002 and price_momentum > 0.01:
            regime_scores[MarketRegime.TRENDING_UP] += 0.8
        elif short_ma < long_ma * 0.998 and price_momentum < -0.01:
            regime_scores[MarketRegime.TRENDING_DOWN] += 0.8
        else:
            regime_scores[MarketRegime.SIDEWAYS] += 0.6
        
        # Volatility detection
        if recent_volatility > historical_volatility * 1.5:
            regime_scores[MarketRegime.HIGH_VOLATILITY] += 0.7
        elif recent_volatility < historical_volatility * 0.7:
            regime_scores[MarketRegime.LOW_VOLATILITY] += 0.7
        
        # Breakout detection
        if prices[-1] > recent_high * 0.999 or prices[-1] < recent_low * 1.001:
            if recent_volatility > historical_volatility * 1.2:
                regime_scores[MarketRegime.BREAKOUT] += 0.8
        
        # Reversal detection
        if len(prices) >= 10:
            recent_trend = np.polyfit(range(10), prices[-10:], 1)[0]
            older_trend = np.polyfit(range(10), prices[-20:-10], 1)[0] if len(prices) >= 20 else recent_trend
            
            if (recent_trend > 0 and older_trend < 0) or (recent_trend < 0 and older_trend > 0):
                regime_scores[MarketRegime.REVERSAL] += 0.7
        
        # Determine primary regime
        primary_regime = max(regime_scores.keys(), key=lambda k: regime_scores[k])
        confidence = regime_scores[primary_regime]
        
        # Update if confidence is high enough
        if confidence > 0.6:
            self.current_regime = primary_regime
            self.regime_confidence = confidence
            
            # Log regime change
            self.regime_history.append({
                'timestamp': datetime.now().isoformat(),
                'regime': primary_regime.value,
                'confidence': confidence,
                'price': prices[-1],
                'volatility': recent_volatility,
                'momentum': price_momentum
            })
            
            # Keep only recent history
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
        
        return self.current_regime
    
    def get_regime_info(self) -> Dict[str, Any]:
        """Get current regime information"""
        return {
            'current_regime': self.current_regime.value,
            'confidence': self.regime_confidence,
            'regime_duration': self._get_regime_duration(),
            'recent_regimes': [r['regime'] for r in self.regime_history[-5:]],
            'market_metrics': self._get_market_metrics()
        }
    
    def _get_regime_duration(self) -> int:
        """Get duration of current regime in periods"""
        if not self.regime_history:
            return 0
        
        duration = 0
        current_regime = self.current_regime.value
        
        for entry in reversed(self.regime_history):
            if entry['regime'] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _get_market_metrics(self) -> Dict[str, float]:
        """Get current market metrics"""
        if len(self.price_history) < 20:
            return {}
        
        prices = np.array(list(self.price_history)[-20:])
        returns = np.diff(prices) / prices[:-1]
        
        return {
            'volatility': float(np.std(returns)),
            'momentum': float((prices[-1] - prices[0]) / prices[0]),
            'trend_strength': float(abs(np.polyfit(range(len(prices)), prices, 1)[0])),
            'range_efficiency': float((prices[-1] - prices[0]) / (np.max(prices) - np.min(prices))) if np.max(prices) != np.min(prices) else 0.0
        }

class DynamicRiskManager:
    """Dynamic risk management with adaptive position sizing"""
    
    def __init__(self, initial_balance: float = 1000.0, max_risk_per_trade: float = 0.02):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = max_risk_per_trade
        
        # Risk tracking
        self.daily_risk_used = 0.0
        self.max_daily_risk = 0.1  # 10% max daily risk
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.peak_balance = initial_balance
        self.current_drawdown = 0.0
        
        # Dynamic parameters
        self.risk_multiplier = 1.0
        self.trade_cooldown = 0
        self.emergency_stop = False
        
        # History tracking
        self.risk_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('RiskManager')
        logger.setLevel(logging.INFO)
        return logger
    
    def calculate_position_size(self, confidence: float, market_regime: MarketRegime, 
                              base_amount: float, volatility: float = 0.01) -> float:
        """Calculate dynamic position size based on multiple factors"""
        
        if self.emergency_stop:
            self.logger.warning("🚨 Emergency stop active - no new positions")
            return 0.0
        
        if self.trade_cooldown > 0:
            self.logger.info(f"⏰ Trade cooldown active: {self.trade_cooldown} periods remaining")
            return 0.0
        
        # Base risk amount
        risk_amount = self.current_balance * self.max_risk_per_trade
        
        # Confidence adjustment
        confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5x to 2.0x
        
        # Market regime adjustment
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        # Volatility adjustment
        volatility_multiplier = max(0.5, min(2.0, 1.0 / (1.0 + volatility * 10)))
        
        # Performance-based adjustment
        performance_multiplier = self._get_performance_multiplier()
        
        # Drawdown adjustment
        drawdown_multiplier = max(0.3, 1.0 - self.current_drawdown * 2)
        
        # Calculate final position size
        final_multiplier = (confidence_multiplier * regime_multiplier * 
                          volatility_multiplier * performance_multiplier * 
                          drawdown_multiplier * self.risk_multiplier)
        
        position_size = min(base_amount * final_multiplier, risk_amount)
        
        # Check daily risk limits
        if self.daily_risk_used + position_size > self.current_balance * self.max_daily_risk:
            remaining_daily_risk = max(0, self.current_balance * self.max_daily_risk - self.daily_risk_used)
            position_size = min(position_size, remaining_daily_risk)
        
        # Minimum position size check
        if position_size < base_amount * 0.1:
            position_size = 0.0
        
        # Log risk decision
        self.risk_history.append({
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'regime': market_regime.value,
            'volatility': volatility,
            'base_amount': base_amount,
            'final_size': position_size,
            'multipliers': {
                'confidence': confidence_multiplier,
                'regime': regime_multiplier,
                'volatility': volatility_multiplier,
                'performance': performance_multiplier,
                'drawdown': drawdown_multiplier,
                'risk': self.risk_multiplier
            }
        })
        
        return round(position_size, 2)
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get position size multiplier based on market regime"""
        regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.3,
            MarketRegime.TRENDING_DOWN: 1.3,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.7,
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.BREAKOUT: 1.5,
            MarketRegime.REVERSAL: 0.9
        }
        return regime_multipliers.get(regime, 1.0)
    
    def _get_performance_multiplier(self) -> float:
        """Get multiplier based on recent performance"""
        
        # Consecutive losses adjustment
        if self.consecutive_losses >= 3:
            loss_penalty = 0.5 ** (self.consecutive_losses - 2)  # Exponential decay
            return max(0.2, loss_penalty)
        
        # Consecutive wins adjustment
        if self.consecutive_wins >= 3:
            win_bonus = min(1.5, 1.0 + (self.consecutive_wins - 2) * 0.1)
            return win_bonus
        
        return 1.0
    
    def update_trade_result(self, trade_amount: float, profit_loss: float, 
                          trade_result: str, market_regime: MarketRegime):
        """Update risk parameters based on trade result"""
        
        # Update balance
        self.current_balance += profit_loss
        self.daily_risk_used += trade_amount
        
        # Update consecutive counters
        if profit_loss > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Update peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        self.current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        # Dynamic risk adjustment
        self._adjust_risk_parameters(profit_loss, trade_result, market_regime)
        
        # Emergency stop check
        self._check_emergency_conditions()
        
        # Store trade history
        self.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'amount': trade_amount,
            'profit_loss': profit_loss,
            'result': trade_result,
            'balance': self.current_balance,
            'drawdown': self.current_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'regime': market_regime.value
        })
        
        self.logger.info(f"💰 Risk Update: Balance=${self.current_balance:.2f}, "
                        f"DD={self.current_drawdown:.1%}, "
                        f"Consecutive: {self.consecutive_losses}L/{self.consecutive_wins}W")
    
    def _adjust_risk_parameters(self, profit_loss: float, result: str, regime: MarketRegime):
        """Dynamically adjust risk parameters"""
        
        # Adjust risk multiplier based on performance
        if profit_loss > 0:
            # Increase risk on wins (but cap it)
            self.risk_multiplier = min(2.0, self.risk_multiplier * 1.05)
        else:
            # Decrease risk on losses
            self.risk_multiplier = max(0.3, self.risk_multiplier * 0.95)
        
        # Regime-specific adjustments
        if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.REVERSAL]:
            self.risk_multiplier *= 0.9  # Be more conservative
        
        # Set cooldown after losses
        if self.consecutive_losses >= 2:
            self.trade_cooldown = min(5, self.consecutive_losses)
        
        # Reduce cooldown over time
        if self.trade_cooldown > 0:
            self.trade_cooldown -= 1
    
    def _check_emergency_conditions(self):
        """Check if emergency stop should be activated"""
        
        # Maximum drawdown stop
        if self.current_drawdown >= 0.15:  # 15% max drawdown
            self.emergency_stop = True
            self.logger.error(f"🚨 EMERGENCY STOP: Max drawdown reached ({self.current_drawdown:.1%})")
        
        # Daily loss limit
        daily_loss = self.daily_risk_used - sum(t['profit_loss'] for t in self.trade_history[-10:] if t['profit_loss'] > 0)
        if daily_loss > self.current_balance * self.max_daily_risk:
            self.emergency_stop = True
            self.logger.error(f"🚨 EMERGENCY STOP: Daily loss limit exceeded")
        
        # Consecutive losses stop
        if self.consecutive_losses >= 5:
            self.emergency_stop = True
            self.logger.error(f"🚨 EMERGENCY STOP: Too many consecutive losses ({self.consecutive_losses})")
    
    def reset_daily_limits(self):
        """Reset daily risk limits (call at start of new day)"""
        self.daily_risk_used = 0.0
        
        # Partial emergency stop reset (if not due to severe drawdown)
        if self.emergency_stop and self.current_drawdown < 0.1:
            self.emergency_stop = False
            self.logger.info("✅ Emergency stop reset for new day")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk management metrics"""
        return {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': self.current_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'risk_multiplier': self.risk_multiplier,
            'daily_risk_used': self.daily_risk_used,
            'emergency_stop': self.emergency_stop,
            'trade_cooldown': self.trade_cooldown,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance
        }

class StrategySelector:
    """Selects optimal trading strategy based on market conditions"""
    
    def __init__(self):
        self.strategy_performance: Dict[TradingStrategy, List[float]] = {
            strategy: [] for strategy in TradingStrategy
        }
        
        self.regime_strategy_mapping = {
            MarketRegime.TRENDING_UP: [TradingStrategy.MOMENTUM, TradingStrategy.TREND_FOLLOWING],
            MarketRegime.TRENDING_DOWN: [TradingStrategy.MOMENTUM, TradingStrategy.TREND_FOLLOWING],
            MarketRegime.SIDEWAYS: [TradingStrategy.MEAN_REVERSION, TradingStrategy.SCALPING],
            MarketRegime.HIGH_VOLATILITY: [TradingStrategy.BREAKOUT, TradingStrategy.SCALPING],
            MarketRegime.LOW_VOLATILITY: [TradingStrategy.MEAN_REVERSION, TradingStrategy.TREND_FOLLOWING],
            MarketRegime.BREAKOUT: [TradingStrategy.BREAKOUT, TradingStrategy.MOMENTUM],
            MarketRegime.REVERSAL: [TradingStrategy.CONTRARIAN, TradingStrategy.MEAN_REVERSION]
        }
        
        self.current_strategy = TradingStrategy.MOMENTUM
        self.strategy_history: List[Dict[str, Any]] = []
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('StrategySelector')
        logger.setLevel(logging.INFO)
        return logger
    
    def select_strategy(self, market_regime: MarketRegime, 
                       regime_confidence: float) -> TradingStrategy:
        """Select best strategy for current market regime"""
        
        # Get suitable strategies for current regime
        suitable_strategies = self.regime_strategy_mapping.get(market_regime, [TradingStrategy.MOMENTUM])
        
        # If regime confidence is low, stick with current strategy
        if regime_confidence < 0.7:
            if self.current_strategy in suitable_strategies:
                return self.current_strategy
        
        # Select best performing strategy from suitable ones
        best_strategy = self.current_strategy
        best_performance = self._get_strategy_performance(self.current_strategy)
        
        for strategy in suitable_strategies:
            performance = self._get_strategy_performance(strategy)
            if performance > best_performance:
                best_strategy = strategy
                best_performance = performance
        
        # Only change strategy if new one is significantly better
        current_performance = self._get_strategy_performance(self.current_strategy)
        if best_performance > current_performance * 1.1 or regime_confidence > 0.9:
            if best_strategy != self.current_strategy:
                self.logger.info(f"🔄 Strategy change: {self.current_strategy.value} → {best_strategy.value}")
                
                self.strategy_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'old_strategy': self.current_strategy.value,
                    'new_strategy': best_strategy.value,
                    'regime': market_regime.value,
                    'regime_confidence': regime_confidence,
                    'reason': f"Better performance in {market_regime.value}"
                })
                
                self.current_strategy = best_strategy
        
        return self.current_strategy
    
    def _get_strategy_performance(self, strategy: TradingStrategy) -> float:
        """Get recent performance score for a strategy"""
        recent_results = self.strategy_performance[strategy][-20:]  # Last 20 trades
        
        if not recent_results:
            return 0.5  # Neutral performance for untested strategies
        
        # Calculate win rate and average return
        wins = len([r for r in recent_results if r > 0])
        win_rate = wins / len(recent_results)
        avg_return = np.mean(recent_results)
        
        # Combine win rate and average return for performance score
        performance_score = (win_rate * 0.6) + (avg_return * 0.4)
        return max(0.0, min(1.0, float(performance_score + 0.5)))  # Normalize to 0-1
    
    def update_strategy_performance(self, strategy: TradingStrategy, result: float):
        """Update performance tracking for a strategy"""
        self.strategy_performance[strategy].append(result)
        
        # Keep only recent results
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        stats = {}
        
        for strategy in TradingStrategy:
            results = self.strategy_performance[strategy]
            if results:
                wins = len([r for r in results if r > 0])
                stats[strategy.value] = {
                    'total_trades': len(results),
                    'win_rate': wins / len(results),
                    'avg_return': np.mean(results),
                    'recent_performance': self._get_strategy_performance(strategy)
                }
            else:
                stats[strategy.value] = {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'recent_performance': 0.5
                }
        
        return {
            'current_strategy': self.current_strategy.value,
            'strategy_stats': stats,
            'recent_changes': self.strategy_history[-5:]
        }

class MultiStrategyOrchestrator:
    """Orchestrates multiple trading strategies with dynamic selection"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = DynamicRiskManager(initial_balance)
        self.strategy_selector = StrategySelector()
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MultiStrategyOrchestrator')
        logger.setLevel(logging.INFO)
        return logger
    
    def update_market_data(self, price: float, volume: Optional[float] = None, timestamp: Optional[int] = None):
        """Update market data for all components"""
        self.regime_detector.update_market_data(price, volume, timestamp)
    
    def get_trading_recommendation(self, indicators: Dict[str, Any], 
                                 price_history: List[float], 
                                 base_trade_amount: float) -> Dict[str, Any]:
        """Get comprehensive trading recommendation"""
        
        # Get current market regime
        regime_info = self.regime_detector.get_regime_info()
        current_regime = MarketRegime(regime_info['current_regime'])
        regime_confidence = regime_info['confidence']
        
        # Select appropriate strategy
        selected_strategy = self.strategy_selector.select_strategy(current_regime, regime_confidence)
        
        # Get base prediction (this would be replaced with actual strategy implementation)
        base_prediction = self._get_strategy_prediction(selected_strategy, indicators, price_history)
        
        # Calculate dynamic position size
        volatility = regime_info['market_metrics'].get('volatility', 0.01)
        position_size = self.risk_manager.calculate_position_size(
            base_prediction['confidence'], current_regime, base_trade_amount, volatility
        )
        
        # Compile final recommendation
        recommendation: Dict[str, Any] = {
            'prediction': base_prediction['prediction'],
            'confidence': base_prediction['confidence'],
            'position_size': position_size,
            'method': f"MULTI_STRATEGY_{selected_strategy.value.upper()}",
            'strategy': selected_strategy.value,
            'market_regime': current_regime.value,
            'regime_confidence': regime_confidence,
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'base_amount': base_trade_amount,
            'volatility_adj': volatility,
            'reason': f"{selected_strategy.value} strategy in {current_regime.value} market"
        }
        
        # Don't trade if position size is too small
        if position_size < base_trade_amount * 0.1:
            recommendation['prediction'] = 'HOLD'
            recommendation['reason'] += ' (Risk management override)'
        
        return recommendation
    
    def _get_strategy_prediction(self, strategy: TradingStrategy, 
                               indicators: Dict[str, Any], 
                               price_history: List[float]) -> Dict[str, Any]:
        """Get prediction based on selected strategy"""
        
        # This is a simplified implementation - in practice, each strategy would have its own logic
        
        rsi = indicators.get('rsi', 50)
        ema_fast = indicators.get('ema_fast', 0)
        ema_slow = indicators.get('ema_slow', 0)
        price = indicators.get('price', 0)
        
        if strategy == TradingStrategy.MOMENTUM:
            # Momentum strategy
            if rsi > 60 and ema_fast > ema_slow:
                return {'prediction': 'BUY', 'confidence': 0.8}
            elif rsi < 40 and ema_fast < ema_slow:
                return {'prediction': 'SELL', 'confidence': 0.8}
            
        elif strategy == TradingStrategy.MEAN_REVERSION:
            # Mean reversion strategy
            if rsi < 30:
                return {'prediction': 'BUY', 'confidence': 0.7}
            elif rsi > 70:
                return {'prediction': 'SELL', 'confidence': 0.7}
                
        elif strategy == TradingStrategy.BREAKOUT:
            # Breakout strategy
            if len(price_history) >= 20:
                recent_high = max(price_history[-20:])
                recent_low = min(price_history[-20:])
                
                if price > recent_high * 1.001:
                    return {'prediction': 'BUY', 'confidence': 0.9}
                elif price < recent_low * 0.999:
                    return {'prediction': 'SELL', 'confidence': 0.9}
        
        elif strategy == TradingStrategy.TREND_FOLLOWING:
            # Trend following strategy
            if ema_fast > ema_slow * 1.002:
                return {'prediction': 'BUY', 'confidence': 0.7}
            elif ema_fast < ema_slow * 0.998:
                return {'prediction': 'SELL', 'confidence': 0.7}
        
        elif strategy == TradingStrategy.SCALPING:
            # Scalping strategy (quick trades)
            if len(price_history) >= 5:
                recent_momentum = (price_history[-1] - price_history[-5]) / price_history[-5]
                if recent_momentum > 0.001:
                    return {'prediction': 'BUY', 'confidence': 0.6}
                elif recent_momentum < -0.001:
                    return {'prediction': 'SELL', 'confidence': 0.6}
        
        elif strategy == TradingStrategy.CONTRARIAN:
            # Contrarian strategy
            if rsi > 80:
                return {'prediction': 'SELL', 'confidence': 0.8}
            elif rsi < 20:
                return {'prediction': 'BUY', 'confidence': 0.8}
        
        # Default to hold
        return {'prediction': 'HOLD', 'confidence': 0.3}
    
    def update_trade_result(self, trade_result: str, profit_loss: float, 
                          position_size: float, strategy_used: Optional[str] = None):
        """Update all components with trade result"""
        
        # Get current regime
        regime_info = self.regime_detector.get_regime_info()
        current_regime = MarketRegime(regime_info['current_regime'])
        
        # Update risk manager
        self.risk_manager.update_trade_result(position_size, profit_loss, trade_result, current_regime)
        
        # Update strategy performance
        if strategy_used:
            try:
                strategy_enum = TradingStrategy(strategy_used)
                normalized_result = 1.0 if profit_loss > 0 else -1.0
                self.strategy_selector.update_strategy_performance(strategy_enum, normalized_result)
            except ValueError:
                pass  # Unknown strategy
        
        self.logger.info(f"📊 Multi-strategy update: {trade_result} "
                        f"P&L={profit_loss:.2f} in {current_regime.value}")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all components"""
        return {
            'regime_info': self.regime_detector.get_regime_info(),
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'strategy_stats': self.strategy_selector.get_strategy_stats(),
            'emergency_stop': self.risk_manager.emergency_stop,
            'system_health': {
                'regime_detection': len(self.regime_detector.regime_history) > 0,
                'risk_management': len(self.risk_manager.risk_history) > 0,
                'strategy_selection': len(self.strategy_selector.strategy_history) >= 0
            }
        }
    
    def reset_daily_limits(self):
        """Reset daily limits (call at start of new day)"""
        self.risk_manager.reset_daily_limits()

# Example usage
if __name__ == "__main__":
    # Test multi-strategy orchestrator
    orchestrator = MultiStrategyOrchestrator(initial_balance=1000)
    
    # Simulate market data
    np.random.seed(42)
    prices: List[float] = [100.0]
    
    for i in range(100):
        # Generate price movement
        change = np.random.normal(0, 0.01)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
        
        # Update market data
        orchestrator.update_market_data(new_price, volume=np.random.uniform(0.5, 1.5))
        
        # Get recommendation every 10 periods
        if i % 10 == 0:
            indicators: Dict[str, Any] = {
                'price': new_price,
                'rsi': 50 + 20 * np.sin(i / 20) + np.random.normal(0, 5),
                'ema_fast': np.mean(prices[-10:]),
                'ema_slow': np.mean(prices[-20:])
            }
            
            recommendation = orchestrator.get_trading_recommendation(
                indicators, prices[-50:], base_trade_amount=10.0
            )
            
            print(f"Period {i}: {recommendation['prediction']} "
                  f"(Size: {recommendation['position_size']:.2f}, "
                  f"Strategy: {recommendation['strategy']}, "
                  f"Regime: {recommendation['market_regime']})")
            
            # Simulate trade result
            if recommendation['prediction'] != 'HOLD':
                result = "WIN" if np.random.random() > 0.4 else "LOSS"
                profit_loss = recommendation['position_size'] * 0.8 if result == "WIN" else -recommendation['position_size']
                
                orchestrator.update_trade_result(
                    result, profit_loss, recommendation['position_size'], recommendation['strategy']
                )
    
    # Get final status
    status = orchestrator.get_comprehensive_status()
    print(f"\nFinal Status:")
    print(f"Balance: ${status['risk_metrics']['current_balance']:.2f}")
    print(f"Drawdown: {status['risk_metrics']['current_drawdown']:.1%}")
    print(f"Current Strategy: {status['strategy_stats']['current_strategy']}")
    print(f"Current Regime: {status['regime_info']['current_regime']}")
