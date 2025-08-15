#!/usr/bin/env python3
"""
DEMO SIMULATOR - Immediate Trading Demo
Works without API tokens - shows full bot functionality
"""

import asyncio
import random
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np

# Import our enhanced systems
from ai_integration_system import AIModelManager
from dynamic_risk_management import DynamicRiskManager
from telegram_bot import NotificationManager
from utils import logger

class DemoTradingSimulator:
    """Complete trading simulation - works without API"""
    
    def __init__(self):
        self.balance = 1000.0
        self.trades_completed = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
        self.running = True
        
        # Enhanced AI systems
        self.ai_manager = AIModelManager(initial_balance=self.balance)
        self.risk_manager = DynamicRiskManager(initial_balance=self.balance)
        self.notifier = NotificationManager()
        
        # Market simulation
        self.current_price = 1000.0
        self.price_trend = 0.0
        self.volatility = 0.002
        self.price_history: List[float] = []
        
        logger.info("🎮 Demo Trading Simulator initialized")
    
    def generate_realistic_price(self) -> float:
        """Generate realistic price movements"""
        # Add some trending behavior
        trend_change = random.uniform(-0.001, 0.001)
        self.price_trend = np.clip(self.price_trend + trend_change, -0.01, 0.01)
        
        # Generate price with trend and noise
        price_change = self.price_trend + random.gauss(0, self.volatility)
        self.current_price *= (1 + price_change)
        
        # Keep price in reasonable range
        self.current_price = max(900, min(1100, self.current_price))
        
        return self.current_price
    
    def create_market_indicators(self) -> Dict[str, Any]:
        """Create realistic market indicators"""
        if len(self.price_history) < 20:
            # Simple indicators for early prices
            return {
                'price': self.current_price,
                'rsi': 50.0,
                'macd': 0.0,
                'signal': 0.0,
                'sma_20': self.current_price,
                'ema_12': self.current_price,
                'ema_26': self.current_price,
                'bb_upper': self.current_price * 1.02,
                'bb_lower': self.current_price * 0.98,
                'bb_middle': self.current_price,
                'volume': random.randint(1000, 5000),
                'timestamp': int(time.time())
            }
        
        # Calculate more realistic indicators
        recent_prices = self.price_history[-20:]
        
        # RSI calculation (simplified)
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        gains = [change if change > 0 else 0 for change in price_changes]
        losses = [-change if change < 0 else 0 for change in price_changes]
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0.001
        rs = avg_gain / avg_loss if avg_loss > 0 else 50
        rsi = 100 - (100 / (1 + rs))
        
        # MACD calculation (simplified)
        ema_12 = np.mean(recent_prices[-12:])
        ema_26 = np.mean(recent_prices[-20:])
        macd = ema_12 - ema_26
        signal = macd * 0.8  # Simplified signal line
        
        # Bollinger Bands
        sma_20 = np.mean(recent_prices)
        std_dev = np.std(recent_prices)
        bb_upper = sma_20 + (2 * std_dev)
        bb_lower = sma_20 - (2 * std_dev)
        
        return {
            'price': self.current_price,
            'rsi': rsi,
            'macd': macd,
            'signal': signal,
            'sma_20': sma_20,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_middle': sma_20,
            'volume': random.randint(1000, 5000),
            'timestamp': int(time.time())
        }
    
    async def simulate_trade(self, prediction: Dict[str, Any], amount: float) -> Dict[str, Any]:
        """Simulate a complete trade with realistic outcome"""
        action = prediction['prediction']
        confidence = prediction['confidence']
        
        # Log trade start
        logger.info(f"🎯 DEMO TRADE: {action} ${amount:.2f} (Confidence: {confidence:.1%})")
        
        # Send trade notification
        await self.notifier.notify_trade({
            "action": action,
            "symbol": "R_75_SIM",
            "amount": amount,
            "price": self.current_price,
            "confidence": confidence,
            "reason": prediction.get('reason', 'AI prediction'),
            "strategy": "Demo Mode",
            "market_condition": "simulated"
        })
        
        # Simulate trade duration (1-3 seconds)
        trade_duration = random.uniform(1, 3)
        await asyncio.sleep(trade_duration)
        
        # Generate price movement during trade
        entry_price = self.current_price
        for _ in range(int(trade_duration * 10)):  # 10 ticks per second
            self.generate_realistic_price()
            self.price_history.append(self.current_price)
        
        exit_price = self.current_price
        price_change = exit_price - entry_price
        
        # Determine win/loss based on prediction and actual movement
        if action == "BUY":
            predicted_correctly = price_change > 0
        else:  # SELL
            predicted_correctly = price_change < 0
        
        # Add confidence-based probability adjustment
        # Higher confidence should lead to better results
        confidence_boost = (confidence - 0.5) * 0.4  # Max 20% boost/penalty
        win_probability = 0.6 + confidence_boost  # Base 60% win rate
        
        # Final outcome with some randomness
        is_win = random.random() < win_probability if predicted_correctly else random.random() < (1 - win_probability)
        
        # Calculate profit/loss
        if is_win:
            payout = amount * 1.85  # 85% profit typical for binary options
            profit_loss = payout - amount
            result = "WIN"
            self.wins += 1
        else:
            payout = 0
            profit_loss = -amount
            result = "LOSS"
            self.losses += 1
        
        # Update balance and stats
        self.balance += profit_loss
        self.total_profit += profit_loss
        self.trades_completed += 1
        
        # Create trade result
        trade_result = {
            "action": action,
            "amount": amount,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "price_change": price_change,
            "result": result,
            "profit_loss": profit_loss,
            "payout": payout,
            "confidence": confidence,
            "duration": trade_duration,
            "balance": self.balance
        }
        
        # Send result notification
        await self.notifier.notify_result({
            "symbol": "R_75_SIM",
            "result": result,
            "profit_loss": profit_loss,
            "duration": int(trade_duration)
        })
        
        # Log result
        logger.info(f"📊 DEMO RESULT: {result} {profit_loss:+.2f} (Balance: ${self.balance:.2f})")
        
        return trade_result
    
    async def run_demo_session(self, num_trades: int = 20) -> None:
        """Run a complete demo trading session"""
        print("🎮 STARTING DEMO TRADING SESSION")
        print("=" * 50)
        print(f"💰 Starting Balance: ${self.balance:.2f}")
        print(f"🎯 Target Trades: {num_trades}")
        print(f"🤖 AI Systems: ACTIVE")
        print("=" * 50)
        
        # Send startup notification
        await self.notifier.telegram.send_startup_message()
        
        session_start = datetime.now()
        trades_data = []
        
        try:
            for trade_num in range(1, num_trades + 1):
                print(f"\n📈 Trade {trade_num}/{num_trades}")
                print("-" * 30)
                
                # Generate new price and indicators
                new_price = self.generate_realistic_price()
                self.price_history.append(new_price)
                
                # Keep reasonable history size
                if len(self.price_history) > 200:
                    self.price_history = self.price_history[-200:]
                
                # Create market indicators
                indicators = self.create_market_indicators()
                
                # Get AI prediction
                prediction = self.ai_manager.get_trading_prediction(
                    indicators,
                    self.price_history[-50:],  # Last 50 prices
                    10.0  # Base amount
                )
                
                # Apply dynamic risk management (simplified for demo)
                risk_amount = min(20.0, max(5.0, 10.0 * prediction['confidence']))
                
                print(f"💲 Price: ${new_price:.2f}")
                print(f"🧠 AI Prediction: {prediction['prediction']} (Confidence: {prediction['confidence']:.1%})")
                print(f"💰 Position Size: ${risk_amount:.2f}")
                print(f"📊 RSI: {indicators['rsi']:.1f}")
                
                # Only trade if confidence meets threshold
                if prediction['confidence'] >= 0.65:  # 65% confidence threshold
                    trade_result = await self.simulate_trade(prediction, risk_amount)
                    trades_data.append(trade_result)
                    
                    # Show current performance
                    win_rate = (self.wins / self.trades_completed) * 100 if self.trades_completed > 0 else 0
                    print(f"🎯 Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
                    print(f"💵 Profit/Loss: ${self.total_profit:+.2f}")
                    
                    # Brief pause between trades
                    await asyncio.sleep(2)
                else:
                    print(f"⏭️ Skipped - Low confidence ({prediction['confidence']:.1%})")
                    await asyncio.sleep(1)
                
                # Stop if balance gets too low
                if self.balance < 50:
                    print("⚠️ Low balance - stopping demo")
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️ Demo stopped by user")
        
        # Session summary
        session_duration = datetime.now() - session_start
        win_rate = (self.wins / self.trades_completed) * 100 if self.trades_completed > 0 else 0
        roi = (self.total_profit / 1000.0) * 100
        
        print("\n" + "=" * 50)
        print("📊 DEMO SESSION SUMMARY")
        print("=" * 50)
        print(f"⏱️ Duration: {str(session_duration).split('.')[0]}")
        print(f"📈 Trades Completed: {self.trades_completed}")
        print(f"✅ Wins: {self.wins}")
        print(f"❌ Losses: {self.losses}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"💰 Starting Balance: $1000.00")
        print(f"💵 Final Balance: ${self.balance:.2f}")
        print(f"📊 Total Profit/Loss: ${self.total_profit:+.2f}")
        print(f"📈 ROI: {roi:+.1f}%")
        print("=" * 50)
        
        # Send session summary
        await self.notifier.notify_daily_summary({
            'total_trades': self.trades_completed,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate / 100,
            'total_profit': self.total_profit,
            'final_balance': self.balance,
            'roi_percent': roi,
            'session_duration': str(session_duration).split('.')[0]
        })
        
        # Performance grade
        if win_rate >= 70 and roi > 5:
            grade = "A+ (Excellent)"
        elif win_rate >= 60 and roi > 2:
            grade = "A (Good)"
        elif win_rate >= 50 and roi > 0:
            grade = "B (Average)"
        else:
            grade = "C (Needs Improvement)"
        
        print(f"🏆 Performance Grade: {grade}")
        print("\n🎉 DEMO COMPLETE! Ready for live trading once API is fixed.")

async def main():
    """Run the demo"""
    simulator = DemoTradingSimulator()
    
    print("🚀 DERIV TRADING BOT - DEMO MODE")
    print("🎮 Full functionality without API tokens!")
    print("📱 Telegram notifications, AI predictions, risk management")
    print()
    
    # Ask user for demo length
    try:
        num_trades = int(input("Enter number of trades to simulate (default 15): ") or "15")
        num_trades = max(5, min(50, num_trades))  # Limit between 5-50
    except:
        num_trades = 15
    
    await simulator.run_demo_session(num_trades)

if __name__ == "__main__":
    asyncio.run(main())
