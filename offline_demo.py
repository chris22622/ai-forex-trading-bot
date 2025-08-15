"""
Offline Demo Mode for Deriv Trading Bot
Tests bot functionality without requiring live connection
"""

import asyncio
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Import bot components
from safe_logger import get_safe_logger
from indicators import TechnicalIndicators
from ai_model import TradingAI
from telegram_bot import NotificationManager
from utils import trade_logger, performance_tracker

logger = get_safe_logger(__name__)

class OfflineDemoBot:
    """Offline demo version of the trading bot"""
    
    def __init__(self):
        self.running = False
        self.current_balance = 1000.0
        self.active_trades: Dict[str, Any] = {}
        self.price_history: List[float] = []
        self.trades_today = 0
        self.session_stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0
        }
        
        # Initialize components
        self.indicators = TechnicalIndicators()
        self.ai_model = TradingAI()
        self.notifier = NotificationManager()
        
        logger.info("🚀 Offline Demo Bot initialized")
        logger.info(f"💰 Starting balance: ${self.current_balance:.2f}")
    
    def generate_mock_price(self, base_price: float = 100.0) -> float:
        """Generate realistic price movements"""
        if not self.price_history:
            return base_price
        
        last_price = self.price_history[-1]
        # Small random movements typical of volatility indices
        change_percent = random.uniform(-0.5, 0.5) / 100  # ±0.5%
        new_price = last_price * (1 + change_percent)
        
        # Add some trend and volatility
        trend = np.sin(len(self.price_history) * 0.01) * 0.001
        volatility = random.uniform(-0.002, 0.002)
        
        new_price *= (1 + trend + volatility)
        return round(new_price, 5)
    
    async def simulate_trade(self, action: str, amount: float) -> Dict[str, Any]:
        """Simulate a trade with realistic outcomes"""
        contract_id = f"demo_{int(time.time())}_{random.randint(1000, 9999)}"
        entry_price = self.price_history[-1] if self.price_history else 100.0
        
        # Simulate trade duration (1-3 seconds)
        duration = random.uniform(1, 3)
        await asyncio.sleep(duration)
        
        # Generate exit price
        exit_price = self.generate_mock_price()
        
        # Determine win/loss based on direction
        price_change = exit_price - entry_price
        if action == "BUY":
            is_win = price_change > 0
        else:  # SELL
            is_win = price_change < 0
        
        # Calculate payout (85% profit typical for binary options)
        if is_win:
            payout = amount * 1.85
            profit_loss = payout - amount
        else:
            payout = 0
            profit_loss = -amount
        
        # Update balance and stats
        self.current_balance += profit_loss
        self.session_stats['total_trades'] += 1
        if is_win:
            self.session_stats['wins'] += 1
        else:
            self.session_stats['losses'] += 1
        self.session_stats['total_profit'] += profit_loss
        
        result = {
            "contract_id": contract_id,
            "action": action,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "amount": amount,
            "result": "WIN" if is_win else "LOSS",
            "profit_loss": profit_loss,
            "duration": duration,
            "balance": self.current_balance
        }
        
        # Log the trade
        trade_record = {
            "trade_id": contract_id,
            "action": action,
            "symbol": "R_75_DEMO",
            "amount": amount,
            "price": entry_price,
            "confidence": 0.75,
            "reason": "Demo AI prediction",
            "duration": int(duration),
            "result": "WIN" if is_win else "LOSS",
            "profit_loss": profit_loss,
            "balance": self.current_balance
        }
        
        trade_logger.log_trade(trade_record)
        
        # Log result
        emoji = "✅" if is_win else "❌"
        logger.info(f"{emoji} Trade: {action} ${amount:.2f} | "
                   f"Entry: {entry_price:.5f} Exit: {exit_price:.5f} | "
                   f"Result: {profit_loss:+.2f} | Balance: ${self.current_balance:.2f}")
        
        return result
    
    async def run_demo_session(self, duration_minutes: int = 5):
        """Run a demo trading session"""
        logger.info(f"🎯 Starting {duration_minutes}-minute demo session...")
        
        self.running = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Initialize with some price history
        base_price = 100.0
        for i in range(50):
            price = self.generate_mock_price(base_price + i * 0.001)
            self.price_history.append(price)
            self.indicators.update_price_data(price, int(time.time() + i))
        
        try:
            while self.running and time.time() < end_time:
                # Generate new price tick
                new_price = self.generate_mock_price()
                self.price_history.append(new_price)
                
                # Keep price history manageable
                if len(self.price_history) > 1000:
                    self.price_history = self.price_history[-500:]
                
                # Update indicators
                self.indicators.update_price_data(new_price, int(time.time()))
                
                # Get AI prediction every 10 ticks
                if len(self.price_history) % 10 == 0:
                    try:
                        indicators_summary = self.indicators.get_indicator_summary()
                        if indicators_summary:
                            ai_prediction = self.ai_model.predict_trade(
                                indicators_summary,
                                self.price_history[-50:]
                            )
                            
                            # Trade if confidence is high enough
                            if (ai_prediction['prediction'] in ['BUY', 'SELL'] and 
                                ai_prediction['confidence'] >= 0.7):
                                
                                await self.simulate_trade(
                                    ai_prediction['prediction'],
                                    1.0  # $1 per trade
                                )
                                
                                # Don't trade too frequently
                                await asyncio.sleep(2)
                    
                    except Exception as e:
                        logger.error(f"Error in demo trading logic: {e}")
                
                # Small delay between ticks
                await asyncio.sleep(0.5)
        
        except KeyboardInterrupt:
            logger.info("⏹️ Demo stopped by user")
        
        finally:
            await self.stop_demo()
    
    async def stop_demo(self):
        """Stop the demo and show results"""
        self.running = False
        
        # Calculate session summary
        win_rate = (self.session_stats['wins'] / max(1, self.session_stats['total_trades'])) * 100
        roi = ((self.current_balance - 1000.0) / 1000.0) * 100
        
        logger.info("=" * 60)
        logger.info("📊 DEMO SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"💰 Starting Balance: $1000.00")
        logger.info(f"💵 Final Balance: ${self.current_balance:.2f}")
        logger.info(f"📈 Total Profit/Loss: ${self.session_stats['total_profit']:+.2f}")
        logger.info(f"📊 ROI: {roi:+.1f}%")
        logger.info(f"🎯 Total Trades: {self.session_stats['total_trades']}")
        logger.info(f"✅ Wins: {self.session_stats['wins']}")
        logger.info(f"❌ Losses: {self.session_stats['losses']}")
        logger.info(f"🏆 Win Rate: {win_rate:.1f}%")
        logger.info("=" * 60)
        
        # Send Telegram summary if configured
        try:
            await self.notifier.telegram.send_message(
                f"🚀 *DEMO SESSION COMPLETE*\n\n"
                f"💰 *Balance:* ${self.current_balance:.2f}\n"
                f"📈 *P/L:* ${self.session_stats['total_profit']:+.2f}\n"
                f"📊 *ROI:* {roi:+.1f}%\n"
                f"🎯 *Trades:* {self.session_stats['total_trades']}\n"
                f"🏆 *Win Rate:* {win_rate:.1f}%"
            )
        except Exception as e:
            logger.info(f"💬 Telegram summary not sent: {e}")

async def main():
    """Run the offline demo"""
    print("=" * 60)
    print("🎮 DERIV TRADING BOT - OFFLINE DEMO")
    print("🤖 Testing AI Trading Logic Without Live Connection")
    print("=" * 60)
    print("📝 This demo simulates:")
    print("   • Real price movements")
    print("   • AI trading decisions")
    print("   • Trade execution and results")
    print("   • Performance tracking")
    print("=" * 60)
    
    demo_bot = OfflineDemoBot()
    
    try:
        # Run 5-minute demo session
        await demo_bot.run_demo_session(duration_minutes=5)
    except Exception as e:
        logger.error(f"Demo error: {e}")
    
    logger.info("✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
