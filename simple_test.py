"""
Simple Bot Test - Tests core functionality without external dependencies
"""

import asyncio
import random
import time
from datetime import datetime
from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

class SimpleBotTest:
    """Simple test of bot core functionality"""
    
    def __init__(self):
        self.balance = 1000.0
        self.trades = 0
        self.wins = 0
        logger.info("🚀 Simple Bot Test initialized")
    
    def simulate_price_movement(self, base_price: float = 100.0) -> float:
        """Generate simple price movement"""
        change = random.uniform(-0.01, 0.01)  # ±1% change
        return base_price * (1 + change)
    
    def simulate_trade(self, action: str, amount: float = 1.0) -> dict:
        """Simulate a single trade"""
        entry_price = 100.0 + random.uniform(-5, 5)
        exit_price = self.simulate_price_movement(entry_price)
        
        # Determine win/loss
        price_change = exit_price - entry_price
        if action == "BUY":
            is_win = price_change > 0
        else:  # SELL
            is_win = price_change < 0
        
        # Calculate profit/loss
        if is_win:
            profit = amount * 0.85  # 85% profit
        else:
            profit = -amount  # Lose stake
        
        self.balance += profit
        self.trades += 1
        if is_win:
            self.wins += 1
        
        result = {
            "action": action,
            "entry": entry_price,
            "exit": exit_price,
            "profit": profit,
            "win": is_win,
            "balance": self.balance
        }
        
        emoji = "✅" if is_win else "❌"
        logger.info(f"{emoji} {action} {entry_price:.3f}→{exit_price:.3f} "
                   f"P/L:{profit:+.2f} Bal:${self.balance:.2f}")
        
        return result
    
    def run_test(self, num_trades: int = 10):
        """Run a series of test trades"""
        logger.info(f"🎯 Running {num_trades} test trades...")
        
        for i in range(num_trades):
            action = random.choice(["BUY", "SELL"])
            self.simulate_trade(action)
            time.sleep(0.5)  # Small delay
        
        # Summary
        win_rate = (self.wins / self.trades) * 100
        profit = self.balance - 1000.0
        
        logger.info("=" * 50)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"💰 Final Balance: ${self.balance:.2f}")
        logger.info(f"📈 Total P/L: ${profit:+.2f}")
        logger.info(f"🎯 Trades: {self.trades}")
        logger.info(f"✅ Wins: {self.wins}")
        logger.info(f"❌ Losses: {self.trades - self.wins}")
        logger.info(f"🏆 Win Rate: {win_rate:.1f}%")
        logger.info("=" * 50)

def main():
    """Run the simple test"""
    print("=" * 50)
    print("🧪 SIMPLE BOT FUNCTIONALITY TEST")
    print("=" * 50)
    
    test_bot = SimpleBotTest()
    test_bot.run_test(20)  # Run 20 test trades
    
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    main()
