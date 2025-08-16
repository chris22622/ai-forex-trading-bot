#!/usr/bin/env python3
"""
Demo Script for Deriv Trading Bot
Shows the bot capabilities without connecting to real API
"""

import asyncio
import random
from typing import Any, Dict, List, Optional

from telegram_bot import NotificationManager
from utils import format_currency, performance_tracker, trade_logger

from ai_model import TradingAI

# Import our modules
from indicators import TechnicalIndicators


class BotDemo:
    """Demo version of the trading bot"""

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.ai_model = TradingAI()
        self.notifier = NotificationManager()
        self.balance = 100.0
        self.trades_executed = 0

    async def simulate_market_data(self) -> List[float]:
        """Simulate realistic market price movements"""
        base_price = 1000.0
        prices: List[float] = []

        print("📊 Generating simulated market data...")

        for i in range(100):
            # Add some trend and noise
            trend = 0.1 * (random.random() - 0.5)
            noise = 2.0 * (random.random() - 0.5)

            price_change = trend + noise
            base_price += price_change

            # Keep price in reasonable range
            base_price = max(990, min(1010, base_price))
            prices.append(base_price)

            # Update indicators
            self.indicators.update_price_data(base_price)

            # Small delay to simulate real-time
            await asyncio.sleep(0.01)

        return prices

    async def demonstrate_analysis(self) -> Dict[str, Any]:
        """Show technical analysis capabilities"""
        print("\n🔧 Technical Analysis Demo")
        print("-" * 40)

        # Get indicators summary
        summary: Dict[str, Any] = self.indicators.get_indicator_summary()

        if summary['price']:
            print(f"💰 Current Price: {summary['price']:.2f}")

            if summary['rsi']:
                print(f"📈 RSI: {summary['rsi']:.1f}")

            if summary['ema_fast']:
                print(f"📊 EMA Fast: {summary['ema_fast']:.2f}")

            if summary['ema_slow']:
                print(f"📊 EMA Slow: {summary['ema_slow']:.2f}")

            # Show signals
            signals: Dict[str, Any] = summary['signals']
            print(f"🎯 Overall Signal: {signals['overall_signal']}")
            print("📊 Signal Breakdown:")
            for signal_type, signal in signals.items():
                if signal_type != 'overall_signal':
                    print(f"   • {signal_type}: {signal}")

        return summary

    async def demonstrate_ai_prediction(self, indicator_data: Dict[str, Any]) -> Dict[str, Any]:
        """Show AI prediction capabilities"""
        print("\n🧠 AI Prediction Demo")
        print("-" * 40)

        # Get AI prediction
        prediction: Dict[str, Any] = self.ai_model.predict_trade(indicator_data)

        print(f"🎯 Prediction: {prediction['prediction']}")
        print(f"📊 Confidence: {prediction['confidence']:.1%}")
        print(f"🧠 Method: {prediction['method']}")
        print(f"📝 Reason: {prediction['reason']}")
        print(f"🔧 Features Used: {len(prediction['features_used'])}")

        return prediction

    async def simulate_trade(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulate executing a trade"""
        if prediction['prediction'] in ['BUY', 'SELL']:
            self.trades_executed += 1
            trade_amount = 1.0

            print(f"\n💼 Executing Trade #{self.trades_executed}")
            print("-" * 40)
            print(f"🎯 Action: {prediction['prediction']}")
            print(f"💰 Amount: ${trade_amount}")
            print(f"📊 Confidence: {prediction['confidence']:.1%}")

            # Simulate trade outcome (85% success rate for demo)
            is_win = random.random() < 0.85
            profit_loss = trade_amount * 0.85 if is_win else -trade_amount

            self.balance += profit_loss

            result = "WIN" if is_win else "LOSS"
            print(f"📊 Result: {result}")
            print(f"💰 P&L: {format_currency(profit_loss)}")
            print(f"💵 New Balance: ${self.balance:.2f}")

            # Log the trade
            trade_data: Dict[str, Any] = {
                'trade_id': f'demo_{self.trades_executed}',
                'action': prediction['prediction'],
                'symbol': 'R_75',
                'amount': trade_amount,
                'price': 1000.0,
                'confidence': prediction['confidence'],
                'reason': prediction['reason'],
                'result': result,
                'profit_loss': profit_loss,
                'balance': self.balance
            }

            trade_logger.log_trade(trade_data)
            performance_tracker.update_trade_result(result, profit_loss, self.balance)

            # Send demo notification
            if self.notifier.telegram.enabled:
                await self.notifier.notify_trade({
                    "action": prediction['prediction'],
                    "symbol": "R_75 (DEMO)",
                    "amount": trade_amount,
                    "price": 1000.0,
                    "confidence": prediction['confidence'],
                    "reason": f"DEMO: {prediction['reason']}"
                })

            return trade_data

        return None

    async def show_performance_summary(self):
        """Show performance metrics"""
        print("\n📈 Performance Summary")
        print("-" * 40)

        metrics = performance_tracker.get_metrics()

        print(f"📊 Total Trades: {metrics['total_trades']}")
        print(f"✅ Wins: {metrics['wins']}")
        print(f"❌ Losses: {metrics['losses']}")
        print(f"📊 Win Rate: {metrics['win_rate']:.1%}")
        print(f"💰 Total Profit: {format_currency(metrics['total_profit'])}")
        print(f"💵 Current Balance: ${self.balance:.2f}")
        print(f"📈 ROI: {(self.balance - 100) / 100 * 100:+.1f}%")

    async def run_demo(self):
        """Run the complete demo"""
        print("🚀 DERIV TRADING BOT - DEMO MODE")
        print("=" * 50)
        print("This demo shows the bot's capabilities without")
        print("connecting to the real Deriv API.")
        print("=" * 50)

        # Generate market data
        prices = await self.simulate_market_data()
        print(f"✅ Generated {len(prices)} price points")

        # Wait for enough data for analysis
        if len(prices) >= 50:
            # Run analysis cycles
            for cycle in range(5):
                print(f"\n🔄 Analysis Cycle {cycle + 1}/5")
                print("=" * 30)

                # Show analysis
                indicator_data = await self.demonstrate_analysis()

                # Get AI prediction
                prediction = await self.demonstrate_ai_prediction(indicator_data)

                # Simulate trade if signal is strong
                if prediction['confidence'] >= 0.6:
                    trade_result = await self.simulate_trade(prediction)
                    if trade_result:
                        print("✅ Trade executed!")
                    else:
                        print("⏸️ No trade signal")
                else:
                    print("⏸️ Confidence too low, no trade")

                # Wait before next cycle
                await asyncio.sleep(1)

            # Show final summary
            await self.show_performance_summary()

        print("\n🎯 Demo completed!")
        print("To run the real bot: python main.py")

async def main():
    """Main demo function"""
    demo = BotDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
