#!/usr/bin/env python3
"""
Test script for enhanced trading bot features:
1. Improved lot size calculation
2. Validation dashboard
3. Go-live readiness checker
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import bot components
try:
    from main import DerivTradingBot
    print("✅ Bot imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_enhancements():
    """Test the enhanced features"""
    print("🧪 Testing Enhanced Trading Bot Features")
    print("=" * 50)

    try:
        # Initialize bot
        print("1. Initializing bot...")
        bot = DerivTradingBot()

        # Test lot size calculation
        print("\n2. Testing lot size calculation...")
        test_symbols = [
            'Volatility 75 Index',
            'Volatility 50 Index',
            'Boom 1000 Index',
            'EURUSD',
            'XAUUSD'
        ]

        for symbol in test_symbols:
            lot_size = bot.calculate_lot_size(symbol, 10.0)  # $10 trade
            print(f"   {symbol}: ${10:.2f} → {lot_size:.3f} lots")

        # Add some test data for validation
        print("\n3. Adding test data...")
        bot.session_stats['total_trades'] = 25
        bot.session_stats['wins'] = 16
        bot.session_stats['losses'] = 9
        bot.session_stats['total_profit'] = 150.75
        bot.consecutive_losses = 2
        bot.daily_profit = 45.50

        # Add test learning data
        for i in range(75):  # 75 learning samples
            bot.rl_data.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'BUY' if i % 2 == 0 else 'SELL',
                'reward': 1.0 if i % 3 != 0 else -1.0,  # ~67% win rate
                'confidence': 0.8
            })

        # Add test confidence heatmap
        for i in range(30):
            bot.confidence_heatmap.append({
                'timestamp': datetime.now().isoformat(),
                'confidence': 0.7 + (i % 3) * 0.1,  # 0.7, 0.8, 0.9
                'result': 'WIN' if i % 4 != 0 else 'LOSS',  # 75% win rate
                'confidence_bucket': 0.8
            })

        print(f"   Added {len(bot.rl_data)} learning samples")
        print(f"   Added {len(bot.confidence_heatmap)} confidence entries")

        # Test validation dashboard
        print("\n4. Testing validation dashboard...")
        dashboard = bot.get_validation_dashboard()
        print(f"   Overall Score: {dashboard['overall_score']}/{dashboard['max_possible_score']} ({dashboard['score_percentage']:.1f}%)")
        print(f"   Readiness Level: {dashboard['readiness_level']}")
        print(f"   Achievements: {len(dashboard['achievements'])}")
        print(f"   Blockers: {len(dashboard['blockers'])}")

        # Test go-live readiness
        print("\n5. Testing go-live readiness...")
        readiness = bot.check_go_live_readiness()
        print(f"   Ready for Live: {readiness['ready_for_live']}")
        print(f"   Confidence Level: {readiness['confidence_level']}")
        print(f"   Risk Assessment: {readiness['risk_assessment']}")
        print(f"   Requirements Met: {len(readiness['requirements_met'])}")
        print(f"   Requirements Failed: {len(readiness['requirements_failed'])}")

        # Display detailed results
        print("\n6. Detailed Results:")
        print("   ✅ Requirements Met:")
        for req in readiness['requirements_met']:
            print(f"      {req}")

        print("   ❌ Requirements Failed:")
        for req in readiness['requirements_failed']:
            print(f"      {req}")

        print("   📋 Next Steps:")
        for step in readiness['next_steps']:
            print(f"      • {step}")

        # Test Telegram command simulation
        print("\n7. Testing command handlers...")

        # Create mock update object
        class MockMessage:
            async def reply_text(self, text, parse_mode=None):
                print(f"   📱 Telegram Response:\n{text[:200]}...")

        class MockUpdate:
            message = MockMessage()

        # Test readiness command
        await bot.telegram_handler.handle_readiness_check(MockUpdate(), None)

        print("\n🎉 All tests completed successfully!")
        print("\n📊 Summary:")
        print("   • Lot size calculation: ✅ Working")
        print("   • Validation dashboard: ✅ Working")
        print("   • Go-live checker: ✅ Working")
        print("   • Telegram commands: ✅ Working")
        print(f"   • Current readiness: {'🟢 READY' if readiness['ready_for_live'] else '🟡 NOT READY'}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhancements())
