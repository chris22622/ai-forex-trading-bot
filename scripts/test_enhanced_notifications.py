#!/usr/bin/env python3
"""
🧪 Test Enhanced Telegram Notifications
Quick test to verify all components work correctly
"""

import asyncio
from datetime import datetime

from enhanced_telegram_notifier import EnhancedTelegramNotifier, TradeRecord

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


async def test_enhanced_notifications():
    """Test the enhanced notification system"""
    print("🧪 Testing Enhanced Telegram Notifications...")
    print(f"📱 Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...")
    print(f"💬 Chat ID: {TELEGRAM_CHAT_ID}")

    try:
        # Initialize the enhanced notifier
        notifier = EnhancedTelegramNotifier()
        print("✅ Enhanced notifier initialized")

        # Test bot status notification
        print("\n📊 Testing bot status notification...")
        result = notifier.send_bot_status(
            status="TESTING",
            balance=54.34,
            active_trades=1
        )

        if result:
            print("✅ Bot status notification sent successfully!")
        else:
            print("❌ Bot status notification failed")

        # Test trade notification
        print("\n📈 Testing trade opened notification...")
        trade_record = TradeRecord(
            trade_id="TEST_40597130243",
            symbol="Volatility 75 Index",
            direction="BUY",
            entry_price=6194.86,
            entry_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            stake=0.5,
            confidence=0.75,
            strategy="AI_ENHANCED"
        )

        result = notifier.notify_trade_opened(trade_record)

        if result:
            print("✅ Trade opened notification sent successfully!")
        else:
            print("❌ Trade opened notification failed")

        print("\n🎉 All tests completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_notifications())
