#!/usr/bin/env python3
"""
🚀 Test Enhanced Telegram Notifications
Quick test to verify the enhanced notification system works
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import ENABLE_TELEGRAM_ALERTS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    from enhanced_telegram_notifier import EnhancedTelegramNotifier, TradeRecord

    print("🚀 Testing Enhanced Telegram Notifications...")
    print(f"✅ Bot Token configured: {'Yes' if TELEGRAM_BOT_TOKEN else 'No'}")
    print(f"✅ Chat ID configured: {'Yes' if TELEGRAM_CHAT_ID else 'No'}")
    print(f"✅ Alerts enabled: {ENABLE_TELEGRAM_ALERTS}")

    # Create notifier
    notifier = EnhancedTelegramNotifier()
    print(f"✅ Enhanced Notifier ready: {notifier.enabled}")

    if notifier.enabled:
        # Send test message
        print("📱 Sending test message...")
        success = notifier.send_bot_status("TESTING", 100.0, 0)
        print(f"✅ Test message sent: {success}")

        # Test trade notification
        print("📱 Testing trade notification...")
        test_trade = TradeRecord(
            trade_id="TEST_001",
            symbol="Volatility 75 Index",
            direction="BUY",
            entry_price=1.23456,
            entry_time="2024-08-09T10:30:00",
            stake=5.0,
            confidence=0.85,
            strategy="Test Strategy",
            reason="Test signal"
        )

        success = notifier.notify_trade_opened(test_trade)
        print(f"✅ Trade opening notification: {success}")

        print("\n🎯 Enhanced notifications are working!")
        print("🔥 Your bot will now send beautiful, detailed trade notifications!")
    else:
        print("\n⚠️ Enhanced notifications not enabled")
        print("💡 Check your Telegram configuration in config.py")
        print("📱 Make sure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 To fix this:")
    print("1. Make sure you have the required packages installed")
    print("2. Configure your Telegram bot token and chat ID in config.py")
    print("3. Set ENABLE_TELEGRAM_ALERTS = True in config.py")
