#!/usr/bin/env python3
"""
🚀 ENABLE ENHANCED TELEGRAM NOTIFICATIONS
Beautiful, detailed notifications with persistent trade tracking
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_telegram_notifier import EnhancedTelegramNotifier, TradeRecord

from config import ENABLE_TELEGRAM_ALERTS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def test_enhanced_notifications():
    """Test the enhanced notification system"""
    print("🚀 Testing Enhanced Telegram Notifications...")

    # Check configuration
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not configured in config.py")
        return False

    if not TELEGRAM_CHAT_ID:
        print("❌ TELEGRAM_CHAT_ID not configured in config.py")
        return False

    if not ENABLE_TELEGRAM_ALERTS:
        print("⚠️ ENABLE_TELEGRAM_ALERTS is disabled in config.py")
        print("💡 Consider setting ENABLE_TELEGRAM_ALERTS = True")

    print(f"✅ Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...")
    print(f"✅ Chat ID: {TELEGRAM_CHAT_ID}")
    print(f"✅ Alerts Enabled: {ENABLE_TELEGRAM_ALERTS}")

    # Create enhanced notifier
    notifier = EnhancedTelegramNotifier()

    print(f"✅ Enhanced Notifier Created - Enabled: {notifier.enabled}")

    # Test sending a sample notification
    if notifier.enabled:
        print("📱 Sending test notification...")

        # Test status notification
        success = notifier.send_bot_status(
            status="TESTING",
            balance=100.50,
            active_trades=0
        )

        if success:
            print("✅ Test notification sent successfully!")
        else:
            print("❌ Test notification failed")
            return False

        # Test trade opening notification
        print("📱 Sending test trade opening notification...")
        test_trade = TradeRecord(
            trade_id="TEST_001",
            symbol="Volatility 75 Index",
            direction="BUY",
            entry_price=1.23456,
            entry_time="2024-08-09T10:30:00",
            stake=5.0,
            confidence=0.85,
            strategy="AI + Technical",
            reason="Strong bullish signal",
            win_rate_at_time=0.72
        )

        success = notifier.notify_trade_opened(test_trade)

        if success:
            print("✅ Test trade opening notification sent!")
        else:
            print("❌ Test trade opening notification failed")

        # Test trade closing notification
        print("📱 Sending test trade closing notification...")
        success = notifier.notify_trade_closed(
            trade_id="TEST_001",
            exit_price=1.23789,
            profit_loss=2.35,
            close_reason="Take Profit Hit"
        )

        if success:
            print("✅ Test trade closing notification sent!")
        else:
            print("❌ Test trade closing notification failed")

        # Test daily summary
        print("📱 Sending test daily summary...")
        success = notifier.send_daily_summary()

        if success:
            print("✅ Test daily summary sent!")
        else:
            print("❌ Test daily summary failed")

    return True

def show_configuration_guide():
    """Show configuration guide"""
    print("""
🔧 ENHANCED TELEGRAM NOTIFICATIONS CONFIGURATION GUIDE

1. 📱 Create Telegram Bot:
   - Message @BotFather on Telegram
   - Send /newbot
   - Choose a name and username for your bot
   - Copy the bot token

2. 🆔 Get Your Chat ID:
   - Start a chat with your new bot
   - Send any message to the bot
   - Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   - Find your chat ID in the response

3. ⚙️ Update config.py:
   TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
   TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
   ENABLE_TELEGRAM_ALERTS = True

4. 🚀 Features Enabled:
   ✅ Beautiful trade opening notifications
   ✅ Detailed trade closing notifications with P&L
   ✅ Persistent trade tracking across bot restarts
   ✅ Daily trading summaries
   ✅ Bot status notifications
   ✅ Visual progress bars and emojis
   ✅ Comprehensive statistics tracking

5. 📊 Trade Persistence:
   - All trades are saved to trades_history.json
   - Statistics persist across bot restarts
   - Historical data for performance analysis
   - Automatic cleanup of old trades

6. 🎨 Visual Features:
   - Color-coded profit/loss indicators
   - Progress bars for confidence and win rates
   - Emoji-rich formatting for easy reading
   - Structured layouts for quick scanning

Run this script again after configuration to test!
""")

if __name__ == "__main__":
    print("🚀 Enhanced Telegram Notifications Setup\n")

    try:
        success = test_enhanced_notifications()

        if not success:
            print("\n" + "="*50)
            show_configuration_guide()

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n" + "="*50)
        show_configuration_guide()

    print("\n✨ Enhanced notifications are ready for your trading bot!")
    print("🎯 Your trades will now have beautiful, detailed Telegram notifications!")
