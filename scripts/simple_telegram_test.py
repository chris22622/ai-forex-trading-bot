#!/usr/bin/env python3
"""
Simple Telegram Test - No HTML formatting
"""

import os
import sys

import requests

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def test_simple_telegram():
    """Test simple Telegram message without HTML"""

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram credentials not configured")
        return False

    print("🚀 Testing simple Telegram message...")
    print(f"📱 Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...")
    print(f"📱 Chat ID: {TELEGRAM_CHAT_ID}")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Simple message without any formatting
    message = """🚀 Enhanced Telegram Notifications Test

✅ Connection successful!
💰 Balance: $100.50
📊 Active Trades: 0
🕒 Time: 10:30:00

🎯 Enhanced notifications are now enabled!
Beautiful trade alerts coming your way!"""

    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'disable_web_page_preview': True
    }

    try:
        response = requests.post(url, data=data, timeout=10)

        if response.status_code == 200:
            print("✅ Simple test message sent successfully!")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_telegram()

    if success:
        print("\n🎯 Telegram is working!")
        print("📱 Your bot can now send beautiful trade notifications!")
    else:
        print("\n⚠️ Telegram test failed")
        print("💡 Check your bot token and chat ID configuration")
