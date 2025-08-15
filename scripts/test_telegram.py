#!/usr/bin/env python3
"""
Quick Telegram test to debug message parsing issues
"""

import asyncio

from telegram_bot import TelegramNotifier


async def test_telegram():
    """Test Telegram notifications"""
    print("🧪 Testing Telegram notifications...")

    # Create notifier
    notifier = TelegramNotifier()

    if not notifier.enabled:
        print("❌ Telegram not enabled")
        return

    print("✅ Telegram notifier created")

    # Test simple message
    print("📱 Sending simple test message...")
    result = await notifier.send_message("🧪 Simple test message")
    print(f"Result: {result}")

    # Test startup message
    print("📱 Sending startup message...")
    result = await notifier.send_startup_message()
    print(f"Result: {result}")

    # Test status message
    print("📱 Sending status message...")
    status_data = {
        'status': 'CONNECTED_MT5',
        'uptime': 0,
        'balance': 10000.0,
        'trades_today': 0,
        'performance': 0.0
    }
    result = await notifier.send_bot_status(status_data)
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_telegram())
