#!/usr/bin/env python3
"""
Get Telegram Chat ID Helper
"""

import asyncio

from telegram import Bot

from config import TELEGRAM_BOT_TOKEN


async def get_chat_id():
    """Get your Telegram chat ID"""
    print("🔍 Getting Telegram chat updates...")
    print("📱 Please send ANY message to your bot first, then run this script")

    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)

        # Get recent updates
        updates = await bot.get_updates()

        if not updates:
            print("❌ No messages found!")
            print("📱 Please:")
            print("   1. Open Telegram")
            print("   2. Search for your bot")
            print("   3. Send any message (like 'hello')")
            print("   4. Run this script again")
            return

        print(f"✅ Found {len(updates)} message(s):")

        for update in updates[-5:]:  # Show last 5 messages
            if update.message:
                chat_id = update.message.chat.id
                username = update.message.from_user.username or "No username"
                first_name = update.message.from_user.first_name or "No name"
                message_text = update.message.text or "No text"

                print("")
                print(f"📱 Chat ID: {chat_id}")
                print(f"👤 Username: @{username}")
                print(f"🏷️ Name: {first_name}")
                print(f"💬 Message: {message_text}")
                print(f"⏰ Time: {update.message.date}")

        print(f"\n🎯 Use the Chat ID in your config.py: TELEGRAM_CHAT_ID = \"{chat_id}\"")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(get_chat_id())
