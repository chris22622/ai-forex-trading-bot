#!/usr/bin/env python3
"""
🤖 Telegram Bot Setup Helper
Get your chat ID and test the bot connection
"""

import requests
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TELEGRAM_BOT_TOKEN

def get_bot_info():
    """Get bot information"""
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not configured")
        return None
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                bot_info = data.get('result', {})
                print(f"✅ Bot Found: @{bot_info.get('username', 'unknown')}")
                print(f"📝 Bot Name: {bot_info.get('first_name', 'unknown')}")
                print(f"🆔 Bot ID: {bot_info.get('id', 'unknown')}")
                return bot_info
            else:
                print(f"❌ Bot Error: {data}")
                return None
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def get_chat_updates():
    """Get recent chat updates to find chat ID"""
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not configured")
        return []
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                updates = data.get('result', [])
                print(f"📨 Found {len(updates)} recent messages")
                
                chats = set()
                for update in updates:
                    message = update.get('message', {})
                    chat = message.get('chat', {})
                    if chat:
                        chat_id = chat.get('id')
                        chat_type = chat.get('type', 'unknown')
                        username = chat.get('username', 'No username')
                        first_name = chat.get('first_name', 'No name')
                        
                        chats.add((chat_id, chat_type, username, first_name))
                
                if chats:
                    print("\n💬 Available Chats:")
                    for chat_id, chat_type, username, first_name in chats:
                        print(f"   🆔 Chat ID: {chat_id}")
                        print(f"   📝 Type: {chat_type}")
                        print(f"   👤 User: {first_name} (@{username})")
                        print(f"   ---")
                else:
                    print("⚠️ No chats found")
                    print("💡 Send a message to your bot first!")
                
                return list(chats)
            else:
                print(f"❌ API Error: {data}")
                return []
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

def test_chat_id(chat_id):
    """Test sending a message to a specific chat ID"""
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not configured")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    message = f"""🎯 Chat ID Test Successful!

✅ Your chat ID {chat_id} is working!
🤖 Bot is ready to send notifications
🚀 Enhanced trading alerts enabled!

You can now use this chat ID in your config.py:
TELEGRAM_CHAT_ID = "{chat_id}"

Happy trading! 📈💰"""
    
    data = {
        'chat_id': chat_id,
        'text': message
    }
    
    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            print(f"✅ Test message sent to chat {chat_id}")
            return True
        else:
            print(f"❌ Failed to send to chat {chat_id}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("🤖 Telegram Bot Setup Helper\n")
    
    # Step 1: Check bot info
    print("📋 Step 1: Checking bot information...")
    bot_info = get_bot_info()
    
    if not bot_info:
        print("\n❌ Bot setup failed!")
        print("💡 Make sure your TELEGRAM_BOT_TOKEN is correct in config.py")
        return
    
    bot_username = bot_info.get('username', 'unknown')
    
    print(f"\n📱 Step 2: Start a chat with your bot")
    print(f"   1. Open Telegram")
    print(f"   2. Search for @{bot_username}")
    print(f"   3. Send /start or any message to the bot")
    print(f"   4. Run this script again")
    
    # Step 3: Get chat updates
    print(f"\n📨 Step 3: Looking for recent messages...")
    chats = get_chat_updates()
    
    if chats:
        print(f"\n🎯 Step 4: Testing chat connections...")
        for chat_id, chat_type, username, first_name in chats:
            if chat_type == 'private':  # Only test private chats
                print(f"\n🧪 Testing chat {chat_id} ({first_name})...")
                if test_chat_id(chat_id):
                    print(f"✅ Success! Use this in your config.py:")
                    print(f"TELEGRAM_CHAT_ID = \"{chat_id}\"")
                    break
    else:
        print(f"\n⚠️ No messages found!")
        print(f"💡 Please send a message to @{bot_username} first")

if __name__ == "__main__":
    main()
