#!/usr/bin/env python3
"""
🔧 Fix Telegram Chat Issue
This script helps resolve the "Chat not found" error by guiding you through the setup.
"""

import requests
import json
import os
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def check_bot_info():
    """Check if the bot token is valid"""
    print("🔍 Checking bot information...")
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            bot_info = response.json()
            if bot_info.get('ok'):
                bot = bot_info['result']
                print(f"✅ Bot found: @{bot.get('username', 'Unknown')}")
                print(f"📝 Bot name: {bot.get('first_name', 'Unknown')}")
                print(f"🆔 Bot ID: {bot.get('id', 'Unknown')}")
                return True
            else:
                print(f"❌ Bot API error: {bot_info.get('description', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return False

def get_chat_updates():
    """Get recent messages to find the chat ID"""
    print("\n🔍 Checking for recent messages...")
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            updates = response.json()
            if updates.get('ok') and updates.get('result'):
                recent_chats = set()
                for update in updates['result']:
                    if 'message' in update:
                        chat = update['message']['chat']
                        chat_id = chat['id']
                        chat_type = chat['type']
                        
                        if chat_type == 'private':
                            first_name = chat.get('first_name', 'Unknown')
                            username = chat.get('username', 'No username')
                            recent_chats.add((chat_id, first_name, username))
                
                if recent_chats:
                    print(f"✅ Found {len(recent_chats)} recent chat(s):")
                    for chat_id, name, username in recent_chats:
                        print(f"   🆔 Chat ID: {chat_id}")
                        print(f"   👤 Name: {name}")
                        print(f"   📱 Username: @{username}")
                        print()
                    return list(recent_chats)
                else:
                    print("⚠️ No recent messages found.")
                    return []
            else:
                print(f"❌ API error: {updates.get('description', 'Unknown error')}")
                return []
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return []
    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return []

def test_send_message(chat_id):
    """Test sending a message to the chat"""
    print(f"\n🧪 Testing message to chat ID: {chat_id}")
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        'chat_id': chat_id,
        'text': '🤖 Hello! Your trading bot is now connected!\n\n✅ Chat ID configured successfully\n📱 You will now receive trading notifications'
    }
    
    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                print("✅ Test message sent successfully!")
                return True
            else:
                print(f"❌ Message failed: {result.get('description', 'Unknown error')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return False

def main():
    print("🔧 TELEGRAM CHAT FIX TOOL")
    print("=" * 50)
    
    # Step 1: Check bot
    if not check_bot_info():
        print("\n❌ Bot token is invalid. Please check your TELEGRAM_BOT_TOKEN in config.py")
        return
    
    # Step 2: Get current chat ID from config
    print(f"\n📋 Current chat ID in config: {TELEGRAM_CHAT_ID}")
    
    # Step 3: Test current chat ID
    if test_send_message(TELEGRAM_CHAT_ID):
        print("\n🎉 SUCCESS! Your Telegram setup is working correctly!")
        return
    
    # Step 4: If failed, look for recent chats
    print("\n⚠️ Current chat ID failed. Searching for recent messages...")
    recent_chats = get_chat_updates()
    
    if recent_chats:
        print("\n💡 SOLUTION FOUND!")
        print("Recent chats detected. Try updating your config.py with one of these chat IDs:")
        for chat_id, name, username in recent_chats:
            print(f"   TELEGRAM_CHAT_ID = '{chat_id}'  # {name} (@{username})")
    else:
        print("\n❌ NO RECENT MESSAGES FOUND")
        print("\n📋 MANUAL SETUP REQUIRED:")
        print("1. Open Telegram app")
        print("2. Search for your bot (check the bot username above)")
        print("3. Send ANY message to the bot (like 'Hello' or '/start')")
        print("4. Run this script again to get your chat ID")
        print("5. Update TELEGRAM_CHAT_ID in config.py with the new chat ID")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
