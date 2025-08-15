#!/usr/bin/env python3
"""
🎯 ACTIVATE UNLIMITED DEMO TRAINING MODE
Forces the bot to trade continuously for AI learning
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import force_enable_demo_training_mode, is_demo_training_mode, global_bot_instance
    
    print("🎯 DEMO TRAINING ACTIVATOR")
    print("=" * 50)
    
    # Check current status
    print("\n📊 Current Status:")
    if global_bot_instance:
        bot = global_bot_instance
        print(f"  Bot Running: ✅ YES")
        print(f"  Account: {getattr(bot.mt5_interface, 'account', 'Unknown')}")
        print(f"  Server: {getattr(bot.mt5_interface, 'server', 'Unknown')}")
        print(f"  Balance: ${bot.current_balance:.2f}")
        print(f"  Demo Mode: {'✅ ACTIVE' if getattr(bot, 'demo_mode', False) else '❌ INACTIVE'}")
        print(f"  Loss Protection: {'❌ DISABLED' if getattr(bot, 'loss_protection_triggered', False) else '✅ ENABLED'}")
    else:
        print("  Bot Running: ❌ NO")
    
    # Force enable demo training
    print("\n🎯 Activating Unlimited Demo Training...")
    result = force_enable_demo_training_mode()
    
    if result:
        print("✅ DEMO TRAINING MODE ACTIVATED!")
        print("🚀 Bot will trade continuously for AI learning!")
        print("🧠 All trades will make the AI smarter!")
        print("💰 No limits on demo account losses!")
        print("⚡ Emergency trading enabled for maximum learning!")
        
        # Verify activation
        is_demo = is_demo_training_mode()
        print(f"\n🎯 Demo Training Status: {'✅ ACTIVE' if is_demo else '❌ FAILED'}")
        
    else:
        print("❌ Failed to activate demo training mode")
        print("   Make sure the bot is running first")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure the bot is running (python main.py)")
    print("2. Check that you're in the correct directory")
    print("3. Verify MT5 connection is active")

print("\n" + "=" * 50)
print("🎯 Demo Training Activation Complete!")
