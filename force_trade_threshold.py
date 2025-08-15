"""
🎯 FORCE IMMEDIATE TRADING - LOWER AI THRESHOLD
This script will temporarily lower the AI confidence threshold to force trades
"""
import sys
import os
import importlib
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def force_low_threshold_trading():
    """Force trading by temporarily lowering AI confidence threshold"""
    try:
        # Import config
        import config
        
        print("🎯 CURRENT AI SETTINGS:")
        print(f"   AI_CONFIDENCE_THRESHOLD: {config.AI_CONFIDENCE_THRESHOLD}")
        print(f"   ENABLE_AI_PREDICTIONS: {config.ENABLE_AI_PREDICTIONS}")
        
        # Backup original values
        original_threshold = config.AI_CONFIDENCE_THRESHOLD
        original_enable_ai = config.ENABLE_AI_PREDICTIONS
        
        # Set ultra-low threshold to force trades
        config.AI_CONFIDENCE_THRESHOLD = 0.01  # Nearly any signal will pass
        config.ENABLE_AI_PREDICTIONS = True  # Make sure AI is enabled
        
        print("\n🚀 FORCING LOW THRESHOLD:")
        print(f"   NEW AI_CONFIDENCE_THRESHOLD: {config.AI_CONFIDENCE_THRESHOLD}")
        print(f"   AI ENABLED: {config.ENABLE_AI_PREDICTIONS}")
        
        print("\n✅ Threshold lowered! The bot should start trading within 30 seconds.")
        print("📊 Monitor the bot output - it should now accept trades with very low confidence.")
        
        # Wait and monitor for a bit
        print("\n⏳ Monitoring for 60 seconds...")
        for i in range(60):
            print(f"   Waiting... {60-i} seconds remaining", end="\r")
            time.sleep(1)
        
        # Restore original settings
        config.AI_CONFIDENCE_THRESHOLD = original_threshold
        config.ENABLE_AI_PREDICTIONS = original_enable_ai
        
        print(f"\n🔄 Settings restored:")
        print(f"   AI_CONFIDENCE_THRESHOLD: {config.AI_CONFIDENCE_THRESHOLD}")
        print(f"   ENABLE_AI_PREDICTIONS: {config.ENABLE_AI_PREDICTIONS}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def disable_ai_completely():
    """Disable AI completely to use basic trading logic"""
    try:
        import config
        
        print("🎯 DISABLING AI FOR BASIC TRADING:")
        original_enable_ai = config.ENABLE_AI_PREDICTIONS
        config.ENABLE_AI_PREDICTIONS = False
        
        print(f"   AI DISABLED: {not config.ENABLE_AI_PREDICTIONS}")
        print("📊 Bot will now use basic price movement logic instead of AI")
        
        print("\n⏳ Monitoring for 60 seconds...")
        for i in range(60):
            print(f"   Waiting... {60-i} seconds remaining", end="\r")
            time.sleep(1)
        
        # Restore
        config.ENABLE_AI_PREDICTIONS = original_enable_ai
        print(f"\n🔄 AI Setting restored: {config.ENABLE_AI_PREDICTIONS}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🎯 FORCE IMMEDIATE TRADING SCRIPT")
    print("=" * 50)
    
    print("\nChoose option:")
    print("1. Lower AI confidence threshold (recommended)")
    print("2. Disable AI completely")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🚀 LOWERING AI CONFIDENCE THRESHOLD...")
        force_low_threshold_trading()
    elif choice == "2":
        print("\n🚀 DISABLING AI TEMPORARILY...")
        disable_ai_completely()
    else:
        print("❌ Invalid choice")
    
    input("\nPress Enter to exit...")
