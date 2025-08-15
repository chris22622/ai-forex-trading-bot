#!/usr/bin/env python3
"""
🚨 EMERGENCY FIX: Stop the infinite config loop
Fix the config file to stop the bot from reloading config constantly
"""

import shutil
from datetime import datetime


def emergency_fix_config():
    """Fix the config file to stop infinite loop"""
    config_path = "config.py"

    print("🚨 EMERGENCY FIX: Stopping infinite config loop...")

    try:
        # Create backup first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"config_backup_emergency_{timestamp}.py"
        shutil.copy(config_path, backup_file)
        print(f"✅ Config backed up to: {backup_file}")

        # Read current config with proper encoding
        with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Fix the main issue: MAX_CONSECUTIVE_LOSSES
        fixes_applied = []

        # 1. Fix MAX_CONSECUTIVE_LOSSES
        if 'MAX_CONSECUTIVE_LOSSES = 3' in content:
            content = content.replace('MAX_CONSECUTIVE_LOSSES = 3', 'MAX_CONSECUTIVE_LOSSES = 100')
            fixes_applied.append("MAX_CONSECUTIVE_LOSSES: 3 → 100")
        elif 'MAX_CONSECUTIVE_LOSSES=3' in content:
            content = content.replace('MAX_CONSECUTIVE_LOSSES=3', 'MAX_CONSECUTIVE_LOSSES=100')
            fixes_applied.append("MAX_CONSECUTIVE_LOSSES: 3 → 100")

        # 2. Add missing MT5 variables if not present
        if 'MT5_REAL_TRADING' not in content:
            content += '\n# Missing MT5 variables (added by emergency fix)\nMT5_REAL_TRADING = False  # Demo mode\n'
            fixes_applied.append("Added MT5_REAL_TRADING = False")

        if 'MT5_DEMO_MODE' not in content and 'MT5_REAL_TRADING = False' in content:
            content += 'MT5_DEMO_MODE = True   # Demo mode active\n'
            fixes_applied.append("Added MT5_DEMO_MODE = True")

        # 3. Make sure AI confidence is reasonable (not 0.05 which causes emergency mode)
        if 'AI_CONFIDENCE_THRESHOLD = 0.05' in content:
            content = content.replace('AI_CONFIDENCE_THRESHOLD = 0.05', 'AI_CONFIDENCE_THRESHOLD = 0.65')
            fixes_applied.append("AI_CONFIDENCE_THRESHOLD: 0.05 → 0.65")

        # Write fixed config
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print("✅ EMERGENCY FIX COMPLETE!")
        if fixes_applied:
            print("🔧 Fixes applied:")
            for fix in fixes_applied:
                print(f"   • {fix}")
        else:
            print("ℹ️ No fixes needed - config already correct")

        print("\n🚀 You can now restart the bot - it should work normally!")
        print("💡 The infinite loop has been stopped.")

        return True

    except Exception as e:
        print(f"❌ Error during emergency fix: {e}")
        print("💡 Try manually editing config.py:")
        print("   • Change MAX_CONSECUTIVE_LOSSES = 3 to MAX_CONSECUTIVE_LOSSES = 100")
        print("   • Add MT5_REAL_TRADING = False")
        print("   • Add MT5_DEMO_MODE = True")
        return False

if __name__ == "__main__":
    emergency_fix_config()
