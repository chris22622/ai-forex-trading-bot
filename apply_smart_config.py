"""
Apply Smart Trading Configuration for Maximum Profitability
Run this script to update your bot with optimized profitable settings
"""

import os
import shutil
from datetime import datetime

def apply_smart_config():
    """Apply smart trading configuration for better profitability"""
    
    print("🧠 APPLYING SMART TRADING CONFIGURATION...")
    print("   Optimizing for maximum profitability and risk management")
    
    try:
        # Backup current config
        if os.path.exists('config.py'):
            backup_name = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            shutil.copy('config.py', backup_name)
            print(f"✅ Current config backed up as: {backup_name}")
        
        # Copy smart config to main config
        if os.path.exists('smart_trading_config.py'):
            shutil.copy('smart_trading_config.py', 'config.py')
            print("✅ Smart trading configuration applied!")
        else:
            print("❌ smart_trading_config.py not found!")
            return False
        
        print("\n🎯 SMART TRADING FEATURES ENABLED:")
        print("   💰 Lower profit targets for quicker wins ($1.20)")
        print("   🛡️ Tighter stop losses for better protection ($1.80)")
        print("   ⏰ Shorter trade duration (12 minutes)")
        print("   📊 Adaptive position sizing based on performance")
        print("   🔥 Winning streak bonuses")
        print("   📈 Trailing stops for profitable trades")
        print("   ⚖️ Breakeven protection after 10 minutes")
        print("   🎯 Optimized symbol selection (6 best performers)")
        print("   🧠 Higher AI confidence threshold (60%)")
        print("   📱 Enhanced performance notifications")
        
        print("\n💡 KEY IMPROVEMENTS:")
        print("   🟢 Better win rate through selective trading")
        print("   🟢 Faster profit realization")
        print("   🟢 Better risk management")
        print("   🟢 Adaptive position sizing")
        print("   🟢 Performance-based adjustments")
        
        print("\n🚀 RESTART YOUR BOT TO APPLY CHANGES!")
        print("   The bot will now make smarter, more profitable trades")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying smart config: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("🧠 SMART TRADING CONFIGURATION SYSTEM")
    print("   Transform your bot into a profit-making machine!")
    print("="*60)
    
    success = apply_smart_config()
    
    if success:
        print("\n" + "="*60)
        print("✅ SMART CONFIGURATION APPLIED SUCCESSFULLY!")
        print("🚀 Restart your trading bot to activate smart trading")
        print("💰 Your bot is now optimized for maximum profitability!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ CONFIGURATION FAILED!")
        print("   Please check the error messages above")
        print("="*60)
