#!/usr/bin/env python3
"""
Setup Checker for Deriv Trading Bot
Quick verification of all components
"""

import sys
import os
from typing import Dict, List, Any, Tuple, Callable

def print_header() -> None:
    print("=" * 60)
    print("🚀 DERIV TRADING BOT - SETUP CHECKER")
    print("🔍 Verifying all components...")
    print("=" * 60)

def check_imports():
    """Check if all required modules can be imported"""
    print("\n📦 Checking module imports...")
    
    modules = [
        ('config', 'Configuration'),
        ('indicators', 'Technical Indicators'),
        ('ai_model', 'AI Model'),
        ('telegram_bot', 'Telegram Integration'),
        ('utils', 'Utilities'),
        ('main', 'Main Bot')
    ]
    
    all_good = True
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            all_good = False
        except Exception as e:
            print(f"⚠️ {description}: {e}")
    
    return all_good

def check_configuration() -> bool:
    """Check configuration settings"""
    print("\n⚙️ Checking configuration...")
    
    try:
        from config import (
            DERIV_API_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
            TRADE_AMOUNT, DEFAULT_SYMBOL, PAPER_TRADING
        )
        
        issues: List[str] = []
        
        if not DERIV_API_TOKEN or len(DERIV_API_TOKEN) < 10:
            issues.append("Deriv API token not configured")
        else:
            print("✅ Deriv API token configured")
        
        if not TELEGRAM_BOT_TOKEN or len(TELEGRAM_BOT_TOKEN) < 10:
            issues.append("Telegram bot token not configured")
        else:
            print("✅ Telegram bot token configured")
        
        if not TELEGRAM_CHAT_ID or len(str(TELEGRAM_CHAT_ID)) < 5:
            issues.append("Telegram chat ID not configured")
        else:
            print("✅ Telegram chat ID configured")
        
        print(f"✅ Trade amount: ${TRADE_AMOUNT}")
        print(f"✅ Trading symbol: {DEFAULT_SYMBOL}")
        print(f"✅ Paper trading: {'Enabled' if PAPER_TRADING else 'Disabled'}")
        
        if issues:
            print("\n⚠️ Configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_directories():
    """Check if necessary directories exist"""
    print("\n📁 Checking directories...")
    
    directories = ['logs']
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}/ directory exists")
        else:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ {directory}/ directory created")
    
    return True

def check_dependencies():
    """Check external dependencies"""
    print("\n📚 Checking dependencies...")
    
    required_packages = [
        'websockets',
        'pandas',
        'numpy', 
        'ta',
        'requests',
        'pytz'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\n💡 To install missing packages, run:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def test_components():
    """Test key components"""
    print("\n🧪 Testing components...")
    
    try:
        # Test technical indicators
        from indicators import TechnicalIndicators
        indicators = TechnicalIndicators()
        print("✅ Technical indicators can be created")
        
        # Test AI model
        from ai_model import TradingAI
        ai = TradingAI()
        print("✅ AI model can be created")
        
        # Test utilities
        from utils import logger, trade_logger
        print("✅ Utilities initialized")
        
        # Test Telegram (if configured)
        try:
            from telegram_bot import TelegramNotifier
            notifier = TelegramNotifier()
            if notifier.enabled:
                print("✅ Telegram integration ready")
            else:
                print("⚠️ Telegram integration disabled")
        except Exception as e:
            print(f"⚠️ Telegram integration: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def print_summary(all_checks_passed):
    """Print final summary"""
    print("\n" + "=" * 60)
    
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Your Deriv Trading Bot is ready to run!")
        print("\n🚀 To start the bot:")
        print("   • Windows: Double-click start_bot.bat")
        print("   • Command line: python main.py")
        print("\n📖 Documentation: README.md")
        print("⚙️ Configuration: config.py")
    else:
        print("❌ SETUP INCOMPLETE")
        print("⚠️ Please fix the issues above before running the bot")
        print("\n📋 Common fixes:")
        print("   • Install missing dependencies: pip install -r requirements.txt")
        print("   • Configure API tokens in config.py")
        print("   • Check Python version (need 3.9+)")
    
    print("=" * 60)

def main():
    """Main setup checker"""
    print_header()
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Imports", check_imports),
        ("Configuration", check_configuration),
        ("Directories", check_directories),
        ("Components", test_components)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            all_passed = False
    
    print_summary(all_passed)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
