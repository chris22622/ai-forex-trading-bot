#!/usr/bin/env python3
"""
🔧 MT5 CONNECTION FIXER
Diagnose and fix MT5 authorization issues
"""

import os
import subprocess
import sys
from datetime import datetime


def check_mt5_terminal():
    """Check if MT5 terminal is running and logged in"""
    print("🔍 CHECKING MT5 TERMINAL STATUS")
    print("=" * 40)

    # Check if MT5 process is running
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq terminal64.exe'],
                              capture_output=True, text=True, shell=True)

        if 'terminal64.exe' in result.stdout:
            print("✅ MT5 Terminal (terminal64.exe) is running")
        else:
            print("❌ MT5 Terminal is NOT running")
            print("💡 Please start MT5 terminal manually first")
            return False
    except Exception as e:
        print(f"⚠️ Could not check MT5 process: {e}")

    return True

def check_mt5_credentials():
    """Check MT5 credentials from config"""
    print("\n🔐 CHECKING MT5 CREDENTIALS")
    print("=" * 40)

    try:
        # Import config
        sys.path.append(os.getcwd())
        from config import MT5_DEMO_MODE, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

        print(f"📊 Account: {MT5_LOGIN}")
        print(f"🖥️ Server: {MT5_SERVER}")
        print(f"🎯 Demo Mode: {MT5_DEMO_MODE}")
        print(f"🔑 Password: {'*' * len(str(MT5_PASSWORD))}")

        # Validate credentials
        if not MT5_LOGIN or MT5_LOGIN == 0:
            print("❌ Invalid MT5_LOGIN")
            return False

        if not MT5_PASSWORD or len(str(MT5_PASSWORD)) < 5:
            print("❌ Invalid MT5_PASSWORD (too short)")
            return False

        if not MT5_SERVER:
            print("❌ Invalid MT5_SERVER")
            return False

        print("✅ All credentials look valid")
        return True

    except ImportError as e:
        print(f"❌ Could not import config: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking credentials: {e}")
        return False

def provide_mt5_solutions():
    """Provide solutions for MT5 connection issues"""
    print("\n🛠️ MT5 CONNECTION SOLUTIONS")
    print("=" * 40)

    print("1. 🎯 MANUAL LOGIN TO MT5:")
    print("   • Open MT5 terminal manually")
    print("   • Go to File → Login to Trade Account")
    print("   • Enter your credentials:")
    print("     - Account: 62233085")
    print("     - Password: ChrisAI2024!")
    print("     - Server: DerivVU-Server")
    print("   • Make sure 'Save account information' is checked")
    print("   • Click 'OK' and verify connection")

    print("\n2. 🔄 RESTART MT5 TERMINAL:")
    print("   • Close MT5 completely")
    print("   • Wait 10 seconds")
    print("   • Restart MT5")
    print("   • Login again")

    print("\n3. 🌐 CHECK INTERNET CONNECTION:")
    print("   • Verify your internet is stable")
    print("   • Try accessing Deriv.com in browser")
    print("   • Check if any firewall is blocking MT5")

    print("\n4. 📱 VERIFY ACCOUNT STATUS:")
    print("   • Login to Deriv.com website")
    print("   • Check your account is active")
    print("   • Verify MT5 account is enabled")

    print("\n5. 🚨 DEMO MODE ALTERNATIVE:")
    print("   • If live account fails, we can switch to demo")
    print("   • Demo accounts usually connect easier")
    print("   • Good for testing the bot functionality")

def create_mt5_test_script():
    """Create a simple MT5 connection test"""
    test_script = '''#!/usr/bin/env python3
"""Simple MT5 connection test"""

try:
    import MetaTrader5 as mt5
    print("✅ MetaTrader5 library imported successfully")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialize failed: {mt5.last_error()}")
        mt5.shutdown()
        exit(1)
    
    print("✅ MT5 initialized successfully")
    
    # Try to login
    from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    
    login_result = mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
    if login_result:
        print("✅ MT5 login successful!")
        
        # Get account info
        account = mt5.account_info()
        if account:
            print(f"💰 Balance: ${account.balance:.2f}")
            print(f"💳 Equity: ${account.equity:.2f}")
            print(f"🏦 Server: {account.server}")
            print(f"💱 Currency: {account.currency}")
        else:
            print("⚠️ Could not get account info")
    else:
        error = mt5.last_error()
        print(f"❌ MT5 login failed: {error}")
        
        if error[0] == -6:
            print("💡 Error -6 means: Authorization failed")
            print("   → Check your login credentials")
            print("   → Make sure MT5 terminal is running")
            print("   → Try manual login in MT5 first")
    
    mt5.shutdown()
    
except ImportError:
    print("❌ MetaTrader5 library not installed")
    print("💡 Install with: pip install MetaTrader5")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
'''

    with open('test_mt5_connection.py', 'w') as f:
        f.write(test_script)

    print("\n📝 Created test_mt5_connection.py")
    print("💡 Run this to test MT5 connection: python test_mt5_connection.py")

def main():
    """Main function"""
    print("🔧 MT5 CONNECTION DIAGNOSTIC TOOL")
    print("=" * 50)
    print(f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Check terminal
    if not check_mt5_terminal():
        print("\n❌ CRITICAL: MT5 Terminal not running!")
        print("🚀 ACTION REQUIRED: Start MT5 terminal first")

    # Step 2: Check credentials
    if not check_mt5_credentials():
        print("\n❌ CRITICAL: Invalid MT5 credentials!")
        print("🚀 ACTION REQUIRED: Fix config.py credentials")

    # Step 3: Provide solutions
    provide_mt5_solutions()

    # Step 4: Create test script
    create_mt5_test_script()

    print("\n" + "=" * 50)
    print("🎯 QUICK FIX STEPS:")
    print("1. Start MT5 terminal manually")
    print("2. Login manually with your credentials")
    print("3. Run: python test_mt5_connection.py")
    print("4. If test passes, restart your trading bot")
    print("=" * 50)

if __name__ == "__main__":
    main()
