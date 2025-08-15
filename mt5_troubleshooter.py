#!/usr/bin/env python3
"""
MT5 Connection Troubleshooter
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def check_mt5_process():
    """Check MT5 process and restart if needed"""
    print("🔍 Checking MT5 process...")
    try:
        # Check if MT5 is running
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq terminal64.exe'], 
                              capture_output=True, text=True, shell=True)
        
        if 'terminal64.exe' in result.stdout:
            print("✅ MT5 Terminal is running")
            
            # Kill MT5 to force fresh login
            print("🔄 Killing MT5 to force fresh login...")
            subprocess.run(['taskkill', '/F', '/IM', 'terminal64.exe'], 
                          capture_output=True, shell=True)
            time.sleep(3)
            print("✅ MT5 terminated")
        else:
            print("❌ MT5 Terminal not running")
        
        # Start MT5
        print("🚀 Starting MT5 Terminal...")
        mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
        
        if os.path.exists(mt5_path):
            subprocess.Popen([mt5_path])
            print("✅ MT5 started")
            print("⏳ Waiting 10 seconds for MT5 to initialize...")
            time.sleep(10)
        else:
            print(f"❌ MT5 not found at: {mt5_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error managing MT5 process: {e}")
        return False

def test_mt5_connection():
    """Test MT5 connection with current config"""
    print("\n🔐 Testing MT5 connection...")
    
    try:
        import MetaTrader5 as mt5
        print("✅ MT5 library imported")
        
        # Initialize MT5
        if not mt5.initialize():
            error = mt5.last_error()
            print(f"❌ MT5 initialize failed: {error}")
            return False
        
        print("✅ MT5 initialized")
        
        # Import config
        sys.path.append(os.getcwd())
        from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
        
        print(f"📊 Using credentials:")
        print(f"   Login: {MT5_LOGIN}")
        print(f"   Password: {'*' * len(str(MT5_PASSWORD))}")
        print(f"   Server: {MT5_SERVER}")
        
        # Try login
        print("🔐 Attempting login...")
        login_result = mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
        
        if login_result:
            print("✅ MT5 LOGIN SUCCESSFUL!")
            
            # Get account info
            account = mt5.account_info()
            if account:
                print(f"💰 Balance: ${account.balance:.2f}")
                print(f"💳 Equity: ${account.equity:.2f}")
                print(f"🏦 Server: {account.server}")
                print(f"💱 Currency: {account.currency}")
                print(f"🔗 Connection: {account.connected}")
            
            mt5.shutdown()
            return True
        else:
            error = mt5.last_error()
            print(f"❌ MT5 login failed: {error}")
            
            if error[0] == -6:
                print("\n💡 ERROR -6 SOLUTIONS:")
                print("1. 🎯 Manual login required first")
                print("2. 🔄 Restart MT5 completely")
                print("3. 🔐 Check password: Goon22622$")
                print("4. 🌐 Check internet connection")
                print("5. 📱 Verify account status on Deriv.com")
            
            mt5.shutdown()
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

def main():
    """Main troubleshooting function"""
    print("🔧 MT5 CONNECTION TROUBLESHOOTER")
    print("=" * 40)
    print(f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Restart MT5
    if not check_mt5_process():
        print("\n❌ Failed to manage MT5 process")
        return
    
    # Step 2: Test connection
    if test_mt5_connection():
        print("\n✅ SUCCESS! MT5 connection working!")
        print("🚀 You can now run your trading bot!")
    else:
        print("\n❌ MT5 connection still failing")
        print("\n📋 MANUAL STEPS REQUIRED:")
        print("1. 🎯 Open MT5 terminal manually")
        print("2. 🔐 Login with these credentials:")
        print("   - Login: 62233085")
        print("   - Password: Goon22622$")
        print("   - Server: DerivVU-Server")
        print("3. ✅ Check 'Save account information'")
        print("4. 🔄 Run this script again")

if __name__ == "__main__":
    main()
