"""
Quick check to see if MT5 terminal is logged in to your account
"""
import MetaTrader5 as mt5

print("🔄 Checking MT5 terminal status...")

# Initialize connection
if not mt5.initialize():
    print("❌ Cannot connect to MT5 terminal")
    print(f"Error: {mt5.last_error()}")
    exit(1)

print("✅ Connected to MT5 terminal")

# Check current account
account_info = mt5.account_info()
if account_info:
    print(f"📊 Currently logged in account:")
    print(f"   Login: {account_info.login}")
    print(f"   Server: {account_info.server}")
    print(f"   Balance: ${account_info.balance:.2f}")
    print(f"   Name: {account_info.name}")
    
    # Check if it's your demo account
    if account_info.login == 31899532:
        print("✅ Your demo account is already logged in!")
    else:
        print(f"⚠️ Different account logged in. Expected: 31899532, Found: {account_info.login}")
        print("🔄 Attempting to login to your demo account...")
        
        # Try to login to your account
        authorized = mt5.login(31899532, password="Goon22622$", server="Deriv-Demo")
        if authorized:
            print("✅ Successfully logged in to your demo account!")
            account_info = mt5.account_info()
            print(f"   Balance: ${account_info.balance:.2f}")
        else:
            error = mt5.last_error()
            print(f"❌ Login failed: {error}")
else:
    print("❌ No account logged in")

mt5.shutdown()
