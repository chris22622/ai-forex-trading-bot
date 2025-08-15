"""
Check MT5 Terminal Status and Connection
"""


import MetaTrader5 as mt5


def check_mt5_terminal():
    """Check if MT5 terminal is available and can be initialized"""
    print("🔍 Checking MT5 Terminal...")

    # Check if MT5 package is working
    print(f"📦 MetaTrader5 package version: {mt5.__version__}")

    # Attempt initialization
    print("\n🔧 Attempting MT5 initialization...")
    if mt5.initialize():
        print("✅ MT5 initialized successfully!")

        # Get terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"📱 Terminal Name: {terminal_info.name}")
            print(f"🏗️ Build: {terminal_info.build}")
            print(f"🔗 Connected: {terminal_info.connected}")
            print(f"📁 Data Path: {terminal_info.data_path}")
            print(f"🌐 Trade Allowed: {terminal_info.trade_allowed}")
        else:
            print("❌ Could not get terminal info")

        # Get current account info (if any)
        account_info = mt5.account_info()
        if account_info:
            print(f"\n👤 Current Account: {account_info.login}")
            print(f"🏦 Server: {account_info.server}")
            print(f"💰 Balance: ${account_info.balance:.2f}")
        else:
            print("\n❌ No account currently logged in")
            print("🔑 Will need to provide login credentials")

        # Test login with credentials
        print("\n🔐 Testing login with demo credentials...")
        login = 31899532
        password = "Goon22622$"
        server = "Deriv-Demo"

        if mt5.login(login, password=password, server=server):
            print("✅ Login successful!")
            account_info = mt5.account_info()
            if account_info:
                print(f"👤 Logged in as: {account_info.login}")
                print(f"🏦 Server: {account_info.server}")
                print(f"💰 Balance: ${account_info.balance:.2f}")
        else:
            error = mt5.last_error()
            print(f"❌ Login failed: {error}")

        # Shutdown
        mt5.shutdown()

    else:
        error = mt5.last_error()
        print(f"❌ MT5 initialization failed: {error}")

        if error[0] == -6:
            print("\n💡 Error -6 suggests:")
            print("   1. MetaTrader 5 terminal is not installed")
            print("   2. MetaTrader 5 terminal is not running")
            print("   3. Authorization/permission issues")
            print("\n🔧 Suggested fixes:")
            print("   1. Download and install MetaTrader 5 from https://www.metatrader5.com/")
            print("   2. Start MetaTrader 5 terminal manually")
            print("   3. Make sure you can log in manually first")

        return False

if __name__ == "__main__":
    check_mt5_terminal()
