#!/usr/bin/env python3
"""Simple MT5 connection test"""

try:
    import MetaTrader5 as mt5
    print("MT5 library imported successfully")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        mt5.shutdown()
        exit(1)
    
    print("MT5 initialized successfully")
    
    # Try to login
    from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    
    login_result = mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
    if login_result:
        print("MT5 login successful!")
        
        # Get account info
        account = mt5.account_info()
        if account:
            print(f"Balance: ${account.balance:.2f}")
            print(f"Equity: ${account.equity:.2f}")
            print(f"Server: {account.server}")
            print(f"Currency: {account.currency}")
        else:
            print("Could not get account info")
    else:
        error = mt5.last_error()
        print(f"MT5 login failed: {error}")
        
        if error[0] == -6:
            print("Error -6 means: Authorization failed")
            print("Solutions:")
            print("1. Check your login credentials")
            print("2. Make sure MT5 terminal is running")
            print("3. Try manual login in MT5 first")
            print("4. Restart MT5 terminal")
    
    mt5.shutdown()
    
except ImportError:
    print("MetaTrader5 library not installed")
    print("Install with: pip install MetaTrader5")
except Exception as e:
    print(f"Unexpected error: {e}")
