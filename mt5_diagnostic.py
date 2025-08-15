#!/usr/bin/env python3
"""
🔍 MT5 CONNECTION DIAGNOSTIC
Check MT5 status and connection details
"""

import MetaTrader5 as mt5

def diagnose_mt5():
    """Diagnose MT5 connection issues"""
    print("🔍 MT5 DIAGNOSTIC REPORT")
    print("=" * 50)
    
    # Check if MT5 package is available
    print(f"📦 MT5 Package Available: {mt5 is not None}")
    
    # Try to initialize
    print("🚀 Initializing MT5...")
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        print(f"   Error: {mt5.last_error()}")
        return False
    
    print("✅ MT5 initialized successfully")
    
    # Get version info
    version = mt5.version()
    print(f"📋 MT5 Version: {version}")
    
    # Get terminal info
    terminal = mt5.terminal_info()
    if terminal:
        print(f"💻 Terminal: {terminal.name}")
        print(f"🏢 Company: {terminal.company}")
        print(f"📁 Data Path: {terminal.data_path}")
        print(f"🌐 Connected: {terminal.connected}")
    
    # Try to get account info without login
    account = mt5.account_info()
    if account:
        print(f"💰 Current Account: {account.login}")
        print(f"💵 Balance: ${account.balance}")
        print(f"🏦 Server: {account.server}")
    else:
        print("❌ No account connected")
    
    # Check available symbols
    symbols = mt5.symbols_get()
    if symbols:
        print(f"📊 Available Symbols: {len(symbols)}")
        # Check for R_75 specifically
        r75_symbols = [s for s in symbols if 'R_75' in s.name]
        if r75_symbols:
            print(f"✅ R_75 found: {r75_symbols[0].name}")
        else:
            print("❌ R_75 not found in available symbols")
            # Show first 10 symbols
            print("📋 First 10 symbols:")
            for i, symbol in enumerate(symbols[:10]):
                print(f"   {i+1}. {symbol.name}")
    else:
        print("❌ No symbols available")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    diagnose_mt5()
