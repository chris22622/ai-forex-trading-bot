#!/usr/bin/env python3
"""
Quick status checker for the trading bot
"""

import MetaTrader5 as mt5
import time
from datetime import datetime

def check_mt5_status():
    """Check if MT5 is connected and what account is logged in"""
    print("🔍 Checking MT5 Status...")
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 not initialized")
        return False
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ No account logged in")
        mt5.shutdown()
        return False
    
    print(f"✅ MT5 Status:")
    print(f"   Account: {account_info.login}")
    print(f"   Server: {account_info.server}")
    print(f"   Balance: ${account_info.balance:.2f}")
    print(f"   Currency: {account_info.currency}")
    print(f"   Trade Mode: {account_info.trade_mode}")
    
    # Check if trading is allowed
    if account_info.trade_allowed:
        print("✅ Trading allowed")
    else:
        print("❌ Trading not allowed")
    
    # Get terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"   Terminal: {terminal_info.name}")
        print(f"   Build: {terminal_info.build}")
        print(f"   Connected: {terminal_info.connected}")
    
    # Check symbol availability
    symbol = "Volatility 75 Index"
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        print(f"✅ {symbol} available for trading")
        print(f"   Spread: {symbol_info.spread}")
        print(f"   Volume min: {symbol_info.volume_min}")
    else:
        print(f"❌ {symbol} not available")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    print(f"🕐 Status check at {datetime.now().strftime('%H:%M:%S')}")
    check_mt5_status()
