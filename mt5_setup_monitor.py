"""
MT5 Visual Setup and Trading Activity Monitor
Shows real-time trading activity and setup verification
"""

import MetaTrader5 as mt5
import time
from datetime import datetime
from typing import Dict, List
from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

class MT5ActivityMonitor:
    """Monitor and display MT5 trading activity"""
    
    def __init__(self):
        self.last_positions = {}
        self.last_orders = {}
        self.activity_log = []
        
    def show_mt5_setup_status(self) -> bool:
        """Show comprehensive MT5 setup status"""
        print("\n🔍 MT5 SETUP VERIFICATION")
        print("=" * 40)
        
        # Check MT5 installation
        try:
            import MetaTrader5 as mt5
            print("✅ MetaTrader5 Python library: Installed")
        except ImportError:
            print("❌ MetaTrader5 Python library: Missing")
            print("   Fix: pip install MetaTrader5")
            return False
        
        # Check MT5 connection
        if not mt5.initialize():
            error = mt5.last_error()
            print("❌ MT5 Terminal: Not connected")
            print(f"   Error: {error}")
            print("\n🔧 REQUIRED ACTIONS:")
            print("1. 📥 Install MT5: https://www.metatrader5.com/")
            print("2. 🚀 Open MT5 terminal")
            print("3. 🔐 Login to your Deriv account")
            print("4. ⚙️  Enable Expert Advisors:")
            print("   Tools → Options → Expert Advisors")
            print("   ✓ Allow automated trading")
            print("   ✓ Allow DLL imports")
            print("5. 🔄 Restart MT5 terminal")
            return False
        
        print("✅ MT5 Terminal: Connected")
        
        # Get account details
        account = mt5.account_info()
        if account:
            print(f"✅ Account: {account.login} ({account.server})")
            print(f"💰 Balance: ${account.balance:.2f}")
            print(f"💎 Equity: ${account.equity:.2f}")
            print(f"📊 Margin: ${account.margin:.2f}")
        
        # Check symbols
        print("\n📊 CHECKING DERIV SYMBOLS:")
        symbols = mt5.symbols_get()
        deriv_symbols = []
        
        if symbols:
            for symbol in symbols:
                name = symbol.name.lower()
                if any(keyword in name for keyword in 
                      ['volatility', 'boom', 'crash', 'step', 'jump']):
                    deriv_symbols.append(symbol.name)
        
        if deriv_symbols:
            print(f"✅ Found {len(deriv_symbols)} Deriv symbols:")
            for symbol in deriv_symbols[:10]:  # Show first 10
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    print(f"   📈 {symbol} - Spread: {symbol_info.spread}")
        else:
            print("❌ No Deriv symbols found!")
            print("   Contact Deriv support to enable synthetic indices")
        
        # Check Expert Advisors setting
        terminal_info = mt5.terminal_info()
        if terminal_info:
            if terminal_info.trade_allowed:
                print("✅ Automated Trading: Enabled")
            else:
                print("❌ Automated Trading: Disabled")
                print("   Fix: Tools → Options → Expert Advisors → Allow automated trading")
        
        mt5.shutdown()
        return len(deriv_symbols) > 0
    
    def monitor_trading_activity(self) -> None:
        """Real-time monitoring of trading activity"""
        print("\n🔄 STARTING REAL-TIME ACTIVITY MONITOR")
        print("=" * 45)
        print("📊 Watching for trading activity...")
        print("⏹️  Press Ctrl+C to stop monitoring")
        print("-" * 45)
        
        try:
            if not mt5.initialize():
                print("❌ Cannot connect to MT5")
                return
            
            while True:
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Check positions
                positions = mt5.positions_get()
                if positions:
                    print(f"\n🎯 [{current_time}] ACTIVE POSITIONS:")
                    for pos in positions:
                        profit_emoji = "📈" if pos.profit > 0 else "📉"
                        print(f"   {profit_emoji} {pos.symbol}: {pos.type_time} "
                             f"Vol: {pos.volume} P&L: ${pos.profit:.2f}")
                
                # Check recent orders
                orders = mt5.history_orders_get(
                    datetime.now() - timedelta(minutes=5), 
                    datetime.now()
                )
                if orders and len(orders) > len(self.last_orders):
                    print(f"\n💼 [{current_time}] NEW ORDERS:")
                    for order in orders[len(self.last_orders):]:
                        print(f"   🔔 {order.symbol}: {order.type_time} "
                             f"Vol: {order.volume_initial}")
                    self.last_orders = list(orders)
                
                # Check account changes
                account = mt5.account_info()
                if account:
                    print(f"\r💰 Balance: ${account.balance:.2f} | "
                         f"Equity: ${account.equity:.2f} | "
                         f"Margin: ${account.margin:.2f} | "
                         f"Time: {current_time}", end="")
                
                time.sleep(2)  # Update every 2 seconds
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Activity monitoring stopped")
        finally:
            mt5.shutdown()
    
    def test_trading_functions(self) -> None:
        """Test basic trading functions"""
        print("\n🧪 TESTING TRADING FUNCTIONS")
        print("=" * 35)
        
        if not mt5.initialize():
            print("❌ Cannot connect to MT5")
            return
        
        try:
            # Test symbol info
            symbol = "Volatility 100 Index"
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                print(f"✅ Symbol Info: {symbol}")
                print(f"   Bid: {symbol_info.bid}")
                print(f"   Ask: {symbol_info.ask}")
                print(f"   Spread: {symbol_info.spread}")
            else:
                print(f"❌ Cannot get info for {symbol}")
            
            # Test tick data
            ticks = mt5.copy_ticks_from(symbol, datetime.now(), 5, mt5.COPY_TICKS_ALL)
            if ticks is not None and len(ticks) > 0:
                print(f"✅ Live Data: Last tick at {ticks[-1].time}")
            else:
                print("❌ Cannot get live tick data")
            
            # Test account info
            account = mt5.account_info()
            if account:
                print(f"✅ Account Access: Balance ${account.balance:.2f}")
            else:
                print("❌ Cannot access account info")
                
        finally:
            mt5.shutdown()

def main():
    """Main function for MT5 setup and monitoring"""
    monitor = MT5ActivityMonitor()
    
    print("🎯" * 20)
    print("📊 MT5 SETUP & ACTIVITY MONITOR")
    print("🎯" * 20)
    
    while True:
        print("\n📋 CHOOSE ACTION:")
        print("1. 🔍 Verify MT5 Setup")
        print("2. 🔄 Monitor Trading Activity")
        print("3. 🧪 Test Trading Functions")
        print("4. 📖 Show Setup Guide")
        print("5. ❌ Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            monitor.show_mt5_setup_status()
        elif choice == "2":
            monitor.monitor_trading_activity()
        elif choice == "3":
            monitor.test_trading_functions()
        elif choice == "4":
            print_detailed_setup_guide()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

def print_detailed_setup_guide():
    """Print detailed MT5 setup guide"""
    print("\n📚 COMPLETE MT5 SETUP GUIDE")
    print("=" * 50)
    
    print("\n🔢 STEP 1: INSTALL METATRADER 5")
    print("-" * 30)
    print("• Go to: https://www.metatrader5.com/")
    print("• Download MT5 for Windows")
    print("• Install and run the application")
    
    print("\n🔐 STEP 2: CONNECT DERIV ACCOUNT")
    print("-" * 35)
    print("• In MT5: File → Login to Trade Account")
    print("• Enter your Deriv MT5 credentials:")
    print("  - Login: Your Deriv MT5 account number")
    print("  - Password: Your Deriv MT5 password")
    print("  - Server: Will auto-detect Deriv server")
    print("• Click 'Login'")
    
    print("\n⚙️  STEP 3: ENABLE AUTOMATED TRADING")
    print("-" * 40)
    print("• In MT5: Tools → Options")
    print("• Go to 'Expert Advisors' tab")
    print("• Check these boxes:")
    print("  ✓ Allow automated trading")
    print("  ✓ Allow DLL imports")
    print("  ✓ Allow imports of external experts")
    print("• Click 'OK'")
    print("• RESTART MT5 completely")
    
    print("\n📊 STEP 4: VERIFY DERIV SYMBOLS")
    print("-" * 35)
    print("• In Market Watch, look for:")
    print("  - Volatility 100 Index")
    print("  - Volatility 75 Index")
    print("  - Boom 1000 Index")
    print("  - Crash 1000 Index")
    print("• If missing, contact Deriv support")
    
    print("\n🚀 STEP 5: RUN THE BOT")
    print("-" * 25)
    print("• Keep MT5 running and logged in")
    print("• Run: python unified_launcher.py")
    print("• Choose option 2 (MetaTrader 5)")
    print("• Watch for trading activity!")

if __name__ == "__main__":
    main()
