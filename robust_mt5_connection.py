"""
Robust MT5 Connection Manager
Handles MT5 initialization with proper error handling and retry logic
"""

import MetaTrader5 as mt5
import time
import logging
from typing import Optional, Tuple
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustMT5Manager:
    """Robust MT5 Connection Manager with retry logic"""
    
    def __init__(self):
        self.connected = False
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔧 MT5 initialization attempt {attempt + 1}/{self.max_retries}")
                
                # Shutdown any existing connection first
                mt5.shutdown()
                time.sleep(1)
                
                # Initialize MT5
                if mt5.initialize():
                    logger.info("✅ MT5 initialized successfully")
                    return True
                else:
                    error = mt5.last_error()
                    logger.error(f"❌ MT5 initialization failed: {error}")
                    
                    if attempt < self.max_retries - 1:
                        logger.info(f"⏳ Waiting {self.retry_delay} seconds before retry...")
                        time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"❌ Exception during initialization: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return False
    
    def login_to_account(self) -> bool:
        """Login to MT5 account"""
        try:
            logger.info(f"🔐 Attempting login to account {MT5_LOGIN}")
            logger.info(f"🏦 Server: {MT5_SERVER}")
            
            # Attempt login
            if mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                logger.info("✅ Login successful!")
                
                # Verify account info
                account_info = mt5.account_info()
                if account_info:
                    logger.info(f"👤 Account: {account_info.login}")
                    logger.info(f"🏦 Server: {account_info.server}")
                    logger.info(f"💰 Balance: ${account_info.balance:.2f}")
                    logger.info(f"💱 Currency: {account_info.currency}")
                    logger.info(f"🎯 Trade Mode: {account_info.trade_mode}")
                    return True
                else:
                    logger.error("❌ Could not retrieve account info after login")
                    return False
            else:
                error = mt5.last_error()
                logger.error(f"❌ Login failed: {error}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Exception during login: {e}")
            return False
    
    def connect_with_full_retry(self) -> bool:
        """Full connection process with retry logic"""
        logger.info("🚀 Starting robust MT5 connection...")
        
        # Step 1: Initialize MT5
        if not self.initialize_mt5():
            logger.error("❌ Failed to initialize MT5 after all retries")
            return False
        
        # Step 2: Login to account
        if not self.login_to_account():
            logger.error("❌ Failed to login to MT5 account")
            mt5.shutdown()
            return False
        
        # Step 3: Test connection
        if self.test_connection():
            self.connected = True
            logger.info("🎉 Successfully connected to MT5!")
            return True
        else:
            logger.error("❌ Connection test failed")
            mt5.shutdown()
            return False
    
    def test_connection(self) -> bool:
        """Test MT5 connection functionality"""
        try:
            logger.info("🧪 Testing MT5 connection...")
            
            # Test 1: Terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info:
                logger.info(f"📱 Terminal: {terminal_info.name} Build {terminal_info.build}")
                logger.info(f"🔗 Connected: {terminal_info.connected}")
            else:
                logger.warning("⚠️ Could not get terminal info")
                return False
            
            # Test 2: Account info
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"💰 Account balance: ${account_info.balance:.2f}")
            else:
                logger.warning("⚠️ Could not get account info")
                return False
            
            # Test 3: Get symbols
            symbols = mt5.symbols_get()
            if symbols:
                logger.info(f"📈 Available symbols: {len(symbols)}")
            else:
                logger.warning("⚠️ Could not get symbols")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def get_account_info(self) -> Optional[dict]:
        """Get current account information"""
        if not self.connected:
            return None
        
        try:
            account_info = mt5.account_info()
            if account_info:
                return {
                    'login': account_info.login,
                    'server': account_info.server,
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'currency': account_info.currency,
                    'trade_mode': account_info.trade_mode
                }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
        
        return None
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("📴 Disconnected from MT5")

def main():
    """Test the robust MT5 manager"""
    print("🚀 Testing Robust MT5 Connection Manager")
    print("=" * 50)
    
    manager = RobustMT5Manager()
    
    # Attempt connection
    if manager.connect_with_full_retry():
        print("\n✅ MT5 CONNECTION SUCCESSFUL!")
        
        # Get account info
        account_info = manager.get_account_info()
        if account_info:
            print("\n📊 Account Information:")
            for key, value in account_info.items():
                print(f"   {key}: {value}")
        
        # Keep connection active for a moment
        print("\n⏳ Keeping connection active for 5 seconds...")
        time.sleep(5)
        
        # Disconnect
        manager.disconnect()
        print("✅ Test completed successfully!")
        
    else:
        print("\n❌ MT5 CONNECTION FAILED!")
        print("🔧 Please check:")
        print("   1. MetaTrader 5 terminal is running")
        print("   2. Account credentials are correct")
        print("   3. Server is accessible")

if __name__ == "__main__":
    main()
