"""
Minimal MT5 Integration Test
"""

import MetaTrader5 as mt5
import asyncio
import logging
from typing import Optional

# Import config for MT5 credentials
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

# Simple logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(handler)

class SimpleMT5Interface:
    """Minimal MT5 Interface for testing"""
    
    def __init__(self):
        self.connected = False
        logger.info("SimpleMT5Interface created")
        
    async def initialize(self) -> bool:
        """Initialize MT5 connection"""
        try:
            logger.info("Initializing MT5...")
            
            # Run MT5 init in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, mt5.initialize)
            
            if result:
                logger.info("✅ MT5 initialized successfully")
                
                # Attempt login with credentials
                logger.info(f"Attempting login with account: {MT5_LOGIN}")
                login_result = await loop.run_in_executor(
                    None, mt5.login, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
                )
                
                if login_result:
                    logger.info("✅ Successfully logged in to MT5")
                    
                    # Get account info
                    account_info = await loop.run_in_executor(None, mt5.account_info)
                    if account_info:
                        logger.info(f"Account: {account_info.login}")
                        logger.info(f"Balance: ${account_info.balance:.2f}")
                        logger.info(f"Server: {account_info.server}")
                        self.connected = True
                        return True
                    else:
                        logger.error("❌ No account info after login")
                        return False
                else:
                    error = await loop.run_in_executor(None, mt5.last_error)
                    logger.error(f"❌ MT5 login failed: {error}")
                    return False
            else:
                error = await loop.run_in_executor(None, mt5.last_error)
                logger.error(f"❌ MT5 initialization failed: {error}")
                return False
                
        except Exception as e:
            logger.error(f"❌ MT5 initialization error: {e}")
            return False
    
    async def get_account_balance(self) -> float:
        """Get account balance"""
        try:
            loop = asyncio.get_event_loop()
            account_info = await loop.run_in_executor(None, mt5.account_info)
            if account_info:
                return float(account_info.balance)
            return 0.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            loop = asyncio.get_event_loop()
            symbol_tick = await loop.run_in_executor(None, mt5.symbol_info_tick, symbol)
            if symbol_tick:
                return float(symbol_tick.bid)
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

# Test function
async def test_simple_mt5():
    """Test the simple MT5 interface"""
    print("🔧 Testing Simple MT5 Interface...")
    
    interface = SimpleMT5Interface()
    print("✅ Interface created")
    
    success = await interface.initialize()
    print(f"Connection result: {success}")
    
    if success:
        balance = await interface.get_account_balance()
        print(f"Balance: ${balance:.2f}")
        
        price = await interface.get_current_price("Volatility 75 Index")
        print(f"V75 Price: {price}")

if __name__ == "__main__":
    print("Testing SimpleMT5Interface...")
    asyncio.run(test_simple_mt5())
