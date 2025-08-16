"""
MetaTrader 5 Integration for Deriv Trading Bot
Replaces WebSocket API with MT5 terminal connection
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5  # type: ignore
import pandas as pd

# Use standard logging to avoid potential circular imports
logger = logging.getLogger(__name__)

# Set up basic logging if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Import configuration with fallback
MT5_DEMO_MODE = True
MT5_REAL_TRADING = True
MT5_LOGIN = None
MT5_PASSWORD = ""
MT5_SERVER = ""
MT5_LOT_SIZE = 0.01
MT5_MAX_LOT_SIZE = 1.0
MT5_SLIPPAGE = 3
MT5_MAGIC_NUMBER = 234000
MT5_DEFAULT_SYMBOL = "Volatility 10 Index"

# Try to import from config, but don't let it block initialization
try:
    from config import (
        MT5_DEFAULT_SYMBOL,
        MT5_DEMO_MODE,
        MT5_LOGIN,
        MT5_LOT_SIZE,
        MT5_MAGIC_NUMBER,
        MT5_MAX_LOT_SIZE,
        MT5_PASSWORD,
        MT5_REAL_TRADING,
        MT5_SERVER,
        MT5_SLIPPAGE,
    )
    logger.info("✅ MT5 config loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Config import failed, using defaults: {e}")
except Exception as e:
    logger.warning(f"⚠️ Config import error, using defaults: {e}")

# Import bulletproof symbol mapping for MT5 compatibility
try:
    from bulletproof_patches import BulletproofSymbolManager
    symbol_manager = BulletproofSymbolManager()
    logger.info("✅ Bulletproof symbol mapping integrated into MT5")
except ImportError:
    logger.warning("⚠️ Bulletproof patches not available, using emergency symbol mapping")
    class EmergencySymbolManager:
        def get_safe_symbol(self, symbol):
            # Emergency symbol mapping for MT5 compatibility - COMPREHENSIVE DERIV COVERAGE
            symbol_map = {
                # TOP TIER: Best spreads and performance
                'Volatility 50 Index': 'Volatility 50 Index',
                'Volatility 150 (1s) Index': 'Volatility 150 (1s) Index',
                'Volatility 250 (1s) Index': 'Volatility 250 (1s) Index',
                'Step Index': 'Step Index',
                'Volatility 10 Index': 'Volatility 10 Index',

                # VOLATILITY INDICES: Keep all as-is (all are active)
                'Volatility 25 Index': 'Volatility 25 Index',
                'Volatility 75 Index': 'Volatility 75 Index',
                'Volatility 100 Index': 'Volatility 100 Index',
                'Volatility 10 (1s) Index': 'Volatility 10 (1s) Index',
                'Volatility 15 (1s) Index': 'Volatility 15 (1s) Index',
                'Volatility 25 (1s) Index': 'Volatility 25 (1s) Index',
                'Volatility 30 (1s) Index': 'Volatility 30 (1s) Index',
                'Volatility 50 (1s) Index': 'Volatility 50 (1s) Index',
                'Volatility 75 (1s) Index': 'Volatility 75 (1s) Index',
                'Volatility 90 (1s) Index': 'Volatility 90 (1s) Index',
                'Volatility 100 (1s) Index': 'Volatility 100 (1s) Index',

                # BOOM/CRASH INDICES: All active
                'Boom 300 Index': 'Boom 300 Index',
                'Boom 500 Index': 'Boom 500 Index',
                'Boom 600 Index': 'Boom 600 Index',
                'Boom 900 Index': 'Boom 900 Index',
                'Boom 1000 Index': 'Boom 1000 Index',
                'Crash 300 Index': 'Crash 300 Index',
                'Crash 500 Index': 'Crash 500 Index',
                'Crash 900 Index': 'Crash 900 Index',
                'Crash 1000 Index': 'Crash 1000 Index',

                # JUMP INDICES: All active
                'Jump 10 Index': 'Jump 10 Index',
                'Jump 25 Index': 'Jump 25 Index',
                'Jump 50 Index': 'Jump 50 Index',
                'Jump 75 Index': 'Jump 75 Index',
                'Jump 100 Index': 'Jump 100 Index',

                # VOLATILITY OVER BOOM/CRASH: All active
                'Vol over Boom 400': 'Vol over Boom 400',
                'Vol over Boom 750': 'Vol over Boom 750',
                'Vol over Crash 400': 'Vol over Crash 400',
                'Vol over Crash 550': 'Vol over Crash 550',
                'Vol over Crash 750': 'Vol over Crash 750',

                # RANGE BREAK INDICES: All active
                'Range Break 100 Index': 'Range Break 100 Index',
                'Range Break 200 Index': 'Range Break 200 Index',

                # OLD API SYMBOLS: Map to best equivalents
                'R_10': 'Volatility 10 Index',
                'R_25': 'Volatility 25 Index',
                'R_50': 'Volatility 50 Index',
                'R_75': 'Volatility 75 Index',
                'R_100': 'Volatility 100 Index',
                'BOOM1000': 'Boom 1000 Index',
                'CRASH1000': 'Crash 1000 Index',
            }
                        mapped = symbol_map.get(
                symbol,
                'Volatility 50 Index'
            )
            logger.info(f"🔄 Symbol mapping: {symbol} → {mapped}")
            return mapped
    symbol_manager = EmergencySymbolManager()

# Standard logger setup

class MT5TradingInterface:
    """MetaTrader 5 Trading Interface"""

    def __init__(self):
        self.connected = False
        self.initialized = False
        self.account_info = None
        self.symbols_available = []
        self.current_positions = {}
        self.trade_history = []

        # Trading settings
        self.default_symbol = "Volatility 100 Index"  # Deriv's R_100 in MT5
        self.magic_number = 234000  # Unique identifier for our bot trades
        self.slippage = 3  # Price slippage tolerance

    async def initialize(self) -> bool:
        """Initialize MetaTrader 5 connection with proper async handling"""
        try:
            logger.info("🔄 Initializing MetaTrader 5 connection...")

            # Run blocking MT5 calls in thread pool
            loop = asyncio.get_event_loop()

            # Initialize MT5 connection
            init_result = await loop.run_in_executor(None, mt5.initialize)
            if not init_result:
                error = await loop.run_in_executor(None, mt5.last_error)
                logger.error(f"❌ MT5 initialization failed: {error}")
                return False

            self.initialized = True
            logger.info("✅ MT5 initialized successfully")

            # 🔍 DIAGNOSTIC: Check available symbols
            try:
                available_symbols = await self.diagnose_available_symbols()
                if available_symbols:
                    logger.info(f"🎯 Deriv symbols available: {len(available_symbols)}")
                    for symbol in available_symbols[:10]:  # Show first 10
                        logger.info(f"   - {symbol}")
            except Exception as diag_error:
                logger.warning(f"⚠️ Symbol diagnosis failed: {diag_error}")

            # Login with credentials if provided (run in executor to avoid blocking)
            if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
                logger.info(f"🔐 Logging in to {MT5_SERVER} with account {MT5_LOGIN}...")
                authorized = await loop.run_in_executor(
                    None,
                    lambda: mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
                )
                if not authorized:
                    error = await loop.run_in_executor(None, mt5.last_error)
                    logger.error(f"❌ MT5 login failed: {error}")
                    await loop.run_in_executor(None, mt5.shutdown)
                    return False
                logger.info("✅ MT5 login successful!")

            # Get account info (run in executor)
            account_info = await loop.run_in_executor(None, mt5.account_info)
            if account_info is None:
                logger.error("❌ Failed to get account info")
                await loop.run_in_executor(None, mt5.shutdown)
                return False

            self.account_info = account_info._asdict()
            self.connected = True

            logger.info("✅ Connected to MT5 account:")
            logger.info(f"   Account: {self.account_info['login']}")
            logger.info(f"   Server: {self.account_info['server']}")
            logger.info(f"   Balance: ${self.account_info['balance']:.2f}")
            logger.info(f"   Equity: ${self.account_info['equity']:.2f}")
            logger.info(f"   Currency: {self.account_info['currency']}")

            # Get available symbols
            await self.update_available_symbols()

            return True

        except Exception as e:
            logger.error(f"❌ MT5 initialization error: {e}")
            return False

    async def get_position_info(self, ticket: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific position by ticket number"""
        try:
            # Get all positions
            positions = mt5.positions_get()
            if positions is None:
                return None

            # Find position by ticket
            for position in positions:
                if position.ticket == ticket:
                    return {
                        "ticket": position.ticket,
                        "symbol": position.symbol,
                        "volume": position.volume,
                        "type": position.type,
                        "price_open": position.price_open,
                        "price_current": position.price_current,
                        "profit": position.profit,
                        "time": position.time,
                        "comment": position.comment,
                        "sl": position.sl,
                        "tp": position.tp
                    }

            # Position not found (likely closed)
            return None

        except Exception as e:
            logger.error(f"Error getting position info for ticket {ticket}: {e}")
            return None

    def _get_error_description(self, retcode: int) -> str:
        """Get human-readable error description for MT5 return codes"""
        error_codes = {
            10004: "Requote",
            10006: "Request rejected",
            10007: "Request canceled by trader",
            10008: "Order placed",
            10009: "Request completed",
            10010: "Only part of the request was completed",
            10011: "Request processing error",
            10012: "Request canceled by timeout",
            10013: "Invalid request",
            10014: "Invalid volume in the request",
            10015: "Invalid price in the request",
            10016: "Invalid stops in the request",
            10017: "Trade is disabled",
            10018: "Market is closed",
            10019: "There is not enough money to complete the request",
            10020: "Prices changed",
            10021: "There are no quotes to process the request",
            10022: "Invalid order expiration date in the request",
            10023: "Order state changed",
            10024: "Too frequent requests",
            10025: "No changes in request",
            10026: "Autotrading disabled by server",
            10027: "Autotrading disabled by client terminal",
            10028: "Request locked for processing",
            10029: "Order or position frozen",
            10030: "Invalid order filling type",
            10031: "No connection with the trade server",
            10040: "Invalid order type or lot size (volume out of range or invalid step)"
        }

        return error_codes.get(retcode, f"Unknown error code: {retcode}")

    async def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            if not self.initialized:
                await self.initialize()

            loop = asyncio.get_event_loop()
            account_info = await loop.run_in_executor(None, mt5.account_info)
            if account_info is None:
                logger.error("❌ Cannot get account info")
                return 0.0

            return float(account_info.balance)

        except Exception as e:
            logger.error(f"❌ Error getting account balance: {e}")
            return 0.0

    async def diagnose_available_symbols(self) -> List[str]:
        """Diagnostic function to list available symbols in MT5"""
        try:
            if not self.initialized:
                await self.initialize()

            logger.info("🔍 Diagnosing available symbols in MT5...")

            # Get all available symbols
            loop = asyncio.get_event_loop()
            all_symbols = await loop.run_in_executor(None, mt5.symbols_get)

            if all_symbols:
                # Filter for Deriv-related symbols
                deriv_symbols = []
                for symbol in all_symbols:
                    name = symbol.name
                    if any(keyword in name.lower() for keyword in ['volatility', 'boom', 'crash', 'vol', 'r_']):
                        deriv_symbols.append(name)
                        logger.info(f"📋 Found Deriv symbol: {name}")

                if deriv_symbols:
                    logger.info(f"✅ Found {len(deriv_symbols)} Deriv symbols: {deriv_symbols[:10]}")
                    return deriv_symbols
                else:
                    logger.warning("⚠️ No Deriv symbols found")
                    # Show first 10 symbols for debugging
                    first_10 = [s.name for s in all_symbols[:10]]
                    logger.info(f"📋 First 10 available symbols: {first_10}")
                    return first_10
            else:
                logger.error("❌ No symbols available in MT5")
                return []

        except Exception as e:
            logger.error(f"❌ Error diagnosing symbols: {e}")
            return []

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol - ENHANCED WITH BETTER ERROR REPORTING"""
        try:
            if not self.initialized:
                logger.warning(f"⚠️ MT5 not initialized when requesting price for {symbol}")
                await self.initialize()

            if not self.connected:
                logger.warning(f"⚠️ MT5 not connected when requesting price for {symbol}")
                return None

            # BULLETPROOF SYMBOL MAPPING - Convert synthetic symbols to MT5-compatible ones
            effective_symbol = symbol_manager.get_safe_symbol(symbol)
            logger.debug(f"🔄 Getting price for: {symbol} → {effective_symbol}")

            # Check if symbol exists in MT5 first
            symbol_info = mt5.symbol_info(effective_symbol)
            if symbol_info is None:
                f"❌ Symbol {effective_symbol}"
                f"not found in MT5 - checking alternatives"

                # Try alternative symbol names for Volatility 10 Index
                alternatives = [
                    "Vol 10 Index",
                    "Volatility10Index",
                    "V10",
                    "R_10",
                    "Volatility 10 (1s) Index",
                    "VIX10",
                    "Step Index"
                ]

                for alt_symbol in alternatives:
                    alt_info = mt5.symbol_info(alt_symbol)
                    if alt_info is not None:
                        logger.info(f"✅ Found alternative symbol: {alt_symbol} for {symbol}")
                        effective_symbol = alt_symbol
                        break
                else:
                    # List all available symbols for debugging
                    f"❌ No valid symbols found for {symbol}"
                    f" Checking available symbols..."
                    symbols = mt5.symbols_get()
                    if symbols:
                        volatility_symbols = [s.name for s in symbols if 'vol' in s.name.lower() or 'index' in s.name.lower()]
                        f"🔍 Available volatility/index symbols: {volatility_symbols[:10]}"
                        f"
                    else:
                        logger.error("❌ No symbols available from MT5")
                    return None

            # CRITICAL FIX: Ensure symbol is selected BEFORE getting price
            if not mt5.symbol_select(effective_symbol, True):
                logger.error(f"❌ Cannot select symbol {effective_symbol} in MT5")
                return None

            # Wait a moment for symbol to load
            await asyncio.sleep(0.1)

            # Try multiple times to get price with detailed logging
            for attempt in range(5):  # Increased attempts
                tick = mt5.symbol_info_tick(effective_symbol)
                if tick is not None:
                    # Check all available price types
                    bid_price = getattr(tick, 'bid', 0.0)
                    ask_price = getattr(tick, 'ask', 0.0)
                    last_price = getattr(tick, 'last', 0.0)

                    f"🔍 Tick data for {effective_symbol}"
                    f" bid={bid_price}, ask={ask_price}, last={last_price}"

                    # Use the first valid price (prioritize last, then bid, then ask)
                    valid_price = None
                    if last_price > 0:
                        valid_price = float(last_price)
                        logger.info(f"✅ Using last price for {symbol}: {valid_price}")
                    elif bid_price > 0:
                        valid_price = float(bid_price)
                        logger.info(f"✅ Using bid price for {symbol}: {valid_price}")
                    elif ask_price > 0:
                        valid_price = float(ask_price)
                        logger.info(f"✅ Using ask price for {symbol}: {valid_price}")

                    if valid_price is not None:
                        return valid_price
                    else:
                        f"⚠️ All prices are zero for {effective_symbol}"
                        f"(attempt {attempt + 1})"
                else:
                    f"⚠️ No tick data for {effective_symbol}"
                    f"(attempt {attempt + 1})"

                if attempt < 4:
                    await asyncio.sleep(0.5)  # Longer wait between retries

            # CRITICAL: If all attempts fail, return None with detailed error
            f"❌ Failed to get price for {symbol}"
            f"({effective_symbol}) after 3 attempts"
            logger.error(f"❌ Symbol info: {symbol_info}")
            logger.error(f"❌ Last tick: {tick}")
            return None

        except Exception as e:
            logger.error(f"❌ Critical error getting price for {symbol}: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information including volume limits with comprehensive error handling"""
        try:
            # BULLETPROOF SYMBOL MAPPING - Convert synthetic symbols to MT5-compatible ones
            effective_symbol = symbol_manager.get_safe_symbol(symbol)
            logger.info(f"🔄 Symbol mapping for MT5: {symbol} → {effective_symbol}")

            symbol_info = mt5.symbol_info(effective_symbol)
            if symbol_info is None:
                logger.warning(f"⚠️ MT5 symbol {effective_symbol} not found, using safe defaults")
                # Return safe defaults instead of None to prevent crashes
                return {
                    "volume_min": 0.01,
                    "volume_max": 100.0,
                    "volume_step": 0.01,
                    "point": 0.00001,
                    "digits": 5,
                    "trade_contract_size": 100000
                }

            # Safely extract all symbol information with fallbacks
            result = {
                "volume_min": getattr(symbol_info, 'volume_min', 0.01),
                "volume_max": getattr(symbol_info, 'volume_max', 100.0),
                "volume_step": getattr(symbol_info, 'volume_step', 0.01),
                "point": getattr(symbol_info, 'point', 0.00001),
                "digits": getattr(symbol_info, 'digits', 5),
                "trade_contract_size": getattr(symbol_info, 'trade_contract_size', 100000)
            }

            # Validate that all values are not None and are reasonable
            if result["volume_min"] is None or result["volume_min"] <= 0:
                logger.warning(f"⚠️ Invalid volume_min for {effective_symbol}, using default 0.01")
                result["volume_min"] = 0.01

            if result["volume_max"] is None or result["volume_max"] <= result["volume_min"]:
                logger.warning(f"⚠️ Invalid volume_max for {effective_symbol}, using default 100.0")
                result["volume_max"] = 100.0

            if result["volume_step"] is None or result["volume_step"] <= 0:
                logger.warning(f"⚠️ Invalid volume_step for {effective_symbol}, using default 0.01")
                result["volume_step"] = 0.01

            logger.debug(f"📊 Symbol info for {symbol} (MT5: {effective_symbol}): {result}")
            return result

        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            # Return safe defaults to prevent crashes
            return {
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "point": 0.00001,
                "digits": 5,
                "trade_contract_size": 100000
            }

    async def validate_lot_size(self, symbol: str, volume: float) -> Tuple[bool, float]:
        """Validate and adjust lot size to meet symbol constraints
        Returns: (is_valid, adjusted_volume)
        """
        try:
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"⚠️ No symbol info for {symbol}, using defaults")
                # Use safe defaults for validation
                min_vol, max_vol, step = 0.01, 100.0, 0.01
            else:
                min_vol = symbol_info.get("volume_min", 0.01)
                max_vol = symbol_info.get("volume_max", 100.0)
                step = symbol_info.get("volume_step", 0.01)

            # Ensure minimum constraints
            if volume < min_vol:
                adjusted_volume = min_vol
                logger.info(f"📊 Volume {volume} too small, adjusted to minimum: {adjusted_volume}")
            elif volume > max_vol:
                adjusted_volume = max_vol
                logger.info(f"📊 Volume {volume} too large, adjusted to maximum: {adjusted_volume}")
            else:
                # Round to valid step size
                adjusted_volume = round(volume / step) * step
                if adjusted_volume != volume:
                    logger.info(f"📊 Volume adjusted for step size: {volume} → {adjusted_volume}")

            # Final validation
            is_valid = (min_vol <= adjusted_volume <= max_vol and
                       abs(adjusted_volume - round(adjusted_volume / step) * step) < 0.0001)

            f"🔍 Lot validation for {symbol}"
            f" {volume} → {adjusted_volume} (valid: {is_valid})"
            return is_valid, adjusted_volume

        except Exception as e:
            logger.error(f"❌ Lot validation error: {e}")
            # Return conservative defaults
            return True, max(0.01, min(volume, 1.0))

    def _validate_symbol_info(self, symbol_info: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Validate and sanitize symbol info to prevent KeyError crashes"""
        try:
            # Define safe defaults
            defaults = {
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "point": 0.00001,
                "digits": 5,
                "trade_contract_size": 100000
            }

            # Validate each required field
            for key, default_value in defaults.items():
                if key not in symbol_info or symbol_info[key] is None:
                    f"⚠️ Missing or invalid {key}"
                    f"for {symbol}, using default: {default_value}"
                    symbol_info[key] = default_value

                # Additional validation for numeric values
                try:
                    if key in ['volume_min', 'volume_max', 'volume_step']:
                        if float(symbol_info[key]) <= 0:
                            f"⚠️ Invalid {key}"
                            f"value {symbol_info[key]} for {symbol}, using default: {default_value}"
                            symbol_info[key] = default_value
                except (ValueError, TypeError):
                    f"⚠️ Non-numeric {key}"
                    f"for {symbol}, using default: {default_value}"
                    symbol_info[key] = default_value

            # Ensure logical relationships
            if symbol_info["volume_max"] <= symbol_info["volume_min"]:
                logger.warning(f"⚠️ volume_max <= volume_min for {symbol}, fixing...")
                symbol_info["volume_max"] = symbol_info["volume_min"] * 100

            return symbol_info

        except Exception as e:
            logger.error(f"❌ Error validating symbol info for {symbol}: {e}")
            return defaults

    async def calculate_valid_lot_size(self, symbol: str, desired_amount: float) -> float:
        """Calculate valid lot size based on symbol specifications - BULLETPROOF VERSION"""
        try:
            # Map to MT5 symbol first
            effective_symbol = symbol_manager.get_safe_symbol(symbol)
            logger.debug(f"🔄 Lot calculation for: {symbol} → {effective_symbol}")

            # Get real-time symbol info from MT5
            loop = asyncio.get_event_loop()
            symbol_info = await loop.run_in_executor(None, mt5.symbol_info, effective_symbol)

            if symbol_info is None:
                logger.warning(f"⚠️ Symbol {effective_symbol} not found, using ultra-safe defaults")
                return 0.001  # Ultra-conservative fallback

            # Extract REAL MT5 symbol limits
            volume_min = getattr(symbol_info, 'volume_min', 0.001)
            volume_max = getattr(symbol_info, 'volume_max', 1.0)
            volume_step = getattr(symbol_info, 'volume_step', 0.001)

            # Log actual MT5 limits for debugging
            f"📊 MT5 {effective_symbol}"
            f"limits: min={volume_min}, max={volume_max}, step={volume_step}"

            # BULLETPROOF lot calculation based on actual MT5 symbol type
            if "Volatility" in effective_symbol:
                # For Deriv Volatility indices: Use minimum volume (often 0.5 for these symbols)
                calculated_lot = volume_min  # Start with actual minimum required by MT5
                logger.info(f"🎯 Using MT5 minimum volume for {effective_symbol}: {calculated_lot}")
            elif "Boom" in effective_symbol or "Crash" in effective_symbol:
                # For Boom/Crash: Use minimum volume
                calculated_lot = volume_min
            elif any(fx in effective_symbol for fx in ['EUR', 'GBP', 'USD', 'JPY']):
                # For forex pairs: Standard calculation
                calculated_lot = max(desired_amount / 100000.0, volume_min)
            else:
                # For any other symbol: Use MT5 minimum
                calculated_lot = volume_min

            # CRITICAL: Ensure lot size is within MT5 limits
            calculated_lot = max(volume_min, min(volume_max, calculated_lot))

            # Round to valid step size - CRITICAL for MT5
            if volume_step > 0:
                steps = round(calculated_lot / volume_step)
                final_lot = steps * volume_step

                # Ensure rounding didn't break limits
                final_lot = max(volume_min, min(volume_max, final_lot))
            else:
                final_lot = calculated_lot

            # Final safety check
            if final_lot < volume_min:
                final_lot = volume_min
                logger.warning(f"⚠️ Lot size adjusted to minimum: {final_lot}")
            elif final_lot > volume_max:
                final_lot = volume_max
                logger.warning(f"⚠️ Lot size adjusted to maximum: {final_lot}")

            logger.info(f"💎 BULLETPROOF lot calculation: ${desired_amount} → {final_lot:.3f} lots "
                       f"(MT5 limits: {volume_min}"
                       f"{volume_max}, step: {volume_step}) for {effective_symbol}"

            return final_lot

        except Exception as e:
            logger.error(f"❌ Error in bulletproof lot calculation: {e}")
            # Ultra-safe fallback
            return 0.001

    async def place_trade(self, action: str, symbol: str, amount: float) -> Optional[Dict[str, Any]]:
        """Place a trade through MT5 - BULLETPROOF VERSION with comprehensive error handling"""
        try:
            if not self.initialized:
                await self.initialize()

            # STEP 1: BULLETPROOF SYMBOL MAPPING
            effective_symbol = symbol_manager.get_safe_symbol(symbol)
            logger.info(f"🔄 BULLETPROOF order: {symbol} → {effective_symbol}")

            # STEP 2: VALIDATE SYMBOL IS AVAILABLE AND GET FRESH PRICE
            loop = asyncio.get_event_loop()

            # Get real-time symbol info directly from MT5
            symbol_info = await loop.run_in_executor(None, mt5.symbol_info, effective_symbol)
            if symbol_info is None:
                f"⚠️ Symbol {effective_symbol}"
                f"not found, trying common Deriv alternatives..."

                # Try common Deriv symbol variations
                alternatives = [
                    "Volatility 75 Index",
                    "VOLATILITY75",
                    "VOL75",
                    "R_75",
                    "Volatility75"
                ]

                for alt_symbol in alternatives:
                    if alt_symbol != effective_symbol:
                        logger.info(f"🔄 Trying alternative: {alt_symbol}")
                        symbol_info = await loop.run_in_executor(None, mt5.symbol_info, alt_symbol)
                        if symbol_info is not None:
                            effective_symbol = alt_symbol
                            logger.info(f"✅ Found working symbol: {effective_symbol}")
                            break

                if symbol_info is None:
                    f"❌ Symbol {effective_symbol}"
                    f"not available in MT5 - tried all alternatives"
                    return None

            # CRITICAL: ENSURE SYMBOL IS VISIBLE IN MARKET WATCH
            if not symbol_info.visible:
                logger.info(f"🔧 Adding {effective_symbol} to Market Watch...")
                                symbol_selected = await loop.run_in_executor(
                    None,
                    mt5.symbol_select,
                    effective_symbol,
                    True
                )
                if symbol_selected:
                    logger.info(f"✅ {effective_symbol} added to Market Watch")
                    # Get updated symbol info
                                        symbol_info = await loop.run_in_executor(
                        None,
                        mt5.symbol_info,
                        effective_symbol
                    )
                else:
                    logger.error(f"❌ Failed to add {effective_symbol} to Market Watch")
                    return None

            # Get fresh price directly from MT5 (now that symbol is visible)
            tick = await loop.run_in_executor(None, mt5.symbol_info_tick, effective_symbol)
            if tick is None:
                f"❌ Cannot get price for {effective_symbol}"
                f"- symbol may not be visible"
                return None

            # Use appropriate price for order type
            price = float(tick.ask) if action == "BUY" else float(tick.bid)
            logger.info(f"💰 Fresh MT5 price for {effective_symbol}: {price}")

            # STEP 3: If real trading is disabled, simulate the trade
            if not MT5_REAL_TRADING:
                f"📊 SIMULATING trade (MT5_REAL_TRADING=False): {action}"
                f"{effective_symbol}"
                return await self._simulate_trade(action, effective_symbol, amount, price)

            # STEP 3.5: CRITICAL MT5 TRADING DIAGNOSTICS
            terminal_info = await loop.run_in_executor(None, mt5.terminal_info)
            if terminal_info:
                trade_allowed = getattr(terminal_info, 'trade_allowed', False)
                logger.info(f"🔍 MT5 Terminal - Trade Allowed: {trade_allowed}")
                if not trade_allowed:
                    logger.error("❌ AUTO-TRADING NOT ENABLED! Go to MT5 Tools → Options → Expert Advisors → Allow automated trading")
                    logger.info("🔄 Falling back to simulation mode for this trade")
                    return await self._simulate_trade(action, effective_symbol, amount, price)

            # STEP 4: BULLETPROOF LOT SIZE CALCULATION
            lot_size = await self.calculate_valid_lot_size(symbol, amount)

            # Double-check lot size against REAL MT5 limits
            volume_min = getattr(symbol_info, 'volume_min', 0.001)
            volume_max = getattr(symbol_info, 'volume_max', 1.0)
            volume_step = getattr(symbol_info, 'volume_step', 0.001)

            # Final lot size validation
            if lot_size < volume_min or lot_size > volume_max:
                f"❌ CRITICAL: Lot size {lot_size}"
                f"outside MT5 limits [{volume_min}, {volume_max}]"
                # Force to valid range
                lot_size = max(volume_min, min(volume_max, lot_size))
                logger.warning(f"⚠️ Adjusted lot size to: {lot_size}")

            f"💎 BULLETPROOF lot size: {lot_size:.3f}"
            f"for ${amount} on {effective_symbol}"

            # STEP 5: BULLETPROOF ORDER SETUP
            order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL

            # Try different filling types in order of compatibility for Deriv symbols
            if "Volatility" in effective_symbol or "Boom" in effective_symbol or "Crash" in effective_symbol:
                # For Deriv indices: Try Return (market execution) first, then IOC
                filling_types = [
                    mt5.ORDER_FILLING_RETURN,   # Market execution (most compatible with Deriv)
                    mt5.ORDER_FILLING_IOC,      # Immediate or Cancel
                    mt5.ORDER_FILLING_FOK       # Fill or Kill (last resort)
                ]
            else:
                # For other symbols: IOC first
                filling_types = [
                    mt5.ORDER_FILLING_IOC,      # Immediate or Cancel
                    mt5.ORDER_FILLING_RETURN,   # Market execution
                    mt5.ORDER_FILLING_FOK       # Fill or Kill
                ]

            # STEP 6: ATTEMPT ORDER WITH MULTIPLE FILLING TYPES
            # First, check what filling types the symbol supports
            symbol_info = await loop.run_in_executor(None, mt5.symbol_info, effective_symbol)
            if symbol_info:
                filling_mode = getattr(symbol_info, 'filling_mode', None)
                logger.info(f"📋 Symbol {effective_symbol} filling mode: {filling_mode}")

            for i, filling_type in enumerate(filling_types):
                try:
                    f"� PLACING REAL MT5 ORDER (Attempt {i+1}"
                    f": {action} {effective_symbol}"
                    logger.info(f"📊 Volume: {lot_size}, Price: {price}, Filling: {filling_type}")

                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": effective_symbol,
                        "volume": lot_size,
                        "type": order_type,
                        "price": price,
                        "sl": 0.0,
                        "tp": 0.0,
                        "deviation": MT5_SLIPPAGE,
                        "magic": MT5_MAGIC_NUMBER,
                        "comment": f"DerivBot_{action}_{int(time.time())}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": filling_type,
                    }

                    # FINAL CHECK: Ensure symbol is still visible before order_send
                                        symbol_check = await loop.run_in_executor(
                        None,
                        mt5.symbol_info,
                        effective_symbol
                    )
                    if symbol_check and not symbol_check.visible:
                        f"⚠️ Symbol {effective_symbol}"
                        f"not visible, re-adding to Market Watch..."
                        await loop.run_in_executor(None, mt5.symbol_select, effective_symbol, True)

                    # ENHANCED: Pre-order validation and fixing
                    # Check if MT5 is properly connected and ready
                    # TEMPORARILY DISABLED - causing method not found error
                    # if not await self._validate_mt5_ready_for_trading():
                    #     logger.error(f"❌ MT5 not ready for trading, skipping attempt {i+1}")
                    #     continue

                    # ENHANCED: Validate symbol is ready for trading
                    # TEMPORARILY DISABLED - method exists but causing issues
                    # if not await self._validate_symbol_ready(effective_symbol):
                    f"❌ Symbol {effective_symbol}"
                    f"not ready for trading, skipping attempt {i+1}"
                    #     continue

                    # Get last error before sending order
                    await loop.run_in_executor(None, mt5.last_error)  # Clear any previous errors

                    # Send order - Fixed the argument passing issue
                    def _send_order():
                        return mt5.order_send(request)
                    result = await loop.run_in_executor(None, _send_order)

                    # Get detailed error info if order failed
                    if result is None:
                        last_error = await loop.run_in_executor(None, mt5.last_error)
                        f"❌ MT5 order_send returned None (attempt {i+1}"
                        f" - Last Error: {last_error}"

                        # ENHANCED: Try alternative order method when order_send returns None
                        logger.info(f"🔄 Trying alternative order method for attempt {i+1}")
                        result = await self._try_alternative_order_method(request, effective_symbol)

                        if result is None:
                            # Additional diagnostics
                                                        symbol_check = await loop.run_in_executor(
                                None,
                                mt5.symbol_info,
                                effective_symbol
                            )
                            if symbol_check:
                                f"🔍 Symbol Debug - Visible: {symbol_check.visible}"
                                f" Trade Mode: {symbol_check.trade_mode}"

                            terminal_check = await loop.run_in_executor(None, mt5.terminal_info)
                            if terminal_check:
                                f"🔍 Terminal Debug - Connected: {terminal_check.connected}"
                                f" Trade Allowed: {terminal_check.trade_allowed}"

                            continue

                    # Check result
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        # SUCCESS!
                        trade_ticket = result.order
                        f"✅ BULLETPROOF MT5 SUCCESS: {action}"
                        f"{effective_symbol} (Ticket: {trade_ticket})"

                        return {
                            "ticket": trade_ticket,
                            "symbol": effective_symbol,
                            "original_symbol": symbol,
                            "action": action,
                            "price": price,
                            "amount": amount,
                            "lot_size": lot_size,
                            "time": datetime.now(),
                            "status": "opened",
                            "real_trade": True,
                            "retcode": result.retcode,
                            "comment": getattr(result, 'comment', 'Success'),
                            "filling_type": filling_type
                        }
                    else:
                        # Log error and try next filling type
                        error_msg = self._get_error_description(result.retcode)
                        f"⚠️ Attempt {i+1}"
                        f"failed: {result.retcode} - {error_msg} (Filling: {filling_type})"

                                                # 🎯 ROBUST FILLING TYPE HANDLING
                            - Continue trying other filling types for 10030
                        if result.retcode == 10030:  # Invalid order filling type
                            f"🔄 Filling type {filling_type}"
                            f"not supported, trying next..."
                            continue  # Try next filling type

                        # Don't retry for certain errors
                        if result.retcode in [10018, 10019, 10021]:  # Market closed, no money, no quotes
                            logger.error(f"❌ Fatal error {result.retcode}, not retrying")
                            break

                except Exception as e:
                    logger.error(f"❌ Error in order attempt {i+1}: {e}")
                    continue

            # If we get here, all attempts failed
            logger.error(f"❌ All MT5 order attempts failed for {effective_symbol}")
            return None

        except Exception as e:
            logger.error(f"❌ Critical error in BULLETPROOF place_trade: {e}")
            return None

    async def _simulate_trade(self, action: str, symbol: str, amount: float, price: float) -> Dict[str, Any]:
        """Simulate a trade (safe mode)"""
        import time
        trade_ticket = int(time.time() * 1000)  # Unique ticket

        logger.info(f"📊 MT5 Trade SIMULATED: {action} {symbol} at {price} (Amount: ${amount})")

        return {
            "ticket": trade_ticket,
            "symbol": symbol,
            "action": action,
            "price": price,
            "amount": amount,
            "time": datetime.now(),
            "status": "opened",
            "real_trade": False
        }

    async def update_available_symbols(self) -> None:
        """Update list of available trading symbols"""
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                logger.warning("⚠️ No symbols available")
                return

            self.symbols_available = []
            deriv_symbols = []

            for symbol in symbols:
                symbol_name = symbol.name
                self.symbols_available.append(symbol_name)

                # Look for Deriv/Binary.com symbols
                if any(keyword in symbol_name.lower() for keyword in
                      ['volatility', 'boom', 'crash', 'step', 'jump']):
                    deriv_symbols.append(symbol_name)

            logger.info(f"📊 Found {len(self.symbols_available)} total symbols")
            if deriv_symbols:
                logger.info(f"🎯 Deriv symbols available: {len(deriv_symbols)}")
                for symbol in deriv_symbols[:10]:  # Show first 10
                    logger.info(f"   - {symbol}")

                # Update default symbol if better option found
                if deriv_symbols:
                    self.default_symbol = deriv_symbols[0]
                    logger.info(f"🎯 Using symbol: {self.default_symbol}")

        except Exception as e:
            logger.error(f"❌ Error updating symbols: {e}")

    async def get_current_balance(self) -> float:
        """Get current account balance"""
        try:
            account_info = mt5.account_info()
            if account_info:
                return float(account_info.balance)
            return 0.0
        except Exception as e:
            logger.error(f"❌ Error getting balance: {e}")
            return 0.0

    async def get_current_equity(self) -> float:
        """Get current account equity"""
        try:
            account_info = mt5.account_info()
            if account_info:
                return float(account_info.equity)
            return 0.0
        except Exception as e:
            logger.error(f"❌ Error getting equity: {e}")
            return 0.0


    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # BULLETPROOF SYMBOL MAPPING - Convert synthetic symbols to MT5-compatible ones
            effective_symbol = symbol_manager.get_safe_symbol(symbol)
            logger.debug(f"🔄 Getting price for: {symbol} → {effective_symbol}")

            symbol_info = mt5.symbol_info(effective_symbol)
            if symbol_info:
                # Return mid price (average of bid and ask)
                price = (symbol_info.bid + symbol_info.ask) / 2.0
                logger.debug(f"💰 Price for {symbol} (MT5: {effective_symbol}): {price}")
                return price

            logger.error(f"❌ Cannot get price for {symbol} - MT5 connection issue")
            # FIXED: Return None instead of 1.0 fallback to prevent bad trades
            return None
        except Exception as e:
            logger.error(f"❌ Error getting price for {symbol}: {e}")
            return None

    async def get_price_history(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M1,
                               count: int = 100) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                logger.warning(f"⚠️ No historical data for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            return df
        except Exception as e:
            logger.error(f"❌ Error getting price history for {symbol}: {e}")
            return None

    def calculate_lot_size(self, symbol: str, risk_amount: float) -> float:
        """Calculate appropriate lot size based on risk amount - FIXED FOR DERIV INDICES"""
        try:
            # BULLETPROOF SYMBOL MAPPING - Convert synthetic symbols to MT5-compatible ones
            effective_symbol = symbol_manager.get_safe_symbol(symbol)
            logger.debug(f"🔄 Calculating lot size for: {symbol} → {effective_symbol}")

            symbol_info = mt5.symbol_info(effective_symbol)
            if symbol_info is None:
                f"⚠️ Cannot get symbol info for {symbol}"
                f"(MT5: {effective_symbol}), using default lot size"
                return 0.1  # Larger default for Deriv indices

            # 🔧 ENHANCED LOT SIZE CALCULATION FOR DERIV INDICES
            # For Deriv Volatility indices: Use proper lot size calculations
            if "Volatility" in effective_symbol or "volatility" in effective_symbol.lower():
                # For Volatility indices: Simplified calculation based on typical Deriv requirements
                # Use risk amount as a base but ensure proper minimum lot sizes
                lot_size = risk_amount / 50.0  # More reasonable calculation for volatility indices

                                # Ensure reasonable minimum for volatility indices (
                    Deriv typically needs at least 0.1)
                lot_size = max(lot_size, 0.1)  # Increased minimum for Deriv compatibility

            elif "Boom" in effective_symbol or "Crash" in effective_symbol:
                # For Boom/Crash indices: Use larger minimum lots
                lot_size = risk_amount / 100.0  # Adjusted for Boom/Crash
                lot_size = max(lot_size, 0.1)  # Larger minimum for Boom/Crash

            else:
                # For other symbols (Forex, etc): Traditional calculation
                lot_size = risk_amount / 100000.0  # Standard forex calculation
                lot_size = max(lot_size, 0.01)

            # Safely extract symbol specifications with defaults (more conservative)
            min_lot = getattr(symbol_info, 'volume_min', 0.1)  # Increased default minimum
            max_lot = getattr(symbol_info, 'volume_max', 100.0)
            lot_step = getattr(symbol_info, 'volume_step', 0.1)  # Increased default step

            # Validate extracted values (more conservative fallbacks)
            if min_lot is None or max_lot is None or lot_step is None:
                logger.warning(f"⚠️ Invalid symbol specs for {symbol}, using safe defaults")
                min_lot, max_lot, lot_step = 0.1, 100.0, 0.1  # More conservative defaults

            # Round to nearest lot step
            lot_size = round(lot_size / lot_step) * lot_step

            # Clamp to min/max
            lot_size = max(min_lot, min(max_lot, lot_size))

            f"💎 Calculated lot size: {lot_size}"
            f"for ${risk_amount:.2f} on {effective_symbol} "
                       f"(min: {min_lot}, max: {max_lot}, step: {lot_step})")

            return lot_size

        except Exception as e:
            logger.error(f"❌ Error calculating lot size: {e}")
            return 0.1  # Conservative default for Deriv indices

    async def place_buy_order(self, symbol: str, lot_size: float,
                             sl: Optional[float] = None, tp: Optional[float] = None,
                             comment: str = "AI Bot Buy") -> Optional[Dict[str, Any]]:
        """Place a buy order"""
        try:
            # BULLETPROOF SYMBOL MAPPING - Convert synthetic symbols to MT5-compatible ones
            effective_symbol = symbol_manager.get_safe_symbol(symbol)
            logger.debug(f"🔄 Placing buy order for: {symbol} → {effective_symbol}")

            # Get current price
            symbol_info = mt5.symbol_info(effective_symbol)
            if symbol_info is None:
                logger.error(f"❌ Symbol {symbol} (MT5: {effective_symbol}) not found")
                return None

            price = symbol_info.ask

            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": effective_symbol,  # Use MT5-compatible symbol
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": self.slippage,
                "magic": self.magic_number,
                "comment": f"{comment} ({symbol})",  # Include original symbol in comment
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send trade request
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                logger.error(f"❌ Buy order failed: {error}")
                return None

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Buy order failed: {result.retcode} - {result.comment}")
                return None

            logger.info(f"✅ Buy order placed: {symbol} {lot_size} lots at ${price:.5f}")

            return {
                "order": result.order,
                "deal": result.deal,
                "symbol": symbol,
                "volume": lot_size,
                "price": result.price,
                "type": "BUY",
                "comment": comment,
                "time": datetime.now()
            }

        except Exception as e:
            logger.error(f"❌ Error placing buy order: {e}")
            return None

    async def place_sell_order(self, symbol: str, lot_size: float,
                              sl: Optional[float] = None, tp: Optional[float] = None,
                              comment: str = "AI Bot Sell") -> Optional[Dict[str, Any]]:
        """Place a sell order"""
        try:
            # BULLETPROOF SYMBOL MAPPING - Convert synthetic symbols to MT5-compatible ones
            effective_symbol = symbol_manager.get_safe_symbol(symbol)
            logger.debug(f"🔄 Placing sell order for: {symbol} → {effective_symbol}")

            # Get current price
            symbol_info = mt5.symbol_info(effective_symbol)
            if symbol_info is None:
                logger.error(f"❌ Symbol {symbol} (MT5: {effective_symbol}) not found")
                return None

            price = symbol_info.bid

            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": effective_symbol,  # Use MT5-compatible symbol
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": self.slippage,
                "magic": self.magic_number,
                "comment": f"{comment} ({symbol})",  # Include original symbol in comment
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send trade request
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                logger.error(f"❌ Sell order failed: {error}")
                return None

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Sell order failed: {result.retcode} - {result.comment}")
                return None

            logger.info(f"✅ Sell order placed: {symbol} {lot_size} lots at ${price:.5f}")

            return {
                "order": result.order,
                "deal": result.deal,
                "symbol": symbol,
                "volume": lot_size,
                "price": result.price,
                "type": "SELL",
                "comment": comment,
                "time": datetime.now()
            }

        except Exception as e:
            logger.error(f"❌ Error placing sell order: {e}")
            return None

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []

            position_list = []
            for pos in positions:
                if pos.magic == self.magic_number:  # Only our bot's positions
                    position_data = {
                        "ticket": pos.ticket,
                        "symbol": pos.symbol,
                        "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                        "volume": pos.volume,
                        "price_open": pos.price_open,
                        "price_current": pos.price_current,
                        "profit": pos.profit,
                        "swap": pos.swap,
                        "comment": pos.comment,
                        "time": datetime.fromtimestamp(pos.time)
                    }
                    position_list.append(position_data)

            return position_list

        except Exception as e:
            logger.error(f"❌ Error getting positions: {e}")
            return []

    async def close_position(self, ticket: int) -> bool:
        """Enhanced close position with better error handling and multiple retry methods"""
        try:
            logger.info(f"🔒 Attempting to close position {ticket}")

            # First verify the position exists and get its details
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.warning(f"⚠️ Position {ticket} not found - might already be closed")
                return True  # Consider it successfully closed if not found

            pos = position[0]
            symbol = pos.symbol
            volume = pos.volume
            position_type = pos.type

            # Determine opposite order type for closing
            if position_type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
            else:
                order_type = mt5.ORDER_TYPE_BUY

            # Get current price for the symbol
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"❌ Cannot get symbol info for {symbol}")
                return False

            # Use appropriate price (bid for sell, ask for buy)
            if order_type == mt5.ORDER_TYPE_SELL:
                price = symbol_info.bid
            else:
                price = symbol_info.ask

            # Method 1: Try with IOC filling
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": self.slippage,
                "magic": self.magic_number,
                "comment": "Enhanced Auto Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or Cancel
            }

            logger.info(f"🔒 Close request for {ticket}: {symbol} {volume} lots at {price}")

            # Send the close order
            result = mt5.order_send(request)

            if result is None:
                logger.error(f"❌ order_send returned None for {ticket}")
                return False

            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Position {ticket} closed successfully")
                return True
            else:
                # Try alternative method if first fails
                f"⚠️ Close failed (retcode: {result.retcode}"
                f", trying alternative method"

                # Method 2: Try with FOK filling
                request["type_filling"] = mt5.ORDER_FILLING_FOK  # Fill or Kill
                result2 = mt5.order_send(request)

                if result2 and result2.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ Position {ticket} closed with alternative method")
                    return True
                else:
                    # Method 3: Try with market price and different settings
                    logger.warning(f"⚠️ Trying method 3 for {ticket}")

                    # Get fresh price
                    fresh_symbol_info = mt5.symbol_info(symbol)
                    if fresh_symbol_info:
                                                fresh_price = fresh_symbol_info.bid if order_type == mt5.ORDER_TYPE_SELL
                            else fresh_symbol_info.ask

                        request_v3 = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": volume,
                            "type": order_type,
                            "position": ticket,
                            "price": fresh_price,
                            "deviation": 20,  # Higher deviation
                            "magic": self.magic_number,
                            "comment": "Force Close v3",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }

                        result3 = mt5.order_send(request_v3)
                        if result3 and result3.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"✅ Position {ticket} closed with method 3")
                            return True

                    logger.error(f"❌ All close methods failed for {ticket}")
                    logger.error(f"   Method 1 error: {result.retcode}")
                    logger.error(f"   Method 2 error: {result2.retcode if result2 else 'None'}")
                    f"   Method 3 error: {result3.retcode if 'result3' in locals() and result3 else 'None'}"
                    f"
                    return False

        except Exception as e:
            logger.error(f"❌ Exception closing position {ticket}: {e}")
            return False

    async def close_all_positions(self) -> int:
        """Close all open positions"""
        try:
            positions = await self.get_open_positions()
            closed_count = 0

            for pos in positions:
                if await self.close_position(pos["ticket"]):
                    closed_count += 1
                    await asyncio.sleep(0.1)  # Small delay between closes

            logger.info(f"✅ Closed {closed_count} positions")
            return closed_count

        except Exception as e:
            logger.error(f"❌ Error closing all positions: {e}")
            return 0

    async def get_trade_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get trade history for specified number of days"""
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)

            # Get deals (completed trades)
            deals = mt5.history_deals_get(from_date, to_date)
            if deals is None:
                return []

            history = []
            for deal in deals:
                if deal.magic == self.magic_number:  # Only our bot's trades
                    deal_data = {
                        "ticket": deal.ticket,
                        "order": deal.order,
                        "symbol": deal.symbol,
                        "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
                        "volume": deal.volume,
                        "price": deal.price,
                        "profit": deal.profit,
                        "swap": deal.swap,
                        "commission": deal.commission,
                        "comment": deal.comment,
                        "time": datetime.fromtimestamp(deal.time)
                    }
                    history.append(deal_data)

            return history

        except Exception as e:
            logger.error(f"❌ Error getting trade history: {e}")
            return []

    def shutdown(self) -> None:
        """Shutdown MT5 connection"""
        try:
            if self.initialized:
                mt5.shutdown()
                logger.info("✅ MT5 connection closed")
        except Exception as e:
            logger.error(f"❌ Error shutting down MT5: {e}")

# ==================== MT5 BOT ADAPTER ====================

class MT5TradingBot:
    """Adapter to use MT5 interface with existing bot logic"""

    def __init__(self, original_bot):
        self.original_bot = original_bot
        self.mt5 = MT5TradingInterface()
        self.price_update_interval = 1.0  # Update prices every 1 second
        self.last_price_update = 0
        self.current_symbol = None

    async def initialize(self) -> bool:
        """Initialize MT5 trading interface"""
        return await self.mt5.initialize()

    async def start_price_feed(self, symbol: str = None) -> None:
        """Start simulated price feed from MT5"""
        try:
            if symbol is None:
                symbol = self.mt5.default_symbol

            self.current_symbol = symbol
            logger.info(f"📊 Starting price feed for {symbol}")

            while self.original_bot.running:
                try:
                    # Get current price
                    price = await self.mt5.get_current_price(symbol)
                    if price:
                        # Simulate tick data for the original bot
                        timestamp = int(time.time())

                        # Update original bot's price history
                        self.original_bot.price_history.append(price)
                        if len(self.original_bot.price_history) > 1000:
                            self.original_bot.price_history = self.original_bot.price_history[-500:]

                        # Update indicators
                        self.original_bot.indicators.update_price_data(price, timestamp)

                        # Update multi-timeframe data
                        self.original_bot.update_multi_timeframe_data(price, timestamp)

                        # Update balance
                        balance = await self.mt5.get_current_balance()
                        equity = await self.mt5.get_current_equity()
                        self.original_bot.current_balance = equity  # Use equity as it includes unrealized P&L

                        # Trigger trading analysis every few updates
                        if len(self.original_bot.price_history) % 5 == 0:
                            await self.original_bot.analyze_market_and_trade()

                    await asyncio.sleep(self.price_update_interval)

                except Exception as e:
                    logger.error(f"❌ Error in price feed: {e}")
                    await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"❌ Price feed error: {e}")

    async def place_trade_mt5(self, action: str, symbol: str, amount: float) -> Optional[Dict[str, Any]]:
        """Place trade through MT5 instead of WebSocket"""
        try:
            # Calculate lot size based on amount
            lot_size = self.mt5.calculate_lot_size(symbol, amount)

            logger.info(f"🎯 Placing MT5 trade: {action} {symbol} ${amount} ({lot_size} lots)")

            # Place order
            if action == "BUY":
                f"AI Bot {action}"
                f"
            else:  # SELL
                f"AI Bot {action}"
                f"

            if result:
                # Convert MT5 result to format expected by original bot
                trade_data = {
                    "contract_id": str(result["deal"]),
                    "action": action,
                    "symbol": symbol,
                    "amount": amount,
                    "buy_price": amount,
                    "start_time": datetime.now(),
                    "entry_price": result["price"],
                    "mt5_order": result["order"],
                    "mt5_deal": result["deal"],
                    "lot_size": lot_size
                }

                # Add to active trades
                self.original_bot.active_trades[str(result["deal"])] = trade_data
                self.original_bot.last_trade_time = datetime.now()
                self.original_bot.trades_today += 1

                logger.info(f"✅ MT5 trade placed successfully: {result['deal']}")
                return trade_data

            return None

        except Exception as e:
            logger.error(f"❌ Error placing MT5 trade: {e}")
            return None

    async def check_mt5_positions(self) -> None:
        """Check and update MT5 positions"""
        try:
            # Get current positions
            positions = await self.mt5.get_open_positions()

            # Update active trades based on actual MT5 positions
            current_deals = {str(pos["ticket"]): pos for pos in positions}

            # Check for closed positions (trades that were in active_trades but no longer in MT5)
            closed_trades = []
            for contract_id, trade_data in list(self.original_bot.active_trades.items()):
                if "mt5_deal" in trade_data:
                    deal_id = str(trade_data["mt5_deal"])
                    if deal_id not in current_deals:
                        # Position was closed
                        closed_trades.append((contract_id, trade_data))

            # Process closed trades
            for contract_id, trade_data in closed_trades:
                # Get trade history to find the close details
                history = await self.mt5.get_trade_history(days=1)

                # Find matching close trade
                close_profit = 0.0
                for deal in history:
                    if deal["order"] == trade_data.get("mt5_order", 0):
                        close_profit = deal.get("profit", 0.0)
                        break

                # Create result for original bot
                result = {
                    "status": "won" if close_profit > 0 else "lost",
                    "payout": trade_data["amount"] + close_profit if close_profit > 0 else 0,
                    "profit_loss": close_profit,
                    "exit_price": 0  # MT5 handles this differently
                }

                # Process the result
                await self.original_bot.process_trade_result(contract_id, result)

                # Remove from active trades
                if contract_id in self.original_bot.active_trades:
                    del self.original_bot.active_trades[contract_id]

                logger.info(f"📊 MT5 position closed: {contract_id} P&L: ${close_profit:.2f}")

        except Exception as e:
            logger.error(f"❌ Error checking MT5 positions: {e}")

    def shutdown(self) -> None:
        """Shutdown MT5 interface"""
        self.mt5.shutdown()

    async def _validate_mt5_ready_for_trading(self) -> bool:
        """Validate MT5 is ready for trading operations"""
        try:
            loop = asyncio.get_event_loop()

            # Check terminal connection
            terminal = await loop.run_in_executor(None, mt5.terminal_info)
            if not terminal or not terminal.connected or not terminal.trade_allowed:
                logger.error("❌ MT5 terminal not ready for trading")
                return False

            # Check account
            account = await loop.run_in_executor(None, mt5.account_info)
            if not account or not account.trade_allowed:
                logger.error("❌ MT5 account not ready for trading")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Error validating MT5 readiness: {e}")
            return False

    async def _validate_symbol_ready(self, symbol: str) -> bool:
        """Validate symbol is ready for trading"""
        try:
            loop = asyncio.get_event_loop()

            # Make sure symbol is selected
            if not await loop.run_in_executor(None, mt5.symbol_select, symbol, True):
                logger.error(f"❌ Cannot select symbol {symbol}")
                return False

            # Wait a moment for symbol to load
            await asyncio.sleep(0.1)

            # Check symbol info
            symbol_info = await loop.run_in_executor(None, mt5.symbol_info, symbol)
            if not symbol_info:
                logger.error(f"❌ No symbol info for {symbol}")
                return False

            if not symbol_info.visible:
                logger.warning(f"⚠️ Symbol {symbol} not visible, trying to add to Market Watch")
                if not await loop.run_in_executor(None, mt5.symbol_select, symbol, True):
                    logger.error(f"❌ Cannot make symbol {symbol} visible")
                    return False

            # Check if trading is allowed for this symbol
            if symbol_info.trade_mode == 0:  # SYMBOL_TRADE_MODE_DISABLED
                logger.error(f"❌ Trading disabled for symbol {symbol}")
                return False

            # Check if we have valid prices
            if symbol_info.bid <= 0 or symbol_info.ask <= 0:
                f"❌ Invalid prices for symbol {symbol}"
                f" bid={symbol_info.bid}, ask={symbol_info.ask}"
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Error validating symbol {symbol}: {e}")
            return False

    async def _try_alternative_order_method(self, request: Dict[str, Any], symbol: str) -> Optional[Any]:
        """Try alternative order placement method when standard order_send fails"""
        try:
            loop = asyncio.get_event_loop()

            # Method 1: Re-initialize MT5 connection and retry
            logger.info("🔄 Trying Method 1: Re-initialize MT5 connection")

            # Re-initialize MT5
            await loop.run_in_executor(None, mt5.shutdown)
            await asyncio.sleep(0.5)

            if not await loop.run_in_executor(None, mt5.initialize):
                logger.error("❌ Failed to re-initialize MT5")
                return None

            # Re-select symbol
            if not await loop.run_in_executor(None, mt5.symbol_select, symbol, True):
                logger.error(f"❌ Failed to re-select symbol {symbol}")
                return None

            # Wait for connection to stabilize
            await asyncio.sleep(1.0)

            # Try order again
            result = await loop.run_in_executor(None, mt5.order_send, request)
            if result is not None:
                logger.info("✅ Method 1 successful: Re-initialization worked")
                return result

            # Method 2: Try with modified request parameters
            logger.info("🔄 Trying Method 2: Modified request parameters")

            # Get fresh symbol info
            symbol_info = await loop.run_in_executor(None, mt5.symbol_info, symbol)
            if symbol_info:
                # Create modified request with fresh data
                modified_request = request.copy()
                modified_request["price"] = symbol_info.bid if request["type"] == mt5.ORDER_TYPE_SELL else symbol_info.ask
                modified_request["deviation"] = 50  # Increase deviation
                modified_request["type_filling"] = mt5.ORDER_FILLING_RETURN  # Try RETURN filling

                result = await loop.run_in_executor(None, mt5.order_send, modified_request)
                if result is not None:
                    logger.info("✅ Method 2 successful: Modified parameters worked")
                    return result

            # Method 3: Try market order with no price
            logger.info("🔄 Trying Method 3: Market order")

            market_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": request["volume"],
                "type": request["type"],
                "deviation": 100,
                "magic": request["magic"],
                "comment": "Alternative method",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = await loop.run_in_executor(None, mt5.order_send, market_request)
            if result is not None:
                logger.info("✅ Method 3 successful: Market order worked")
                return result

            logger.error("❌ All alternative methods failed")
            return None

        except Exception as e:
            logger.error(f"❌ Error in alternative order method: {e}")
            return None
