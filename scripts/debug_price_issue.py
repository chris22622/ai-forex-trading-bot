#!/usr/bin/env python3
"""Debug the price retrieval issue"""

import asyncio
import sys

sys.path.append('.')
import MetaTrader5 as mt5

from mt5_integration import MT5TradingInterface


async def test_price_retrieval():
    print("🔍 Testing price retrieval issue...")

    # Test 1: Direct MT5
    print("\n1. Direct MT5 test:")
    if mt5.initialize():
        tick = mt5.symbol_info_tick('Volatility 10 Index')
        info = mt5.symbol_info('Volatility 10 Index')
        print(f"   Tick available: {tick is not None}")
        print(f"   Symbol info available: {info is not None}")
        if tick:
            print(f"   Direct MT5 price: {tick.bid}")
        if info:
            print(f"   Direct MT5 bid/ask: {info.bid}/{info.ask}")

    # Test 2: Through MT5TradingInterface
    print("\n2. Through MT5TradingInterface:")
    interface = MT5TradingInterface()
    init_result = await interface.initialize()
    print(f"   Interface initialized: {init_result}")

    price = await interface.get_current_price('Volatility 10 Index')
    print(f"   Interface price: {price}")

    # Test 3: Symbol mapping
    print("\n3. Symbol mapping test:")
    try:
        from mt5_integration import symbol_manager
        mapped = symbol_manager.get_safe_symbol('Volatility 10 Index')
        print(f"   Symbol mapping: Volatility 10 Index → {mapped}")
    except Exception as e:
        print(f"   Symbol mapping error: {e}")

if __name__ == "__main__":
    asyncio.run(test_price_retrieval())
