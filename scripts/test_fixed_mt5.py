#!/usr/bin/env python3
"""
Quick test to verify MT5 trading is now working with correct settings
"""

import asyncio
import os
import sys

sys.path.append(os.getcwd())

from mt5_integration import MT5TradingInterface


async def test_fixed_mt5():
    print("🔧 Testing FIXED MT5 Integration")
    print("=" * 50)

    # Create MT5 interface
    mt5_interface = MT5TradingInterface()

    # Initialize
    success = await mt5_interface.initialize()
    if not success:
        print("❌ MT5 initialization failed")
        return

    print("✅ MT5 Initialized successfully")

    # Test price retrieval
    symbol = "Volatility 10 Index"
    price = await mt5_interface.get_current_price(symbol)

    if price and price > 0:
        print(f"✅ Price retrieval working: {symbol} = {price}")
    else:
        print(f"❌ Price retrieval failed for {symbol}")
        return

    # Test a small trade (this should now work with FOK filling)
    print("\n🎯 Testing trade execution...")

    trade_result = await mt5_interface.place_trade(
        action="BUY",
        symbol=symbol,
        amount=1.0  # Small amount
    )

    if trade_result:
        print("✅ TRADE SUCCESSFUL!")
        print(f"   Ticket: {trade_result.get('ticket', 'N/A')}")
        print(f"   Symbol: {trade_result.get('symbol', 'N/A')}")
        print(f"   Action: {trade_result.get('action', 'N/A')}")
        print(f"   Price: {trade_result.get('price', 'N/A')}")
        print(f"   Volume: {trade_result.get('lot_size', 'N/A')}")
    else:
        print("❌ Trade failed")

    # Cleanup
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_fixed_mt5())
