"""
Test the current get_symbol_info method to see what it returns
"""

import asyncio
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mt5_integration import MT5TradingInterface


async def test_symbol_info():
    """Test symbol info retrieval"""
    print("🔍 Testing MT5 symbol info retrieval...")

    mt5_interface = MT5TradingInterface()

    # Initialize
    if not await mt5_interface.initialize():
        print("❌ Failed to initialize MT5")
        return

    # Test symbols
    test_symbols = ["Volatility 75 Index", "Volatility 100 Index"]

    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")

        # Get symbol info using our method
        symbol_info = await mt5_interface.get_symbol_info(symbol)

        if symbol_info:
            print("✅ Symbol info retrieved successfully:")
            for key, value in symbol_info.items():
                print(f"   {key}: {value} (type: {type(value).__name__})")

            # Test validation
            print("\n🔍 Testing validation:")
            validated_info = mt5_interface._validate_symbol_info(symbol_info.copy(), symbol)
            print("Validated info:")
            for key, value in validated_info.items():
                print(f"   {key}: {value} (type: {type(value).__name__})")
        else:
            print("❌ Failed to get symbol info")

if __name__ == "__main__":
    asyncio.run(test_symbol_info())
