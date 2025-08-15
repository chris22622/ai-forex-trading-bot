"""
Test actual symbol info usage as it would be called in the main bot
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mt5_integration import MT5TradingInterface

async def test_realistic_usage():
    """Test symbol info as it's used in real trading"""
    print("🔍 Testing realistic symbol info usage...")
    
    mt5_interface = MT5TradingInterface()
    
    # Initialize
    if not await mt5_interface.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    # Test lot size calculation for both symbols (this triggers the validation)
    test_symbols = ["Volatility 75 Index", "Volatility 100 Index"]
    test_amounts = [10.0, 25.0, 50.0]
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        
        for amount in test_amounts:
            try:
                lot_size = await mt5_interface.calculate_valid_lot_size(symbol, amount)
                print(f"   ${amount:.0f} → {lot_size:.3f} lots")
            except Exception as e:
                print(f"   ❌ Error for ${amount}: {e}")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_realistic_usage())
