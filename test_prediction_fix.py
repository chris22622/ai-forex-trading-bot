#!/usr/bin/env python3
"""
Test script for prediction function
"""

import asyncio
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from main import DerivTradingBot

async def test_prediction():
    """Test the AI prediction function"""
    try:
        print("🧪 Testing AI Prediction Function...")
        
        bot = DerivTradingBot()
        
        # Initialize with some test data
        bot.price_history = [
            1.1000, 1.1005, 1.1010, 1.1008, 1.1012, 
            1.1015, 1.1020, 1.1018, 1.1022, 1.1025, 
            1.1028, 1.1030, 1.1032, 1.1029, 1.1035
        ]
        
        # Test prediction
        prediction = await bot.get_ai_prediction()
        
        if prediction:
            print("✅ PREDICTION TEST PASSED!")
            print(f"   🎯 Action: {prediction.get('action')}")
            print(f"   📊 Confidence: {prediction.get('confidence'):.2%}")
            print(f"   🔄 Strategy: {prediction.get('strategy')}")
            print(f"   📈 Market Condition: {prediction.get('market_condition')}")
            print(f"   💡 Reason: {prediction.get('reason', 'N/A')}")
            return True
        else:
            print("❌ PREDICTION TEST FAILED: No prediction returned")
            return False
            
    except Exception as e:
        print(f"❌ PREDICTION TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_prediction())
    if result:
        print("\n🎉 All tests passed! Prediction error is FIXED.")
    else:
        print("\n💥 Tests failed. Check the errors above.")
    
    sys.exit(0 if result else 1)
