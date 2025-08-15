#!/usr/bin/env python3
"""
Test script to verify the bot will send REAL trades to MT5 terminal
This will simulate a trading signal and attempt to place a real trade
"""
import asyncio
import sys
import os
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TradingBot

async def test_real_trading():
    """Test that the bot will place real trades in MT5"""
    print("🧪 TESTING REAL MT5 TRADE PLACEMENT")
    print("=" * 50)
    
    try:
        # Create bot instance
        print("📝 Creating bot instance...")
        bot = TradingBot()
        
        # Initialize MT5 connection
        print("🔗 Testing MT5 connection...")
        if not hasattr(bot, 'mt5_interface') or not bot.mt5_interface:
            print("❌ ERROR: No MT5 interface found!")
            return False
        
        # Initialize the interface
        connection_result = await bot.mt5_interface.initialize()
        if not connection_result:
            print("❌ ERROR: MT5 connection failed!")
            return False
        
        print("✅ MT5 Connected!")
        
        # Get account info
        balance = await bot.mt5_interface.get_account_balance()
        print(f"💰 Account Balance: ${balance:.2f}")
        
        # Test symbol availability  
        symbol = "Volatility 75 Index"
        symbol_info = await bot.mt5_interface.get_symbol_info(symbol)
        if not symbol_info:
            print(f"❌ ERROR: Symbol {symbol} not available!")
            return False
        
        print(f"📊 Symbol {symbol} available!")
        print(f"📈 Current price: {symbol_info.get('bid', 'N/A')}")
        
        # Calculate trade parameters
        print("\n🧮 CALCULATING TRADE PARAMETERS...")
        
        # Use small test amount
        test_amount = 5.0  # $5 test trade
        
        # Calculate lot size (same formula as in bot)
        price = float(symbol_info.get('bid', 0))
        if price <= 0:
            print("❌ ERROR: Invalid price!")
            return False
        
        # Risk parameters
        stop_loss_percentage = 2.0  # 2% stop loss
        stop_loss_distance = price * (stop_loss_percentage / 100)
        
        # Contract size for volatility indices (usually 1)
        contract_size = 1.0
        
        # Calculate lot size
        lot_size = test_amount / (stop_loss_distance * contract_size)
        
        # Round to 2 decimal places (MT5 lot size precision)
        lot_size = round(lot_size, 2)
        
        # Ensure minimum lot size
        if lot_size < 0.01:
            lot_size = 0.01
        
        print(f"💎 Trade Amount: ${test_amount}")
        print(f"📏 Lot Size: {lot_size}")
        print(f"🛑 Stop Loss Distance: {stop_loss_distance:.4f}")
        
        # Test trade parameters
        trade_params = {
            'symbol': symbol,
            'action': 'BUY',  # Test buy order
            'volume': lot_size,
            'price': price,
            'stop_loss': price - stop_loss_distance,
            'take_profit': price + (stop_loss_distance * 2),  # 2:1 risk/reward
            'comment': "TEST TRADE - Bot Verification",
            'type_time': 'GTC'
        }
        
        print(f"\n🎯 TEST TRADE PARAMETERS:")
        print(f"Symbol: {trade_params['symbol']}")
        print(f"Action: {trade_params['action']}")
        print(f"Volume: {trade_params['volume']}")
        print(f"Price: {trade_params['price']:.4f}")
        print(f"Stop Loss: {trade_params['stop_loss']:.4f}")
        print(f"Take Profit: {trade_params['take_profit']:.4f}")
        
        print(f"\n⚠️  WARNING: This will place a REAL ${test_amount} trade in your MT5 account!")
        print(f"⚠️  Make sure you're ready for this test!")
        print(f"⚠️  The trade will appear in your MT5 terminal!")
        
        # Ask for confirmation
        response = input("\n❓ Do you want to proceed with the REAL test trade? (type 'YES' to confirm): ")
        
        if response.upper() != 'YES':
            print("🚫 Test cancelled by user")
            return False
        
        print(f"\n🚀 PLACING REAL TEST TRADE...")
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Place the actual trade
        trade_result = await bot.mt5_interface.place_trade(**trade_params)
        
        if trade_result and trade_result.get('success'):
            print(f"\n✅ SUCCESS! REAL TRADE PLACED!")
            print(f"🎫 Order ID: {trade_result.get('order_id', 'N/A')}")
            print(f"🎯 Position: {trade_result.get('position_id', 'N/A')}")
            print(f"📊 Volume: {trade_result.get('volume', 'N/A')}")
            print(f"💰 Price: {trade_result.get('price', 'N/A')}")
            
            print(f"\n🔍 CHECK YOUR MT5 TERMINAL!")
            print(f"📱 The trade should appear in your MT5 terminal immediately")
            print(f"📈 Look in the 'Trade' tab and 'Terminal' window")
            
            return True
        else:
            print(f"\n❌ TRADE FAILED!")
            print(f"Error: {trade_result.get('error', 'Unknown error')}")
            print(f"Details: {trade_result}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 REAL MT5 TRADING TEST")
    print("This script will test if the bot can place REAL trades in MT5")
    print("=" * 60)
    
    # Run the test
    result = asyncio.run(test_real_trading())
    
    if result:
        print(f"\n🎉 TEST PASSED! Bot can place REAL trades in MT5!")
        print(f"✅ Your trading bot is ready for live trading")
    else:
        print(f"\n❌ TEST FAILED! Bot cannot place real trades")
        print(f"🔧 Check MT5 connection and permissions")
