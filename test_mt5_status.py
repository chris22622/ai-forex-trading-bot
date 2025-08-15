#!/usr/bin/env python3
"""Test current MT5 connection status"""

from main import DerivTradingBot
import asyncio

async def check_mt5_status():
    print('🔧 Checking current MT5 connection status...')
    bot = DerivTradingBot()
    
    print(f'MT5 Interface Type: {type(bot.mt5_interface).__name__}')
    print(f'Using MT5: {bot.using_mt5}')
    print(f'Connected: {bot.connected}')
    
    # Check if we're using the dummy interface
    if type(bot.mt5_interface).__name__ == 'DummyMT5Interface':
        print('❌ PROBLEM FOUND: Bot is using DummyMT5Interface!')
        print('   This means trades are only simulated, not sent to real MT5')
        print('   Need to fix MT5 connection')
        return False
    
    if hasattr(bot, 'mt5_interface') and bot.mt5_interface:
        print('\n🔧 Testing MT5 interface...')
        try:
            result = await bot.mt5_interface.initialize()
            print(f'MT5 Initialize Result: {result}')
            if result:
                balance = await bot.mt5_interface.get_account_balance()
                print(f'Account Balance: ${balance:.2f}')
                print('✅ MT5 connection is working!')
                return True
            else:
                print('❌ MT5 initialization failed')
                return False
        except Exception as e:
            print(f'❌ MT5 Test Error: {e}')
            return False
    else:
        print('❌ No MT5 interface found')
        return False

if __name__ == "__main__":
    result = asyncio.run(check_mt5_status())
    print(f'\n🎯 MT5 Status: {"WORKING" if result else "NEEDS FIXING"}')
