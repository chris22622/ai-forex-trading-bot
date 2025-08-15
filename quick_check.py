"""
Quick Bot Diagnostic - Check why no trades
"""
import sys
import os
import asyncio

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def quick_diagnostic():
    try:
        from main import global_bot_instance
        
        if not global_bot_instance:
            print("❌ Bot not running!")
            return
        
        bot = global_bot_instance
        print(f"🔍 QUICK STATUS:")
        print(f"   MT5: {bot.mt5_connected}")
        print(f"   Balance: ${bot.current_balance:.2f}")
        print(f"   Active Trades: {len(bot.active_trades)}")
        print(f"   Symbols: {len(bot.active_symbols)}")
        
        # Check if we have enough price data
        ready_symbols = 0
        for symbol, history in bot.symbol_price_histories.items():
            points = len(history)
            status = "✅ READY" if points >= 5 else f"⏳ {points}/5"
            print(f"   {symbol}: {status}")
            if points >= 5:
                ready_symbols += 1
        
        print(f"\n🎯 Ready to trade: {ready_symbols}/{len(bot.active_symbols)} symbols")
        
        if ready_symbols > 0:
            print("🚀 Bot should be analyzing trades soon!")
        else:
            print("⏳ Still collecting price data...")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(quick_diagnostic())
