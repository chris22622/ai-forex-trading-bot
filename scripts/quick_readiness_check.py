import asyncio
import sys
sys.path.append('.')
from main import DerivTradingBot

async def test_readiness():
    print("🔍 CHECKING BOT READINESS...")
    bot = DerivTradingBot()
    readiness = await bot.check_readiness()
    print('\n📊 READINESS CHECK RESULTS:')
    print("=" * 40)
    for category, status in readiness.items():
        status_icon = "✅" if "ready" in status.lower() or "ok" in status.lower() else "❌"
        print(f'{status_icon} {category}: {status}')
    
    # Check MT5 specifically
    print(f"\n🔗 MT5 CONNECTION TEST:")
    if hasattr(bot, 'mt5_interface') and bot.mt5_interface:
        try:
            connection_result = await bot.mt5_interface.initialize()
            if connection_result:
                balance = await bot.mt5_interface.get_account_balance()
                print(f"✅ MT5 Connected! Balance: ${balance:.2f}")
                print(f"✅ Interface Type: {type(bot.mt5_interface).__name__}")
                print(f"🚀 READY FOR REAL TRADING!")
            else:
                print(f"❌ MT5 Connection Failed")
        except Exception as e:
            print(f"❌ MT5 Error: {e}")
    else:
        print(f"❌ No MT5 Interface Available")

if __name__ == "__main__":
    asyncio.run(test_readiness())
