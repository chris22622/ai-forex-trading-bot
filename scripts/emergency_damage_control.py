#!/usr/bin/env python3
"""
🚨 EMERGENCY DAMAGE CONTROL SCRIPT
Immediately stops current losses and provides damage assessment
"""

import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def emergency_damage_control():
    """🚨 Emergency stop for current losing trades"""
    try:
        print("🚨 EMERGENCY DAMAGE CONTROL ACTIVATED!")
        print("=" * 60)
        
        # Import bot from main
        try:
            from main import global_bot_instance, TradingBot
            if not global_bot_instance:
                print("❌ No active bot instance found")
                print("💡 Bot is not currently running")
                return
            
            bot = global_bot_instance
            
        except ImportError as e:
            print(f"❌ Failed to import bot: {e}")
            return
        
        # Stop the bot immediately
        print("🛑 STOPPING BOT IMMEDIATELY...")
        bot.running = False
        
        # Check current status
        current_balance = bot.current_balance
        starting_balance = bot.starting_balance
        current_loss = current_balance - starting_balance
        
        print(f"📊 ACCOUNT STATUS:")
        print(f"   Starting Balance: ${starting_balance:.2f}")
        print(f"   Current Balance: ${current_balance:.2f}")
        print(f"   Current P&L: ${current_loss:+.2f}")
        print("")
        
        # Check active trades
        active_trades = len(bot.active_trades)
        if active_trades == 0:
            print("✅ No active trades found")
            return
        
        print(f"🔍 CHECKING {active_trades} ACTIVE TRADES:")
        print("-" * 40)
        
        trades_closed = 0
        total_loss_prevented = 0.0
        profitable_trades = 0
        total_unrealized_loss = 0.0
        
        for trade_id, trade in list(bot.active_trades.items()):
            try:
                mt5_ticket = trade.get('mt5_ticket', trade_id)
                symbol = trade.get('symbol', 'Unknown')
                
                if bot.mt5_interface and mt5_ticket:
                    position_info = await bot.mt5_interface.get_position_info(mt5_ticket)
                    if position_info:
                        profit = float(getattr(position_info, 'profit', 0.0))
                        volume = float(getattr(position_info, 'volume', 0.0))
                        
                        print(f"📊 {symbol} ({mt5_ticket}): ${profit:+.2f} (Vol: {volume})")
                        
                        if profit < 0:
                            total_unrealized_loss += abs(profit)
                            
                            # Close trades losing more than $3
                            if profit <= -3.00:
                                print(f"🔴 CLOSING HEAVY LOSER: ${profit:.2f}")
                                close_result = await bot.mt5_interface.close_position(mt5_ticket)
                                if close_result:
                                    trades_closed += 1
                                    total_loss_prevented += abs(profit)
                                    del bot.active_trades[trade_id]
                                    print(f"✅ Trade {mt5_ticket} closed successfully")
                                else:
                                    print(f"❌ Failed to close trade {mt5_ticket}")
                            
                        else:
                            profitable_trades += 1
                            print(f"✅ Keeping profitable trade: ${profit:.2f}")
                    
                    else:
                        print(f"⚠️ Position {mt5_ticket} not found in MT5")
                        del bot.active_trades[trade_id]
                        
            except Exception as trade_error:
                print(f"❌ Error checking trade {trade_id}: {trade_error}")
        
        print("-" * 40)
        print(f"📊 DAMAGE CONTROL SUMMARY:")
        print(f"   Trades Closed: {trades_closed}")
        print(f"   Loss Prevented: ${total_loss_prevented:.2f}")
        print(f"   Profitable Trades Kept: {profitable_trades}")
        print(f"   Total Unrealized Loss: ${total_unrealized_loss:.2f}")
        print(f"   Remaining Active Trades: {len(bot.active_trades)}")
        print("")
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        if current_loss < -25:
            print("   🔴 SEVERE LOSS - Consider taking a break from trading")
            print("   🔴 Review strategy and reduce position sizes")
            print("   🔴 Consider closing remaining losing trades manually")
        elif current_loss < -10:
            print("   ⚠️ MODERATE LOSS - Reduce position sizes significantly")
            print("   ⚠️ Take profits quickly on any recovery")
        else:
            print("   ✅ Loss is manageable")
            print("   ✅ Apply new profitable settings and restart carefully")
        
        print("")
        print("🛡️ BOT HAS BEEN STOPPED")
        print("🔄 Apply profitable settings before restarting!")
        
    except Exception as e:
        print(f"❌ Emergency damage control error: {e}")

async def main():
    """Main entry point"""
    await emergency_damage_control()

if __name__ == "__main__":
    print("🚨 EMERGENCY DAMAGE CONTROL")
    print("   Stopping losses and assessing damage...")
    print("")
    
    asyncio.run(main())
