#!/usr/bin/env python3
"""
🚨 MANUAL MT5 TRADE CHECKER & EMERGENCY CLOSER
Check and manually close losing MT5 trades
"""

import MetaTrader5 as mt5
from datetime import datetime

def check_and_close_losing_trades():
    """Check all open MT5 positions and close heavy losers"""
    print("🚨 MANUAL MT5 TRADE CHECKER & EMERGENCY CLOSER")
    print("=" * 60)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    try:
        # Get account info
        account_info = mt5.account_info()
        if not account_info:
            print("❌ No account info available")
            return
        
        print(f"💰 Account Balance: ${account_info.balance:.2f}")
        print(f"💵 Account Equity: ${account_info.equity:.2f}")
        print(f"📉 Unrealized P&L: ${account_info.equity - account_info.balance:+.2f}")
        print("")
        
        # Get all open positions
        positions = mt5.positions_get()
        
        if not positions:
            print("✅ No open positions found")
            return
        
        print(f"🎯 Found {len(positions)} open position(s):")
        print("-" * 60)
        
        total_unrealized = 0.0
        heavy_losers = []
        
        for i, position in enumerate(positions):
            profit = position.profit
            total_unrealized += profit
            symbol = position.symbol
            volume = position.volume
            ticket = position.ticket
            type_str = "BUY" if position.type == 0 else "SELL"
            
            print(f"{i+1}. {symbol} ({type_str})")
            print(f"   🎫 Ticket: {ticket}")
            print(f"   📊 Volume: {volume}")
            print(f"   💰 P&L: ${profit:+.2f}")
            
            # Flag heavy losers
            if profit <= -3.00:
                heavy_losers.append((ticket, symbol, profit, volume))
                print(f"   🔴 HEAVY LOSER!")
            elif profit <= -1.50:
                print(f"   ⚠️ Moderate loss")
            elif profit >= 0.50:
                print(f"   ✅ Profitable")
            
            print("")
        
        print(f"📊 SUMMARY:")
        print(f"   Total Unrealized P&L: ${total_unrealized:+.2f}")
        print(f"   Heavy Losers (>$3): {len(heavy_losers)}")
        print("")
        
        # Emergency closure of heavy losers
        if heavy_losers:
            print("🚨 EMERGENCY CLOSURE RECOMMENDATION:")
            print(f"   Found {len(heavy_losers)} trades losing more than $3.00")
            print("")
            
            response = input("🤔 Close all heavy losers (>$3 loss)? (y/n): ").lower().strip()
            
            if response in ['y', 'yes']:
                print("🔴 CLOSING HEAVY LOSERS...")
                closed_count = 0
                total_loss_prevented = 0.0
                
                for ticket, symbol, profit, volume in heavy_losers:
                    print(f"🔴 Closing {symbol} ({ticket}): ${profit:.2f}")
                    
                    # Close the position
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": ticket,
                        "symbol": symbol,
                        "volume": volume,
                        "type": mt5.ORDER_TYPE_BUY if profit < 0 else mt5.ORDER_TYPE_SELL,
                        "magic": 0,
                        "comment": "Emergency closure - heavy loss",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"   ✅ Successfully closed {symbol}")
                        closed_count += 1
                        total_loss_prevented += abs(profit)
                    else:
                        print(f"   ❌ Failed to close {symbol}")
                        if result:
                            print(f"      Error: {result.comment}")
                
                print("")
                print(f"📊 EMERGENCY CLOSURE COMPLETE:")
                print(f"   Trades Closed: {closed_count}")
                print(f"   Loss Prevented: ${total_loss_prevented:.2f}")
                
            else:
                print("❌ Emergency closure cancelled")
        
        else:
            print("✅ No heavy losers found (all losses < $3.00)")
        
        print("")
        print("💡 RECOMMENDATIONS:")
        if total_unrealized < -25:
            print("   🚨 SEVERE LOSSES - Consider closing all losing trades")
            print("   🚨 Take a break from trading")
            print("   🚨 Review strategy completely")
        elif total_unrealized < -10:
            print("   ⚠️ MODERATE LOSSES - Close losing trades over $2")
            print("   ⚠️ Reduce position sizes significantly")
        elif total_unrealized < 0:
            print("   📊 MINOR LOSSES - Monitor closely")
            print("   📊 Consider closing trades losing more than $1.50")
        else:
            print("   ✅ PROFITABLE SESSION - Consider taking profits")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    check_and_close_losing_trades()
