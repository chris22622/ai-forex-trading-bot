#!/usr/bin/env python3
"""
Test script to manually close a trade and verify tracking works
"""

# Simple test to simulate a trade close and track results
print("🧪 Testing Trade Close and Tracking...")

# Simulate the current SELL trade that's active
entry_price = 98984.145  # From the logs
current_price = 98969.515  # Current price from logs
action = "SELL"
amount = 5.76
confidence = 0.80

# Calculate profit for SELL trade
if action == "SELL":
    # SELL profits when price goes down
    profit = (entry_price - current_price) * 0.0001 * amount  # Simple pip calculation
else:
    # BUY profits when price goes up
    profit = (current_price - entry_price) * 0.0001 * amount

print("📊 Trade Analysis:")
print(f"   Action: {action}")
print(f"   Entry Price: {entry_price}")
print(f"   Current Price: {current_price}")
print(f"   Amount: ${amount:.2f}")
print(f"   Confidence: {confidence:.1%}")
print(f"   Price Movement: {current_price - entry_price:.3f} points")
print(f"   Calculated Profit: ${profit:+.4f}")

if profit > 0:
    print("🟢 This would be a WINNING trade!")
    print("💰 Profit would update win rate and confidence tracking")
else:
    print("🔴 This would be a LOSING trade!")
    print("💸 Loss would update loss tracking")

print("\n✅ Manual trade analysis completed!")
print("The bot should automatically close trades and update stats when implemented.")
