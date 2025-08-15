#!/usr/bin/env python3
"""
Force a high-confidence trade signal to test volume guard in action
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure environment to force a trade
os.environ['FORCE_TRADE_TEST'] = '1'

# Temporarily modify config to increase trading activity
print("🎯 Setting up forced trading test...")

# Read current config
with open('config.py', 'r') as f:
    config_content = f.read()

# Create backup and modify for testing
with open('config_backup_volume_test.py', 'w') as f:
    f.write(config_content)

# Modify config to force trades
modified_config = config_content.replace(
    'CONFIDENCE_THRESHOLD = 0.35',
    'CONFIDENCE_THRESHOLD = 0.01  # TESTING: Ultra-low threshold'
).replace(
    'ENHANCED_RISK_MANAGEMENT = True',
    'ENHANCED_RISK_MANAGEMENT = False  # TESTING: Simplified for demo'
)

with open('config.py', 'w') as f:
    f.write(modified_config)

print("✅ Config modified for volume guard testing")
print("📊 CONFIDENCE_THRESHOLD lowered to 0.01 (will trigger trades)")
print("🎯 Run: python main.py")
print("🔍 Watch for volume guard messages when trades trigger")
print("📝 To restore: copy config_backup_volume_test.py back to config.py")
