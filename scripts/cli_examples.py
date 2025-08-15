#!/usr/bin/env python3
"""
CLI Examples for the Enhanced Trading Bot
Demonstrates various command line options and Telegram controls

Run these commands from the terminal:
"""

print("""
🚀 ENHANCED TRADING BOT - CLI EXAMPLES

═══════════════════════════════════════════════════════════════

📋 BASIC COMMANDS:

# Start with default settings
python main.py

# Show help and all options
python main.py --help

# Check status without starting bot
python main.py --status

# Check account balance
python main.py --balance

# Test API connection
python main.py --test-connection

# Validate configuration
python main.py --validate-config

═══════════════════════════════════════════════════════════════

🎯 TRADING MODE EXAMPLES:

# Force demo mode (safe testing)
python main.py --demo

# Force live trading (use with caution!)
python main.py --live

# Enable paper trading (simulation)
python main.py --paper

# Dry run (analyze but don't trade)
python main.py --dry-run

═══════════════════════════════════════════════════════════════

💰 PARAMETER EXAMPLES:

# Custom trade amount
python main.py --amount 50.00

# Different symbol
python main.py --symbol R_75

# Specific execution mode
python main.py --execution MT5

# Lower AI confidence threshold
python main.py --confidence 0.6

# Disable AI (basic signals only)
python main.py --no-ai

═══════════════════════════════════════════════════════════════

📱 TELEGRAM EXAMPLES:

# Enable Telegram commands
python main.py --telegram

# Disable Telegram notifications
python main.py --no-telegram

# Custom Telegram chat ID
python main.py --chat-id 123456789

═══════════════════════════════════════════════════════════════

🔧 ADVANCED EXAMPLES:

# Conservative live trading
python main.py --live --amount 10 --confidence 0.8

# Aggressive demo testing
python main.py --demo --amount 100 --confidence 0.5

# Full monitoring setup
python main.py --telegram --verbose --execution AUTO

# MT5 only with custom settings
python main.py --execution MT5 --amount 25 --symbol R_100

═══════════════════════════════════════════════════════════════

📱 TELEGRAM COMMANDS (available when bot is running):

/start     - Start the trading bot
/stop      - Stop the trading bot
/status    - Get detailed bot status
/trades    - View active trades
/balance   - Check account balance
/pause     - Pause trading (keep monitoring)
/resume    - Resume trading
/stats     - Performance statistics
/help      - Show all commands
/emergency - Emergency stop all trading
/export    - Export session data
/settings  - View current settings
/ai        - AI model status
/risk      - Risk management info

═══════════════════════════════════════════════════════════════

🛡️ SAFETY REMINDERS:

1. Always test with --demo or --paper first
2. Start with small amounts (--amount 5)
3. Use higher confidence thresholds (--confidence 0.8)
4. Enable Telegram for remote monitoring
5. Use --dry-run to test strategies

═══════════════════════════════════════════════════════════════

🚀 QUICK START RECOMMENDATIONS:

# New users - safe testing:
python main.py --demo --paper --telegram --amount 5

# Experienced users - live trading:
python main.py --live --telegram --amount 25 --confidence 0.75

# Development/testing:
python main.py --dry-run --verbose --telegram

═══════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    print("Run the commands above in your terminal!")
