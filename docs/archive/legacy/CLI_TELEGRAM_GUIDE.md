# 🚀 Enhanced Trading Bot - CLI & Telegram Controls

Your elite trading bot now includes comprehensive command-line interface (CLI) and interactive Telegram controls for professional-grade operation.

## 🎯 New Features Added

### 📱 Telegram Interactive Commands
- `/start` - Start the trading bot
- `/stop` - Stop the trading bot  
- `/status` - Get detailed bot status
- `/trades` - View active trades
- `/balance` - Check account balance
- `/pause` - Pause trading (keep monitoring)
- `/resume` - Resume trading
- `/stats` - Performance statistics
- `/help` - Show all commands
- `/emergency` - Emergency stop all trading
- `/export` - Export session data
- `/settings` - View current settings
- `/ai` - AI model status
- `/risk` - Risk management info

### 🔧 Command Line Interface
- **Trading Modes**: `--demo`, `--live`, `--paper`, `--dry-run`
- **Parameters**: `--amount`, `--symbol`, `--execution`
- **AI Settings**: `--ai-model`, `--confidence`, `--no-ai`
- **Communication**: `--telegram`, `--no-telegram`, `--chat-id`
- **Utilities**: `--status`, `--balance`, `--test-connection`
- **Debug**: `--verbose`, `--debug`

## 🚀 Quick Start Examples

### Safe Testing (Recommended for New Users)
```bash
python main.py --demo --telegram --amount 5
```

### Live Trading (Experienced Users)
```bash
python main.py --live --telegram --amount 25 --confidence 0.75
```

### Analysis Only (No Trades)
```bash
python main.py --dry-run --verbose --telegram
```

### Check Status Without Starting
```bash
python main.py --status
python main.py --balance
python main.py --test-connection
```

## 📱 Telegram Setup

1. **Enable Telegram in config.py**:
   ```python
   ENABLE_TELEGRAM_ALERTS = True
   TELEGRAM_BOT_TOKEN = "your_bot_token"
   TELEGRAM_CHAT_ID = "your_chat_id"
   ```

2. **Start bot with Telegram**:
   ```bash
   python main.py --telegram
   ```

3. **Send commands to your bot**:
   - Open Telegram
   - Send `/help` to see all commands
   - Use `/status` to monitor performance
   - Use `/pause` and `/resume` for control

## 🛡️ Safety Features

### Pause/Resume Control
- Use `/pause` to temporarily stop trading
- Bot continues monitoring but won't place trades
- Use `/resume` to restart trading

### Emergency Stop
- Use `/emergency` for immediate halt
- Closes active positions (where possible)
- Complete trading stop

### Dry Run Mode
- Use `--dry-run` for analysis only
- Shows what trades would be placed
- Perfect for strategy testing

### Risk Management
- Use `/risk` to check current risk levels
- Automatic cooldowns after losses
- Position size controls

## 🔧 Advanced Usage

### Custom Trading Settings
```bash
# Conservative approach
python main.py --live --amount 10 --confidence 0.8

# Aggressive testing
python main.py --demo --amount 100 --confidence 0.5

# MT5 only
python main.py --execution MT5 --amount 25
```

### Multiple AI Models
```bash
# Neural network focus
python main.py --ai-model neural

# Random forest
python main.py --ai-model random_forest

# Disable AI
python main.py --no-ai
```

### Development & Testing
```bash
# Verbose logging
python main.py --verbose --debug

# Paper trading
python main.py --paper --telegram

# Configuration check
python main.py --validate-config
```

## 📊 Monitoring & Analytics

### Real-time Status
- Use `/status` for comprehensive overview
- Shows balance, trades, performance
- Market conditions and AI analysis

### Performance Tracking
- Use `/stats` for detailed metrics
- Win/loss ratios and streaks
- AI confidence analysis

### Data Export
- Use `/export` to save session data
- CSV files for analysis
- Historical performance data

## 🎮 Interactive Launcher

For users who prefer menus over command line:

```bash
python launch_bot.py
```

This provides an interactive menu with:
1. Demo Mode
2. Live Mode  
3. Paper Trading
4. Status Checks
5. Custom Settings

## 📋 CLI Examples Reference

See `cli_examples.py` for comprehensive examples:

```bash
python cli_examples.py
```

## 🔒 Security Notes

1. **Always test with `--demo` first**
2. **Start with small amounts** (`--amount 5`)
3. **Use higher confidence** (`--confidence 0.8`)
4. **Enable Telegram for monitoring**
5. **Keep your bot token secure**

## 📞 Support

- Use `/help` in Telegram for quick reference
- Check logs/ directory for detailed information  
- Use `--validate-config` to check setup
- Use `--test-connection` for network issues

---

**Your trading bot is now elite-level with professional controls! 🏆**
