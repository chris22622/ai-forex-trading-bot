# 🚀 MT5 REAL TRADING SETUP GUIDE

## 📋 **CURRENT STATUS:**

Your bot is currently in **SIMULATION MODE** - it's getting prices from MT5 but not placing real orders.

## 🔧 **HOW TO ENABLE REAL MT5 TRADING:**

### **Step 1: Setup MetaTrader 5**

1. **Download MT5:** Get from your broker or MetaQuotes
2. **Login to your account:**
   - For **DEMO**: Use demo account credentials
   - For **REAL**: Use live account credentials
3. **Keep MT5 running** when you start the bot

### **Step 2: Choose Demo vs Real Account**

Edit `config.py` and set these options:

```python
# ==================== MT5 CONFIGURATION ====================

# Demo vs Real Account
MT5_DEMO_MODE = True   # True = Demo account, False = Real account
MT5_REAL_TRADING = False  # True = Real orders, False = Simulation

# Optional: Specify account (leave None to use current MT5 login)
MT5_LOGIN = None  # e.g., 123456789
MT5_PASSWORD = ""  # Your MT5 password
MT5_SERVER = ""   # e.g., "MetaQuotes-Demo" or "ICMarkets-Live"
```

### **Step 3: Configuration Options**

#### **🔹 For SAFE TESTING (Current Setup):**
```python
MT5_DEMO_MODE = True      # Use demo account
MT5_REAL_TRADING = False  # Simulate trades (no real orders)
```

#### **🔹 For DEMO ACCOUNT TRADING:**
```python
MT5_DEMO_MODE = True     # Use demo account  
MT5_REAL_TRADING = True  # Place real orders on demo
```

#### **🔹 For LIVE ACCOUNT TRADING:**
```python
MT5_DEMO_MODE = False    # Use real account
MT5_REAL_TRADING = True  # Place real orders on live account
```

## ⚠️ **IMPORTANT SAFETY NOTES:**

### **Current Safety Features:**
- ✅ Paper trading enabled (`PAPER_TRADING = True`)
- ✅ Small position sizes (`MT5_LOT_SIZE = 0.01`)
- ✅ Position limits (`MT5_MAX_LOT_SIZE = 1.0`)
- ✅ Risk management active
- ✅ Auto-cooldown after losses

### **Before Going Live:**
1. **Test thoroughly** on demo account first
2. **Start with small amounts** (`MT5_LOT_SIZE = 0.01`)
3. **Monitor closely** for first few trades
4. **Have stop-loss strategy** ready

## 🎯 **TRADING SYMBOLS:**

The bot will trade these MT5 symbols (configure in `config.py`):
- `Volatility 75 Index` (current default)
- `Volatility 100 Index`
- `EURUSD`, `GBPUSD`, `USDJPY` (forex)
- `XAUUSD` (gold)

## 🔄 **HOW TO SWITCH MODES:**

### **To Enable Demo Trading:**
1. Edit `config.py`:
   ```python
   MT5_DEMO_MODE = True
   MT5_REAL_TRADING = True  # Change to True
   ```
2. Restart the bot
3. Make sure MT5 is logged into demo account

### **To Enable Live Trading:**
1. **FIRST**: Test thoroughly on demo!
2. Edit `config.py`:
   ```python
   MT5_DEMO_MODE = False    # Change to False
   MT5_REAL_TRADING = True  # Change to True
   ```
3. Make sure MT5 is logged into live account
4. Restart the bot

## 📊 **POSITION SIZING:**

Current settings in `config.py`:
```python
MT5_LOT_SIZE = 0.01      # Micro lot (very small)
MT5_MAX_LOT_SIZE = 1.0   # Maximum 1 standard lot
```

### **Lot Size Guide:**
- `0.01` = Micro lot (safest for testing)
- `0.1` = Mini lot (medium risk)
- `1.0` = Standard lot (high risk)

## 🚦 **STEP-BY-STEP ACTIVATION:**

### **Phase 1: Demo Testing (Recommended)**
1. Keep current settings: `MT5_REAL_TRADING = False`
2. Run bot and observe behavior
3. Check if it connects to MT5 properly

### **Phase 2: Demo Trading**
1. Set: `MT5_REAL_TRADING = True`
2. Set: `MT5_DEMO_MODE = True`
3. Test with demo money

### **Phase 3: Live Trading (When Ready)**
1. Set: `MT5_REAL_TRADING = True` 
2. Set: `MT5_DEMO_MODE = False`
3. Start with small lot sizes
4. Monitor closely!

## 🔍 **CURRENT BOT STATUS:**

Your bot is currently:
- ✅ Connected to MT5
- ✅ Getting real prices
- ✅ Analyzing markets with AI
- ✅ **Simulating trades** (safe mode)

To check current mode:
```bash
python check_godlike_status.py
```

## ❓ **FAQ:**

**Q: Will it trade on my real account now?**
A: No, currently it's simulating. Set `MT5_REAL_TRADING = True` to enable real orders.

**Q: How to choose demo vs real account?**
A: Use `MT5_DEMO_MODE = True/False` in config.py

**Q: Is it safe to test?**
A: Yes! Current setup is simulation-only. Enable real trading only when ready.

**Q: What if I want to stop real trading?**
A: Set `MT5_REAL_TRADING = False` and restart the bot.

## 🎯 **QUICK START:**

For **demo trading** (real orders on demo account):
1. Login to MT5 demo account
2. Edit `config.py`: Set `MT5_REAL_TRADING = True`
3. Restart bot: `LAUNCH_GODLIKE_BOT.bat`
4. Monitor the trades!

---
**⚠️ Remember: Always test on demo first before going live!**
