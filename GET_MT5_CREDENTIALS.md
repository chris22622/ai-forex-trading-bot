# 🔑 GET YOUR MT5 DEMO ACCOUNT CREDENTIALS

## Why Do You Need These?
Your bot is ready to trade but needs login credentials to connect to MetaTrader 5. Without these, the bot can't access market data or place trades.

## Quick Steps to Get MT5 Demo Credentials:

### Option 1: If You Already Have MT5 Installed
1. **Open MetaTrader 5** on your computer
2. **Check your current login**:
   - Go to "Tools" → "Options" → "Server" tab
   - Your login number and server will be displayed
3. **Use those credentials** in config.py

### Option 2: Create New Deriv Demo Account (Recommended)
1. **Go to** https://deriv.com/
2. **Click "Try Demo"** or "Demo Account"
3. **Sign up** for free demo account
4. **Download MT5** from Deriv
5. **Login with** the credentials they provide
6. **Copy credentials** to config.py

### Option 3: Other Brokers (MetaQuotes, ICMarkets, etc.)
1. Visit broker website
2. Click "Demo Account" 
3. Fill registration form
4. Download their MT5 platform
5. Use provided login credentials

## 📝 Update Your Config:

Open `config.py` and update these lines:

```python
# Replace these with YOUR actual credentials:
MT5_LOGIN = 5020568354  # ← Your demo account number
MT5_PASSWORD = "demopassword123"  # ← Your demo password  
MT5_SERVER = "Deriv-Demo"  # ← Your broker's server
```

## Common Server Names:
- **Deriv**: "Deriv-Demo" or "Deriv-Server"
- **MetaQuotes**: "MetaQuotes-Demo" 
- **ICMarkets**: "ICMarkets-Demo"
- **XM**: "XMGlobal-Demo"

## ⚡ After Setting Credentials:

1. **Save config.py**
2. **Run**: `LAUNCH_GODLIKE_BOT.bat`
3. **Watch Telegram** for trading updates!

## 🚨 Important Notes:

- ✅ **Use DEMO accounts only** - never risk real money while testing
- ✅ **Keep credentials private** - don't share your login details
- ✅ **Test first** - always verify the bot works before going live

## Need Help?
If you're unsure about any step, just ask! I can guide you through the specific broker setup process.
