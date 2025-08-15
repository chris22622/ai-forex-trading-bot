🔧 MT5 MANUAL LOGIN GUIDE
==========================

❌ CURRENT ISSUE: MT5 Authorization Failed (-6)
✅ SOLUTION: Manual Login Required

📋 STEP-BY-STEP INSTRUCTIONS:

1. 🎯 OPEN MT5 TERMINAL
   • Look for MetaTrader 5 in your taskbar/desktop
   • If not running, start it from Start Menu
   • Should be already running based on our check

2. 🔐 ACCESS LOGIN MENU
   • In MT5 terminal, click "File" menu
   • Select "Login to Trade Account"
   • OR press Ctrl+L

3. 📝 ENTER YOUR CREDENTIALS
   • Login: 62233085
   • Password: ChrisAI2024!
   • Server: DerivVU-Server
   
   ⚠️ IMPORTANT: Make sure these are entered EXACTLY

4. ✅ SAVE SETTINGS
   • Check the box "Save account information"
   • This will remember your login for future sessions

5. 🔗 CONNECT
   • Click "OK" or "Login"
   • Wait for connection (may take 10-30 seconds)
   • Look for green connection indicator in bottom right

6. ✅ VERIFY SUCCESS
   • Check bottom right corner shows green connection
   • You should see your account balance
   • Market Watch should show available instruments

🚨 TROUBLESHOOTING:

If login fails:
• Double-check the password: ChrisAI2024!
• Verify server: DerivVU-Server
• Check internet connection
• Try restarting MT5 completely
• Contact Deriv support if account is locked

If server not found:
• Click "Scan" button to find servers
• Look for "DerivVU-Server" in the list
• Make sure you have a Deriv MT5 account

🎯 AFTER SUCCESSFUL LOGIN:

1. Keep MT5 terminal open
2. Run this test: python test_mt5_simple.py
3. If test passes, start your trading bot
4. The bot should now connect successfully

💡 WHY THIS HAPPENS:
MT5 needs to be manually logged in at least once to establish the connection. 
The Python library can only connect if MT5 terminal is already authenticated.

📞 NEED HELP?
If you continue having issues:
1. Screenshot the login error
2. Check your Deriv account status online
3. Verify your MT5 account is active
4. Consider using demo account for testing
