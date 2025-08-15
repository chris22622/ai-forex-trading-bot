import MetaTrader5 as mt5

mt5.initialize()
account = mt5.account_info()
print("Balance:", account.balance if account else "No account")
mt5.shutdown()
