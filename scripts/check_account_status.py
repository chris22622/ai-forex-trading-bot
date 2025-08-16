import MetaTrader5 as mt5

import config

# Initialize MT5
if not mt5.initialize():
    print('MT5 initialization failed')
    exit()

# Login
if not mt5.login(config.MT5_LOGIN, password=config.MT5_PASSWORD, server=config.MT5_SERVER):
    print('Login failed')
    exit()

# Get account info
account_info = mt5.account_info()
if account_info:
    print(f'💰 Current Balance: ${account_info.balance:.2f}')
    print(f'💎 Current Equity: ${account_info.equity:.2f}')
    print(f'📊 Profit/Loss: ${account_info.profit:.2f}')
    print(f'💹 Margin Used: ${account_info.margin:.2f}')

# Get open positions
positions = mt5.positions_get()
if positions:
    print(f'\n🎯 Open Positions: {len(positions)}')
    for pos in positions:
        pos_type = 'BUY' if pos.type == 0 else 'SELL'
                print(f'   Ticket: {pos.ticket} | {pos.symbol} | {pos_type} | Volume: '
        '{pos.volume} | P/L: ${pos.profit:.2f}')
else:
    print('\n📭 No open positions')

mt5.shutdown()
