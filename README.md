# AI Forex Trading Bot

Smart, risk-controlled FX trading with **backtesting**, **paper trading**, and **live execution** (when enabled). Clean `src/` layout, CI-ready, and built to be audited quickly by hiring managers.

[![CI](https://github.com/chris22622/ai-forex-trading-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/chris22622/ai-forex-trading-bot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.8–3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> _"Backtest honestly. Paper trade safely. Go live only when the data says so."_

---

## ✨ Features
- **3 modes**: Backtest → Paper → Live (flag-driven).
- **Risk first**: per-trade % risk, max daily loss, kill-switch, cooldown after losses.
- **Config-driven**: YAML for pairs, timeframes, indicators, and risk.
- **Pluggable data/execution**: MT5 / broker APIs (when keys provided).
- **Clean logs + CSV exports**: trades, equity curve, metrics.
- **CI friendly**: lint + tests; deterministic seeds for backtests.

---

## 🚀 Quickstart (30 sec)

### 1) Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
cp .env.example .env  # add your credentials if you plan to paper/live trade
```

### 2) Smoke test

```bash
pytest -q
```

### 3) Backtest (recommended first)

```bash
python -m src.main --config config/config.yaml --backtest --from 2023-01-01 --to 2024-12-31
```

### 4) Paper trade (no real orders)

```bash
python -m src.main --config config/config.yaml --paper --session 4h
```

### 5) Live trade (⚠️ real orders if broker creds exist)

```bash
python -m src.main --config config/config.yaml --live --session 4h
```

---

## 🔧 Configuration (YAML)

`config/config.yaml` drives everything. Example:

```yaml
seed: 42
data:
  provider: mt5              # mt5 | oanda | csv
  timezone: "UTC"
  symbols: ["EURUSD", "GBPUSD", "XAUUSD"]
  timeframe: "M15"           # M1/M5/M15/M30/H1/H4/D1
strategy:
  name: "ai_ensemble"
  params:
    rsi_period: 14
    rsi_buy: 30
    rsi_sell: 70
    ema_fast: 21
    ema_slow: 55
risk:
  balance_start: 10000
  risk_per_trade_pct: 0.5
  max_daily_loss_pct: 2.0
  max_consecutive_losses: 3
  cooldown_minutes_after_stopout: 60
execution:
  slippage_pips: 1.0
  tp_rr: 1.5              # take-profit = 1.5x stop distance
  hard_stop_pips: 20
logging:
  folder: "out"
  csv_trades: "out/trades.csv"
  csv_equity: "out/equity.csv"
```

---

## 🔑 Environment (.env)

Only needed for paper/live or remote data:

```dotenv
# MT5 (example)
MT5_SERVER=Deriv-Demo
MT5_LOGIN=
MT5_PASSWORD=

# Optional alerts
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

---

## 🧪 Backtest Outputs

* `out/trades.csv` — every filled trade (timestamp, symbol, size, entry, SL/TP, PnL).
* `out/equity.csv` — equity curve for plotting/analysis.
* Console summary — Win rate, PF, Avg R, Max DD, Sharpe (if enabled).

Tip: keep backtests **deterministic** (fixed `seed`) so CI comparisons are stable.

---

## 🏃 Runtime Controls

* **Kill switch:** trip after `max_daily_loss_pct` or `max_consecutive_losses`.
* **Cooldown:** pause new entries for `cooldown_minutes_after_stopout`.
* **Flatten on exit:** all positions closed on SIGINT (Ctrl-C) or fatal error.

---

## 🧱 Project Structure

```
ai-forex-trading-bot/
├─ src/
│  ├─ ai_model.py         # AI ensemble (RF/GB/MLP)
│  ├─ indicators.py       # technical indicators
│  ├─ mt5_integration.py  # MT5 data/execution
│  ├─ safe_logger.py      # logging utilities
│  └─ main.py             # main trading entry point
├─ config/
│  ├─ config.example.yaml
│  └─ config.yaml         # your working config
├─ tests/                 # pytest suite
├─ scripts/               # utility scripts
├─ docs/                  # documentation
├─ requirements.txt
├─ .env.example
└─ README.md
```

---

## 🐳 Docker (optional)

```bash
docker build -t fxbot .
docker run --env-file .env -v "$PWD/config:/app/config" -v "$PWD/out:/app/out" fxbot \
  python -m src.main --config config/config.yaml --backtest --from 2023-01-01 --to 2024-12-31
```

---

## 📈 CI & Quality

* **CI** runs lint + tests on push/PR (badge above).
* Add longer backtests behind a flag to keep CI fast.

---

## ⚠️ Risk Disclaimer

This code is for **research/education**. Markets carry risk. **No performance guarantees.** Use paper trading first and never exceed capital you can afford to lose.

---

## 📚 Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - System design and data flow
- **[Security](SECURITY.md)** - Security guidelines and best practices
- **[Contributing](CONTRIBUTING.md)** - How to contribute to the project
- **[Legacy Guides](docs/legacy/)** - Development history and troubleshooting

---

## 📝 License

MIT © [Chris](https://github.com/chris22622)
