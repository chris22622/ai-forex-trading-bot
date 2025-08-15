"""
🚀 Enhanced Telegram Notification System with Persistent Trade Tracking
Beautiful, detailed notifications with visual flair and comprehensive trade monitoring
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import config
try:
    from config import ENABLE_TELEGRAM_ALERTS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    telegram_bot_token = TELEGRAM_BOT_TOKEN
    telegram_chat_id = TELEGRAM_CHAT_ID
    enable_telegram_alerts = ENABLE_TELEGRAM_ALERTS
except ImportError:
    telegram_bot_token = ""
    telegram_chat_id = ""
    enable_telegram_alerts = False

# Try to import telegram modules
try:
    import requests
    requests_available = True
except ImportError:
    requests_available = False
    logging.warning("requests not available")

@dataclass
class TradeRecord:
    """Complete trade record with all details"""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    entry_time: str
    stake: float

    # Optional fields for closed trades
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    profit_loss: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED
    duration_seconds: Optional[int] = None

    # Analysis fields
    confidence: Optional[float] = None
    strategy: Optional[str] = None
    reason: Optional[str] = None
    win_rate_at_time: Optional[float] = None

    # MT5 specific
    mt5_ticket: Optional[str] = None

    def __post_init__(self):
        if not self.entry_time:
            self.entry_time = datetime.now().isoformat()

    @property
    def is_profit(self) -> bool:
        """Check if trade is profitable"""
        return self.profit_loss is not None and self.profit_loss > 0

    @property
    def duration_formatted(self) -> str:
        """Get formatted duration string"""
        if self.duration_seconds is None:
            return "N/A"

        hours = self.duration_seconds // 3600
        minutes = (self.duration_seconds % 3600) // 60
        seconds = self.duration_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

@dataclass
class TradingStats:
    """Track comprehensive trading statistics"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    total_volume: float = 0.0
    average_trade_duration: float = 0.0
    daily_profit: float = 0.0
    weekly_profit: float = 0.0
    monthly_profit: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    last_reset_date: str = ""

    def __post_init__(self):
        if not self.last_reset_date:
            self.last_reset_date = datetime.now().date().isoformat()

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.total_trades == 0:
            return 0.0
        return (self.wins / self.total_trades) * 100

class EnhancedTelegramNotifier:
    """Enhanced Telegram notifier with beautiful formatting and persistent tracking"""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None, trades_file: str = "trades_history.json"):
        self.bot_token = bot_token or telegram_bot_token
        self.chat_id = chat_id or telegram_chat_id
        self.trades_file = trades_file
        self.trades: Dict[str, TradeRecord] = {}
        self.stats = TradingStats()
        self.enabled = enable_telegram_alerts and bool(self.bot_token) and bool(self.chat_id)

        # External stats source (can be set by main bot)
        self.external_session_stats = None

        # Load existing trades and stats
        self.load_trades()

        # Rate limiting - More conservative for reliability
        self.last_message_time = 0.0
        self.rate_limit_delay = 2.0  # 2 seconds between messages to avoid 429 errors
        self.rate_limit_violations = 0  # Track violations for backoff

        logging.info(f"🚀 Enhanced Telegram Notifier initialized - Enabled: {self.enabled}")

    def set_external_session_stats(self, session_stats: dict) -> None:
        """Set external session stats from main bot for synchronized tracking"""
        self.external_session_stats = session_stats
        logging.debug(f"📊 External session stats updated: {session_stats}")

    def get_current_stats(self) -> dict:
        """Get current stats - prefer external stats if available"""
        if self.external_session_stats:
            # Use external stats from main bot
            total_trades = self.external_session_stats.get('total_trades', 0)
            wins = self.external_session_stats.get('wins', 0)
            losses = self.external_session_stats.get('losses', 0)
            total_profit = self.external_session_stats.get('total_profit', 0.0)

            # Calculate win rate
            win_rate = (wins / max(1, total_trades)) * 100

            return {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_profit': total_profit
            }
        else:
            # Fallback to internal stats
            return {
                'total_trades': self.stats.total_trades,
                'wins': self.stats.wins,
                'losses': self.stats.losses,
                'win_rate': self.stats.win_rate,
                'total_profit': self.stats.total_profit
            }

    def load_trades(self):
        """Load trades from persistent storage"""
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)

                    # Load trades
                    for trade_id, trade_data in data.get('trades', {}).items():
                        self.trades[trade_id] = TradeRecord(**trade_data)

                    # Load stats
                    stats_data = data.get('stats', {})
                    if stats_data:
                        for key, value in stats_data.items():
                            if hasattr(self.stats, key):
                                setattr(self.stats, key, value)

                logging.info(f"📂 Loaded {len(self.trades)} trades from storage")
            else:
                logging.info("📂 No existing trades file found, starting fresh")
        except Exception as e:
            logging.error(f"❌ Error loading trades: {e}")
            self.trades = {}

    def save_trades(self):
        """Save trades to persistent storage"""
        try:
            data: Dict[str, any] = {  # type: ignore
                'trades': {tid: asdict(trade) for tid, trade in self.trades.items()},
                'stats': asdict(self.stats),
                'last_updated': datetime.now().isoformat()
            }

            # Create backup before saving
            if os.path.exists(self.trades_file):
                backup_file = f"{self.trades_file}.backup"
                try:
                    os.rename(self.trades_file, backup_file)
                except:
                    pass

            with open(self.trades_file, 'w') as f:
                json.dump(data, f, indent=2)

            logging.debug(f"💾 Saved {len(self.trades)} trades to storage")
        except Exception as e:
            logging.error(f"❌ Error saving trades: {e}")

    def send_message_sync(self, message: str) -> bool:
        """Send message synchronously using requests"""
        if not self.enabled or not requests_available:
            logging.info(f"� Telegram disabled - Would send: {message[:100]}...")
            return False

        try:
            # Use retry logic with exponential backoff
            return self._send_with_retry(message)

        except Exception as e:
            logging.error(f"❌ Error sending Telegram message: {e}")
            return False

    def _send_with_retry(self, message: str) -> bool:
        """Send message with retry logic and chunking - Enhanced for 429 handling"""
        import time  # Import at function level for retry logic

        max_retries = 5  # Increased retries for rate limit issues
        retry_delays = [2, 5, 10, 20, 30]  # Longer delays for 429 errors

        for attempt in range(max_retries):
            try:
                # Enhanced rate limiting with backoff for violations
                current_time = time.time()
                required_delay = self.rate_limit_delay

                # Increase delay if we've had recent rate limit violations
                if self.rate_limit_violations > 0:
                    required_delay *= (2 ** min(self.rate_limit_violations, 3))  # Exponential backoff

                time_since_last = current_time - self.last_message_time
                if time_since_last < required_delay:
                    sleep_time = required_delay - time_since_last
                    logging.info(f"🕐 Rate limiting: sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)

                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

                # Split long messages to prevent timeouts
                max_length = 2500  # More conservative length for rate limiting
                if len(message) > max_length:
                    # Send in chunks
                    chunks = [message[i:i+max_length] for i in range(0, len(message), max_length)]
                    success = True
                    for i, chunk in enumerate(chunks):
                        chunk_msg = f"📄 Part {i+1}/{len(chunks)}:\n{chunk}" if len(chunks) > 1 else chunk
                        if not self._send_single_chunk(url, chunk_msg):
                            success = False
                            break
                        if i < len(chunks) - 1:  # Longer delay between chunks
                            time.sleep(1.5)

                    if success:
                        self.rate_limit_violations = max(0, self.rate_limit_violations - 1)  # Reduce violations on success
                    return success
                else:
                    result = self._send_single_chunk(url, message)
                    if result:
                        self.rate_limit_violations = max(0, self.rate_limit_violations - 1)  # Reduce violations on success
                    return result

            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "too many requests" in error_msg:
                    self.rate_limit_violations += 1
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    logging.warning(f"🚫 Rate limit hit (429). Backoff #{self.rate_limit_violations}. Waiting {delay}s...")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        continue
                else:
                    logging.warning(f"❌ Telegram attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:  # Don't sleep on last attempt
                        time.sleep(retry_delays[attempt])
                        logging.info(f"🔄 Retrying Telegram in {retry_delays[attempt]}s...")

        logging.error(f"❌ Telegram failed after {max_retries} attempts")
        return False

    def _send_single_chunk(self, url: str, message: str) -> bool:
        """Send a single message chunk with enhanced 429 handling"""
        import time  # Import needed for last_message_time

        import requests  # Import requests for exception handling

        data: Dict[str, any] = {  # type: ignore
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }

        try:
            response = requests.post(url, data=data, timeout=30)  # type: ignore

            if response.status_code == 200:
                self.last_message_time = time.time()
                return True
            elif response.status_code == 429:
                # Parse retry-after header if available
                retry_after = response.headers.get('retry-after', '60')
                self.rate_limit_violations += 1
                logging.warning(f"🚫 Telegram rate limit (429). Retry after: {retry_after}s. Violations: {self.rate_limit_violations}")
                raise Exception(f"429 Too Many Requests - retry after {retry_after}s")
            else:
                logging.error(f"❌ Telegram API error: {response.status_code} - {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Telegram request error: {e}")
            raise e

    async def send_message(self, message: str) -> bool:
        """Send message asynchronously"""
        return self.send_message_sync(message)

    def notify_trade_opened(self, trade_record: TradeRecord) -> bool:
        """🚀 Send beautiful trade opening notification"""
        try:
            # Store trade record
            self.trades[trade_record.trade_id] = trade_record
            self.save_trades()

            # Create beautiful notification
            direction_emoji = "🟢 📈" if trade_record.direction.upper() == "BUY" else "🔴 📉"
            direction_text = "LONG" if trade_record.direction.upper() == "BUY" else "SHORT"

            # Confidence visualization (handle both 0-1 and 0-100 formats)
            raw_confidence = trade_record.confidence or 0
            if raw_confidence > 1.0:
                # Already in percentage format (0-100)
                confidence_pct = raw_confidence
                confidence_norm = raw_confidence / 100  # Normalize to 0-1 for bars
            else:
                # In decimal format (0-1)
                confidence_pct = raw_confidence * 100
                confidence_norm = raw_confidence

            conf_bars = "█" * min(10, max(0, int(confidence_norm * 10)))
            conf_empty = "░" * (10 - len(conf_bars))
            confidence_bar = conf_bars + conf_empty

            # Win rate visualization - Use current synced stats instead of trade record
            current_stats = self.get_current_stats()
            current_win_rate = current_stats['win_rate']

            # Handle win rate format (use current calculated win rate)
            if current_win_rate > 1.0:
                # Already in percentage format (0-100)
                wr_pct = current_win_rate
                wr_norm = current_win_rate / 100  # Normalize to 0-1 for bars
            else:
                # In decimal format (0-1) - but current_win_rate should already be percentage
                wr_pct = current_win_rate
                wr_norm = current_win_rate / 100

            wr_bars = "█" * min(10, max(0, int(wr_norm * 10)))
            wr_empty = "░" * (10 - len(wr_bars))
            winrate_bar = wr_bars + wr_empty

            message = f"""🚨 <b>TRADE OPENED</b> 🚨
{direction_emoji} ━━━━━━━━━━━━━━━ {direction_emoji}

📊 <b>POSITION DETAILS</b>
🎯 <b>Direction:</b> {direction_text} {trade_record.direction}
💎 <b>Symbol:</b> {trade_record.symbol}
💰 <b>Stake:</b> ${trade_record.stake:.2f}
📈 <b>Entry Price:</b> {trade_record.entry_price:.5f}
🎫 <b>Trade ID:</b> #{trade_record.trade_id}

🤖 <b>AI ANALYSIS</b>
🧠 <b>Strategy:</b> {trade_record.strategy or 'Algorithmic'}
💡 <b>Reason:</b> {trade_record.reason or 'Technical Signal'}
📊 <b>Confidence:</b> {confidence_pct:.1f}%
{confidence_bar}
🏆 <b>Bot Win Rate:</b> {wr_pct:.1f}%
{winrate_bar}

📈 <b>SESSION INFO</b>
🔢 <b>Total Trades:</b> {current_stats['total_trades']}
✅ <b>Wins:</b> {current_stats['wins']} | ❌ <b>Losses:</b> {current_stats['losses']}
💰 <b>Session P&L:</b> {'+' if current_stats['total_profit'] >= 0 else ''}${current_stats['total_profit']:.2f}

⏰ <b>Opened:</b> {datetime.fromisoformat(trade_record.entry_time).strftime('%H:%M:%S')}
📅 <b>Date:</b> {datetime.fromisoformat(trade_record.entry_time).strftime('%m/%d/%Y')}

🎯 <i>May the pips be with you!</i> 🎯"""

            return self.send_message_sync(message)

        except Exception as e:
            logging.error(f"❌ Error sending trade opening notification: {e}")
            return False

    def notify_trade_closed(self, trade_id: str, exit_price: float, profit_loss: float,
                           close_reason: str = "Manual") -> bool:
        """🎯 Send beautiful trade closing notification with P&L details"""
        try:
            if trade_id not in self.trades:
                logging.warning(f"⚠️ Trade {trade_id} not found in records")
                return False

            trade = self.trades[trade_id]

            # Update trade record
            trade.exit_price = exit_price
            trade.exit_time = datetime.now().isoformat()
            trade.profit_loss = profit_loss
            trade.status = "CLOSED"

            # Calculate duration
            if trade.entry_time:
                entry_dt = datetime.fromisoformat(trade.entry_time)
                exit_dt = datetime.fromisoformat(trade.exit_time)
                trade.duration_seconds = int((exit_dt - entry_dt).total_seconds())

            # Update statistics
            self.update_stats(trade)
            self.save_trades()

            # Create beautiful result notification
            if profit_loss > 0:
                result_emoji = "🎉 💰"
                result_text = "WINNER!"
                pl_emoji = "💵"
                pl_text = f"+${profit_loss:.2f}"
                border_emoji = "🟢"
                celebration = "🎊 🏆 🎊"
                profit_color = "🟢"
            elif profit_loss < 0:
                result_emoji = "💔 📉"
                result_text = "LOSS"
                pl_emoji = "📉"
                pl_text = f"-${abs(profit_loss):.2f}"
                border_emoji = "🔴"
                celebration = "😔 💪 🔄"
                profit_color = "🔴"
            else:
                result_emoji = "⚖️ 💫"
                result_text = "BREAK EVEN"
                pl_emoji = "💼"
                pl_text = f"${profit_loss:.2f}"
                border_emoji = "🟡"
                celebration = "📊 ✅ 📊"
                profit_color = "🟡"

            # Calculate price movement
            if trade.direction.upper() == "BUY":
                price_change = exit_price - trade.entry_price
                pips_direction = "📈" if price_change > 0 else "📉"
            else:
                price_change = trade.entry_price - exit_price
                pips_direction = "📈" if price_change > 0 else "📉"

            price_change_pct = (abs(price_change) / trade.entry_price) * 100

            # Updated statistics - Use current stats (external if available)
            current_stats = self.get_current_stats()
            current_winrate = current_stats['win_rate']

            message = f"""{celebration}
<b>{result_text}</b> 
{border_emoji} ━━━━━━━━━━━━━━━ {border_emoji}

{result_emoji} <b>TRADE RESULT</b>
💎 <b>Symbol:</b> {trade.symbol}
📊 <b>Direction:</b> {trade.direction.upper()}
{pl_emoji} <b>P&L:</b> {profit_color} <b>{pl_text}</b>
📈 <b>Entry:</b> {trade.entry_price:.5f}
📉 <b>Exit:</b> {exit_price:.5f}
{pips_direction} <b>Movement:</b> {price_change_pct:.3f}%

⏱️ <b>Duration:</b> {trade.duration_formatted}
💰 <b>Stake:</b> ${trade.stake:.2f}
📋 <b>Reason:</b> {close_reason}
🎫 <b>Trade ID:</b> #{trade_id}

📊 <b>SESSION STATS</b>
🎯 <b>Total Trades:</b> {current_stats['total_trades']}
🏆 <b>Wins:</b> {current_stats['wins']} | 💔 <b>Losses:</b> {current_stats['losses']}
📈 <b>Win Rate:</b> {current_winrate:.1f}%
💰 <b>Session P&L:</b> {'+' if current_stats['total_profit'] >= 0 else ''}${current_stats['total_profit']:.2f}

⏰ <b>Closed:</b> {datetime.now().strftime('%H:%M:%S')}
📅 <b>Date:</b> {datetime.now().strftime('%m/%d/%Y')}

{celebration}"""

            return self.send_message_sync(message)

        except Exception as e:
            logging.error(f"❌ Error sending trade closing notification: {e}")
            return False

    def update_stats(self, trade: TradeRecord):
        """Update trading statistics"""
        try:
            self.stats.total_trades += 1
            self.stats.total_volume += trade.stake

            if trade.profit_loss is not None:
                self.stats.total_profit += trade.profit_loss
                self.stats.daily_profit += trade.profit_loss

                if trade.profit_loss > 0:
                    self.stats.wins += 1
                    self.stats.consecutive_wins += 1
                    self.stats.consecutive_losses = 0
                    self.stats.max_consecutive_wins = max(self.stats.max_consecutive_wins, self.stats.consecutive_wins)

                    if trade.profit_loss > self.stats.best_trade:
                        self.stats.best_trade = trade.profit_loss
                else:
                    self.stats.losses += 1
                    self.stats.consecutive_losses += 1
                    self.stats.consecutive_wins = 0
                    self.stats.max_consecutive_losses = max(self.stats.max_consecutive_losses, self.stats.consecutive_losses)

                    if trade.profit_loss < self.stats.worst_trade:
                        self.stats.worst_trade = trade.profit_loss

            # Update averages
            if trade.duration_seconds:
                total_duration = self.stats.average_trade_duration * (self.stats.total_trades - 1)
                self.stats.average_trade_duration = (total_duration + trade.duration_seconds) / self.stats.total_trades

        except Exception as e:
            logging.error(f"❌ Error updating stats: {e}")

    def send_daily_summary(self) -> bool:
        """📊 Send comprehensive daily trading summary"""
        try:
            # Calculate daily statistics
            today_trades = [t for t in self.trades.values()
                          if t.status == "CLOSED" and t.exit_time and
                          datetime.fromisoformat(t.exit_time).date() == datetime.now().date()]

            daily_wins = len([t for t in today_trades if t.profit_loss and t.profit_loss > 0])
            daily_losses = len([t for t in today_trades if t.profit_loss and t.profit_loss <= 0])
            daily_total = len(today_trades)
            daily_profit = sum(t.profit_loss for t in today_trades if t.profit_loss is not None)
            daily_winrate = (daily_wins / daily_total * 100) if daily_total > 0 else 0

            best_today = max([t.profit_loss for t in today_trades if t.profit_loss is not None], default=0)
            worst_today = min([t.profit_loss for t in today_trades if t.profit_loss is not None], default=0)

            # Overall performance emoji
            if daily_profit > 10:
                perf_emoji = "🚀 💎"
                perf_text = "EXCELLENT DAY!"
            elif daily_profit > 5:
                perf_emoji = "📈 💰"
                perf_text = "Great Performance"
            elif daily_profit > 0:
                perf_emoji = "📊 ✅"
                perf_text = "Profitable Day"
            elif daily_profit == 0:
                perf_emoji = "⚖️ 📊"
                perf_text = "Break Even Day"
            elif daily_profit > -5:
                perf_emoji = "📉 ⚠️"
                perf_text = "Small Loss Day"
            else:
                perf_emoji = "🔴 💔"
                perf_text = "Tough Day"

            # Win rate visualization
            wr_bars = "█" * min(10, max(0, int(daily_winrate / 10)))
            wr_empty = "░" * (10 - len(wr_bars))
            winrate_bar = wr_bars + wr_empty

            message = f"""📊 <b>DAILY TRADING SUMMARY</b> 📊
{perf_emoji} ━━━━━━━━━━━━━━━ {perf_emoji}

🎯 <b>{perf_text}</b>

💰 <b>TODAY'S PERFORMANCE</b>
💵 <b>Total P&L:</b> {'+' if daily_profit >= 0 else ''}${daily_profit:.2f}
🎯 <b>Trades:</b> {daily_total} ({daily_wins}W/{daily_losses}L)
📈 <b>Win Rate:</b> {daily_winrate:.1f}%
{winrate_bar}
🏆 <b>Best Trade:</b> +${best_today:.2f}
💔 <b>Worst Trade:</b> ${worst_today:.2f}
📊 <b>Avg Trade:</b> ${(daily_profit/daily_total):.2f if daily_total > 0 else 0:.2f}

📈 <b>OVERALL STATS</b>
📊 <b>Total Trades:</b> {self.stats.total_trades}
🏆 <b>Overall Win Rate:</b> {self.stats.win_rate:.1f}%
💰 <b>Total Profit:</b> {'+' if self.stats.total_profit >= 0 else ''}${self.stats.total_profit:.2f}
🎯 <b>Best Ever:</b> +${self.stats.best_trade:.2f}
💔 <b>Worst Ever:</b> ${self.stats.worst_trade:.2f}

🔥 <b>STREAKS</b>
🏆 <b>Current:</b> {self.stats.consecutive_wins}W / {self.stats.consecutive_losses}L
📊 <b>Max Win Streak:</b> {self.stats.max_consecutive_wins}
⚠️ <b>Max Loss Streak:</b> {self.stats.max_consecutive_losses}

📅 <b>Date:</b> {datetime.now().strftime('%A, %B %d, %Y')}
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

🎯 <i>Keep grinding! Every trade is a lesson!</i> 💪"""

            return self.send_message_sync(message)

        except Exception as e:
            logging.error(f"❌ Error sending daily summary: {e}")
            return False

    def send_bot_status(self, status: str, balance: float, active_trades: int = 0) -> bool:
        """🤖 Send bot status notification"""
        try:
            status_emoji = {
                "STARTING": "🚀 ⚡",
                "RUNNING": "✅ 🤖",
                "STOPPING": "⏹️ 🛑",
                "ERROR": "❌ ⚠️",
                "PAUSED": "⏸️ 🔄"
            }.get(status.upper(), "🤖 📊")

            # Performance context
            profit_today = self.stats.daily_profit
            if profit_today > 0:
                profit_emoji = "📈 💰"
                profit_text = f"+${profit_today:.2f}"
            elif profit_today < 0:
                profit_emoji = "📉 💔"
                profit_text = f"${profit_today:.2f}"
            else:
                profit_emoji = "⚖️ 📊"
                profit_text = "$0.00"

            message = f"""{status_emoji} <b>BOT STATUS UPDATE</b>
━━━━━━━━━━━━━━━━━━━

🤖 <b>Status:</b> {status.upper()}
💰 <b>Balance:</b> ${balance:.2f}
📊 <b>Active Trades:</b> {active_trades}
{profit_emoji} <b>Today's P&L:</b> {profit_text}

📈 <b>QUICK STATS</b>
🎯 <b>Total Trades:</b> {self.stats.total_trades}
🏆 <b>Win Rate:</b> {self.stats.win_rate:.1f}%
💵 <b>Total Profit:</b> ${self.stats.total_profit:.2f}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
📅 <b>Date:</b> {datetime.now().strftime('%m/%d/%Y')}

🎯 <i>Bot is {"working hard" if status.upper() == "RUNNING" else "standing by"}!</i>"""

            return self.send_message_sync(message)

        except Exception as e:
            logging.error(f"❌ Error sending bot status: {e}")
            return False

    def get_open_trades(self) -> List[TradeRecord]:
        """Get all currently open trades"""
        return [trade for trade in self.trades.values() if trade.status == "OPEN"]

    def get_closed_trades_today(self) -> List[TradeRecord]:
        """Get all trades closed today"""
        today = datetime.now().date()
        return [trade for trade in self.trades.values()
                if trade.status == "CLOSED" and trade.exit_time and
                datetime.fromisoformat(trade.exit_time).date() == today]

    def cleanup_old_trades(self, days_to_keep: int = 30):
        """Clean up old trade records to prevent file bloat"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            old_trades = [tid for tid, trade in self.trades.items()
                         if trade.status == "CLOSED" and trade.exit_time and
                         datetime.fromisoformat(trade.exit_time) < cutoff_date]

            for trade_id in old_trades:
                del self.trades[trade_id]

            if old_trades:
                self.save_trades()
                logging.info(f"🧹 Cleaned up {len(old_trades)} old trade records")

        except Exception as e:
            logging.error(f"❌ Error cleaning up trades: {e}")
