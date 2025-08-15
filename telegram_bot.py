"""
Telegram Bot Module for Deriv Trading Bot
Handles all Telegram notifications and user interactions
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from config import *

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    print("⚠️ Telegram library not available. Install with: pip install python-telegram-bot")
    TELEGRAM_AVAILABLE = False
    # Define dummy classes to prevent import errors
    class Bot:
        def __init__(self, *args, **kwargs):
            pass
    class TelegramError(Exception):
        pass

class TelegramNotifier:
    """Handles Telegram notifications for the trading bot"""

    def __init__(self):
        self.bot = None
        self.chat_id = TELEGRAM_CHAT_ID
        self.enabled = ENABLE_TELEGRAM_ALERTS and TELEGRAM_AVAILABLE
        self.message_queue = []
        self.last_message_time: Dict[str, Any] = {}
        self.rate_limit_delay = 1  # Minimum seconds between messages

        # Check if we have a valid token before trying to initialize
        if self.enabled and TELEGRAM_BOT_TOKEN and len(TELEGRAM_BOT_TOKEN) > 10:
            try:
                self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
                print("✅ Telegram bot initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Telegram bot: {e}")
                self.enabled = False
        else:
            # Disable if no valid token
            if not TELEGRAM_BOT_TOKEN or len(TELEGRAM_BOT_TOKEN) < 10:
                print("⚠️ Telegram token not configured - disabling Telegram notifications")
            self.enabled = False

    async def send_message(self, message: str, parse_mode: Optional[str] = "HTML", disable_notification: bool = False) -> bool:
        """Send a message to Telegram with proper HTML formatting and conflict resolution"""
        if not self.enabled or not self.bot:
            print(f"📱 Telegram disabled - Would send: {message}")
            return False

        try:
            # Rate limiting
            current_time = datetime.now()
            if 'last_send' in self.last_message_time:
                time_diff = (current_time - self.last_message_time['last_send']).total_seconds()
                if time_diff < self.rate_limit_delay:
                    await asyncio.sleep(self.rate_limit_delay - time_diff)

            # Send message with HTML parsing enabled by default
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_notification=disable_notification,
                disable_web_page_preview=True  # Cleaner messages
            )

            self.last_message_time['last_send'] = datetime.now()
            return True

        except TelegramError as e:
            # Handle specific Telegram errors more gracefully
            error_msg = str(e).lower()
            if "conflict" in error_msg:
                print(f"⚠️ Telegram conflict detected: {e}")
                print("💡 Another bot instance may be running. Retrying in 3 seconds...")
                await asyncio.sleep(3)
                # Try once more
                try:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode=None,  # Disable parsing on retry
                        disable_notification=True
                    )
                    return True
                except:
                    print("❌ Telegram retry failed. Continuing without this message.")
                    return False
            elif "chat not found" in error_msg:
                print("❌ Telegram error: Chat not found. Please start the bot by messaging it first.")
                # Disable further attempts to avoid spam
                self.enabled = False
            elif "bot was blocked" in error_msg:
                print("❌ Telegram error: Bot was blocked by user.")
                self.enabled = False
            elif "too many requests" in error_msg:
                print("⚠️ Telegram rate limit exceeded. Waiting 5 seconds...")
                await asyncio.sleep(5)
                return False
            else:
                print(f"❌ Telegram error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error sending Telegram message: {e}")
            return False

    def send_message_sync(self, message: str) -> bool:
        """Synchronous wrapper for sending messages"""
        if not self.enabled:
            return False

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, schedule the coroutine
                asyncio.create_task(self.send_message(message))
                return True
            else:
                # Run in new event loop
                return loop.run_until_complete(self.send_message(message))
        except Exception as e:
            print(f"Error in sync message send: {e}")
            return False

    async def send_trade_alert(self, trade_data: Dict[str, Any]) -> bool:
        """Send beautiful trade execution alert"""
        try:
            action = trade_data.get('action', 'UNKNOWN')
            symbol = trade_data.get('symbol', 'Unknown')
            amount = trade_data.get('amount', 0)
            price = trade_data.get('price', 0)
            confidence = trade_data.get('confidence', 0)
            reason = trade_data.get('reason', 'AI Analysis')
            strategy = trade_data.get('strategy', 'Momentum')
            win_rate = trade_data.get('win_rate', 0)
            execution = trade_data.get('execution', 'API')

            # Handle confidence format (normalize to 0-100 range)
            if confidence > 1.0:
                confidence_display = min(100, confidence)  # Already in percentage
            else:
                confidence_display = confidence * 100  # Convert from decimal

            # Handle win rate format (normalize to 0-100 range)
            if win_rate > 1.0:
                win_rate_display = min(100, win_rate)  # Already in percentage
            else:
                win_rate_display = win_rate * 100  # Convert from decimal

            # Create action-based styling
            if action == "BUY":
                action_emoji = "🟢"
                action_text = "BUY SIGNAL"
                border = "🟢" * 15
            elif action == "SELL":
                action_emoji = "🔴"
                action_text = "SELL SIGNAL"
                border = "🔴" * 15
            else:
                action_emoji = "⚪"
                action_text = f"{action} SIGNAL"
                border = "⚪" * 15

            # Confidence bar visualization (use normalized values)
            conf_norm = confidence_display / 100  # Normalize to 0-1
            conf_bars = int(conf_norm * 10)
            confidence_bar = "█" * conf_bars + "░" * (10 - conf_bars)

            # Win rate visualization (use normalized values)
            wr_norm = win_rate_display / 100  # Normalize to 0-1
            wr_bars = int(wr_norm * 10) if wr_norm > 0 else 5
            winrate_bar = "█" * wr_bars + "░" * (10 - wr_bars)

            message = f"""🚨 <b>{action_text}</b> 🚨
{border}

📊 <b>TRADE DETAILS</b>
💎 Symbol: <b>{symbol}</b>
💰 Amount: <b>${amount:.2f}</b>
📈 Price: <b>{price:.4f}</b>
🚀 Execution: <b>{execution}</b>

🎯 <b>AI ANALYSIS</b>
🤖 Strategy: <b>{strategy}</b>
🧠 Reason: <b>{reason}</b>
📊 Confidence: <b>{confidence_display:.1f}%</b>
{confidence_bar}
🏆 Win Rate: <b>{win_rate_display:.1f}%</b>
{winrate_bar}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
📅 <b>Date:</b> {datetime.now().strftime('%m/%d/%Y')}"""

            return await self.send_message(message, parse_mode="HTML")

        except Exception as e:
            print(f"Error sending trade alert: {e}")
            # Fallback to simple message with safe defaults
            try:
                action = trade_data.get('action', 'TRADE')
                symbol = trade_data.get('symbol', 'Unknown')
                amount = trade_data.get('amount', 0)
                simple_msg = f"🔔 {action} {symbol} ${amount:.2f} at {datetime.now().strftime('%H:%M:%S')}"
                return await self.send_message(simple_msg, parse_mode=None)
            except:
                return False

    async def send_trade_result(self, result_data: Dict[str, Any]) -> bool:
        """Send beautiful trade result notification"""
        try:
            result = result_data.get('result', 'UNKNOWN')
            profit_loss = result_data.get('profit_loss', 0)
            symbol = result_data.get('symbol', 'Unknown')
            duration = result_data.get('duration', 0)

            # Create beautiful result styling
            if result == "WIN":
                result_emoji = "🎉"
                status_text = "TRADE WON!"
                pl_emoji = "💰"
                pl_text = f"+${profit_loss:.2f}"
                border = "🟢" * 15
                celebration = "🎊🏆🎊"
            elif result == "LOSS":
                result_emoji = "💔"
                status_text = "TRADE LOST"
                pl_emoji = "📉"
                pl_text = f"-${abs(profit_loss):.2f}"
                border = "🔴" * 15
                celebration = "😔💪🔄"
            else:
                result_emoji = "⏹️"
                status_text = "TRADE CLOSED"
                pl_emoji = "💼"
                pl_text = f"${profit_loss:.2f}"
                border = "⚪" * 15
                celebration = "📊✅📊"

            # Duration formatting
            if duration >= 60:
                duration_text = f"{duration // 60}m {duration % 60}s"
            else:
                duration_text = f"{duration}s"

            message = f"""{celebration}
<b>{status_text}</b>
{border}

{result_emoji} <b>RESULT SUMMARY</b>
💎 Symbol: <b>{symbol}</b>
{pl_emoji} P&L: <b>{pl_text}</b>
⏱️ Duration: <b>{duration_text}</b>
📅 Closed: <b>{datetime.now().strftime('%H:%M:%S')}</b>

{celebration}"""

            return await self.send_message(message, parse_mode="HTML")

        except Exception as e:
            print(f"Error sending trade result: {e}")
            # Fallback to simple message
            try:
                result = result_data.get('result', 'CLOSED')
                profit_loss = result_data.get('profit_loss', 0)
                simple_msg = f"🔔 Trade {result}: ${profit_loss:.2f} at {datetime.now().strftime('%H:%M:%S')}"
                return await self.send_message(simple_msg, parse_mode=None)
            except:
                return False

    async def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send daily trading summary"""
        try:
            total_trades = summary_data.get('total_trades', 0)
            wins = summary_data.get('wins', 0)
            losses = summary_data.get('losses', 0)
            total_profit = summary_data.get('total_profit', 0)
            win_rate = summary_data.get('win_rate', 0)
            best_trade = summary_data.get('best_trade', 0)
            worst_trade = summary_data.get('worst_trade', 0)

            # Determine overall performance emoji
            if total_profit > 0:
                performance_emoji = "📈"
                performance_text = "Profitable Day"
            elif total_profit < 0:
                performance_emoji = "📉"
                performance_text = "Loss Day"
            else:
                performance_emoji = "➡️"
                performance_text = "Break Even"

            message = f"""
{performance_emoji} *DAILY SUMMARY*

📊 *Performance:* {performance_text}
💰 *Total P&L:* ${total_profit:+.2f}
🎯 *Trades:* {total_trades} ({wins}W/{losses}L)
📈 *Win Rate:* {win_rate:.1%}
🏆 *Best Trade:* +${best_trade:.2f}
💔 *Worst Trade:* -${abs(worst_trade):.2f}
📅 *Date:* {datetime.now().strftime('%Y-%m-%d')}
"""

            return await self.send_message(message, parse_mode='Markdown')

        except Exception as e:
            print(f"Error sending daily summary: {e}")
            return False

    async def send_error_alert(self, error_message: str, error_type: str = "ERROR") -> bool:
        """Send error notification"""
        try:
            emoji = "🚨" if error_type == "CRITICAL" else "⚠️"

            message = f"""
{emoji} *{error_type}*

🔥 *Issue:* {error_message}
⏰ *Time:* {datetime.now().strftime('%H:%M:%S')}

_Bot may need attention_
"""

            return await self.send_message(message, parse_mode='Markdown')

        except Exception as e:
            print(f"Error sending error alert: {e}")
            return False

    async def send_bot_status(self, status_data: Dict[str, Any]) -> bool:
        """Send beautiful bot status update"""
        try:
            status = status_data.get('status', 'UNKNOWN')
            uptime = status_data.get('uptime', 0)
            balance = status_data.get('balance', 0)
            trades_today = status_data.get('trades_today', 0)
            performance = status_data.get('performance', 0)

            # Status emoji
            status_emoji = {
                'RUNNING': '🟢',
                'STOPPED': '🔴',
                'PAUSED': '🟡',
                'ERROR': '🚨',
                'CONNECTED_MT5': '🔧'
            }.get(status, '⚪')

            # Create beautiful status message
            border = "🤖" * 15
            message = f"""🤖 BOT STATUS UPDATE
{border}

{status_emoji} Status: {status}
⏰ Uptime: {uptime}
💰 Balance: ${balance:.2f}
📊 Trades Today: {trades_today}
📈 Performance: {performance:+.2f}%

🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"""

            return await self.send_message(message, parse_mode=None)

        except Exception as e:
            print(f"Error sending bot status: {e}")
            # Fallback to simple message
            try:
                status = status_data.get('status', 'UNKNOWN')
                simple_msg = f"🔔 Bot Status: {status} at {datetime.now().strftime('%H:%M:%S')}"
                return await self.send_message(simple_msg, parse_mode=None)
            except:
                return False

    async def send_market_analysis(self, analysis_data: Dict[str, Any]) -> bool:
        """Send market analysis update"""
        try:
            symbol = analysis_data.get('symbol', 'Unknown')
            price = analysis_data.get('price', 0)
            trend = analysis_data.get('trend', 'NEUTRAL')
            rsi = analysis_data.get('rsi', 50)
            signal = analysis_data.get('signal', 'HOLD')
            confidence = analysis_data.get('confidence', 0)

            # Trend emoji
            trend_emoji = {
                'BULLISH': '📈',
                'BEARISH': '📉',
                'NEUTRAL': '➡️'
            }.get(trend.upper(), '❔')

            # Signal emoji
            signal_emoji = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '🟡'
            }.get(signal.upper(), '⚪')

            message = f"""
📊 *MARKET ANALYSIS*

💎 *Symbol:* {symbol}
💰 *Price:* {price}
{trend_emoji} *Trend:* {trend}
📈 *RSI:* {rsi:.1f}
{signal_emoji} *Signal:* {signal}
🎯 *Confidence:* {confidence:.1%}
⏰ *Time:* {datetime.now().strftime('%H:%M:%S')}
"""

            return await self.send_message(message, parse_mode='Markdown')

        except Exception as e:
            print(f"Error sending market analysis: {e}")
            return False

    async def send_startup_message(self) -> bool:
        """Send bot startup notification"""
        try:
            # Use simpler formatting to avoid parsing errors
            message = f"""🚀 DERIV TRADING BOT STARTED

🤖 Status: Online
💎 Symbol: {DEFAULT_SYMBOL}
💰 Trade Amount: ${TRADE_AMOUNT}
🎯 AI Model: Ensemble
📱 Notifications: Enabled
⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Ready to trade! 🎯"""

            return await self.send_message(message)

        except Exception as e:
            print(f"Error sending startup message: {e}")
            return False

    async def send_shutdown_message(self, reason: str = "Manual stop") -> bool:
        """Send bot shutdown notification"""
        message = f"""
🛑 *DERIV TRADING BOT STOPPED*

⏹️ *Reason:* {reason}
⏰ *Stopped:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_Bot is now offline_
"""

        return await self.send_message(message, parse_mode='Markdown')

    def test_connection(self) -> bool:
        """Test Telegram connection"""
        if not self.enabled:
            print("❌ Telegram not enabled")
            return False

        try:
            # Send test message
            test_message = f"🧪 Test message from Deriv Trading Bot\n⏰ {datetime.now().strftime('%H:%M:%S')}"
            return self.send_message_sync(test_message)
        except Exception as e:
            print(f"❌ Telegram test failed: {e}")
            return False

# Utility functions for message formatting
def format_currency(amount: float) -> str:
    """Format currency with proper sign and precision"""
    if amount >= 0:
        return f"+${amount:.2f}"
    else:
        return f"-${abs(amount):.2f}"

def format_percentage(value: float) -> str:
    """Format percentage with proper sign"""
    if value >= 0:
        return f"+{value:.1f}%"
    else:
        return f"{value:.1f}%"

def format_duration(seconds: int) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

def truncate_message(message: str, max_length: int = 4000) -> str:
    """Truncate message if too long for Telegram"""
    if len(message) <= max_length:
        return message

    return message[:max_length-3] + "..."

# Main notification manager
class NotificationManager:
    """Centralized notification management"""

    def __init__(self):
        self.telegram = TelegramNotifier()
        self.notification_history = []
        self.max_history = 1000

    def add_to_history(self, notification_type: str, data: Dict[str, Any]) -> None:
        """Add notification to history"""
        self.notification_history.append({
            'timestamp': datetime.now(),
            'type': notification_type,
            'data': data
        })

        # Keep history size manageable
        if len(self.notification_history) > self.max_history:
            self.notification_history = self.notification_history[-self.max_history//2:]

    async def notify_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Send trade notification"""
        self.add_to_history('trade', trade_data)
        return await self.telegram.send_trade_alert(trade_data)

    async def notify_result(self, result_data: Dict[str, Any]) -> bool:
        """Send trade result notification"""
        self.add_to_history('result', result_data)
        return await self.telegram.send_trade_result(result_data)

    async def notify_error(self, error_message: str, error_type: str = "ERROR") -> bool:
        """Send error notification"""
        self.add_to_history('error', {'message': error_message, 'type': error_type})
        return await self.telegram.send_error_alert(error_message, error_type)

    async def notify_status(self, status_data: Dict[str, Any]) -> bool:
        """Send status notification"""
        self.add_to_history('status', status_data)
        return await self.telegram.send_bot_status(status_data)

    async def notify_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send daily summary notification"""
        self.add_to_history('daily_summary', summary_data)
        return await self.telegram.send_daily_summary(summary_data)

    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        if not self.notification_history:
            return {'total': 0, 'by_type': {}}

        stats = {'total': len(self.notification_history), 'by_type': {}}

        for notification in self.notification_history:
            notif_type = notification['type']
            stats['by_type'][notif_type] = stats['by_type'].get(notif_type, 0) + 1

        return stats

if __name__ == "__main__":
    # Test the Telegram bot
    print("📱 Testing Telegram Bot...")

    async def test_telegram():
        notifier = TelegramNotifier()

        if notifier.enabled:
            # Test connection
            success = notifier.test_connection()
            if success:
                print("✅ Telegram test message sent successfully")

                # Test trade alert
                test_trade = {
                    'action': 'BUY',
                    'symbol': 'R_75',
                    'amount': 1.0,
                    'price': 1.5,
                    'confidence': 0.75,
                    'reason': 'RSI oversold + EMA uptrend'
                }

                await notifier.send_trade_alert(test_trade)
                print("✅ Test trade alert sent")
            else:
                print("❌ Telegram test failed")
        else:
            print("⚠️ Telegram not enabled")

    # Run test
    asyncio.run(test_telegram())
