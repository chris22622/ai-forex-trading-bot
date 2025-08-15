"""
Utilities Module for Deriv Trading Bot
Contains helper functions, logging, and common utilities
"""

import csv
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config import *


# ==================== LOGGING SETUP ====================
def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Setup main logger
    logger = logging.getLogger('trading_bot')
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # File handler for general logs
    file_handler = logging.FileHandler(BOT_LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)

    # File handler for errors
    error_handler = logging.FileHandler(ERROR_LOG_FILE)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if DEBUG_MODE else logging.WARNING)
    console_handler.setFormatter(simple_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize logger with safe logging for Windows
from safe_logger import get_safe_logger

logger = get_safe_logger('trading_bot')

# ==================== TRADE LOGGING ====================
class TradeLogger:
    """Handles trade logging and CSV export"""

    def __init__(self):
        self.csv_file = TRADE_LOG_FILE
        self.fieldnames = [
            'timestamp', 'action', 'symbol', 'amount', 'price',
            'confidence', 'reason', 'duration', 'result',
            'profit_loss', 'balance', 'trade_id'
        ]
        self.initialize_csv()

    def initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log a trade to CSV file"""
        try:
            # Ensure all required fields exist
            trade_record: Dict[str, Any] = {
                'timestamp': datetime.now().isoformat(),
                'action': trade_data.get('action', ''),
                'symbol': trade_data.get('symbol', ''),
                'amount': trade_data.get('amount', 0),
                'price': trade_data.get('price', 0),
                'confidence': trade_data.get('confidence', 0),
                'reason': trade_data.get('reason', ''),
                'duration': trade_data.get('duration', 0),
                'result': trade_data.get('result', ''),
                'profit_loss': trade_data.get('profit_loss', 0),
                'balance': trade_data.get('balance', 0),
                'trade_id': trade_data.get('trade_id', '')
            }

            with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writerow(trade_record)

            logger.info(f"Trade logged: {trade_record['action']} {trade_record['symbol']} - {trade_record['result']}")

        except Exception as e:
            logger.error(f"Error logging trade: {e}")

    def get_trade_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get trade history from CSV"""
        try:
            if not os.path.exists(self.csv_file):
                return []

            trades: List[Dict[str, Any]] = []
            cutoff_date = datetime.now() - timedelta(days=days)

            with open(self.csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    try:
                        # Check if required fields exist
                        if 'timestamp' not in row or not row['timestamp']:
                            continue

                        trade_date = datetime.fromisoformat(row['timestamp'])
                        if trade_date >= cutoff_date:
                            # Convert numeric fields with robust error handling
                            processed_row = dict(row)  # Create a copy

                            # Handle all numeric fields safely
                            numeric_fields = ['amount', 'price', 'confidence', 'profit_loss', 'balance']
                            for field in numeric_fields:
                                try:
                                    if field in row and row[field] not in [None, '', 'None']:
                                        processed_row[field] = float(row[field])
                                    else:
                                        processed_row[field] = 0.0
                                except (ValueError, TypeError):
                                    processed_row[field] = 0.0

                            # Handle duration separately (int conversion)
                            try:
                                if 'duration' in row and row['duration'] not in [None, '', 'None']:
                                    processed_row['duration'] = int(float(row['duration']))
                                else:
                                    processed_row['duration'] = 0
                            except (ValueError, TypeError):
                                processed_row['duration'] = 0

                            trades.append(processed_row)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing trade record: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Unexpected error parsing trade record: {e}")
                        continue

            return trades

        except Exception as e:
            logger.error(f"Error reading trade history: {e}")
            return []

    def get_daily_stats(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get trading statistics for a specific day"""
        if date is None:
            date = datetime.now()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        trades = self.get_trade_history(days=1)
        daily_trades = [
            trade for trade in trades
            if start_of_day <= datetime.fromisoformat(trade['timestamp']) < end_of_day
        ]

        if not daily_trades:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'avg_duration': 0
            }

        wins = [t for t in daily_trades if t['result'] == 'WIN']
        losses = [t for t in daily_trades if t['result'] == 'LOSS']

        profits = [t['profit_loss'] for t in daily_trades if t['profit_loss']]
        durations = [t['duration'] for t in daily_trades if t['duration']]

        return {
            'total_trades': len(daily_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(daily_trades) if daily_trades else 0,
            'total_profit': sum(profits),
            'best_trade': max(profits) if profits else 0,
            'worst_trade': min(profits) if profits else 0,
            'avg_duration': sum(durations) / len(durations) if durations else 0
        }

# ==================== PERFORMANCE TRACKING ====================
class PerformanceTracker:
    """Track bot performance metrics"""

    def __init__(self):
        self.start_time = datetime.now()
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.last_balance = 0.0
        self.peak_balance = 0.0
        self.max_drawdown = 0.0

    def update_trade_result(self, result: str, profit_loss: float, balance: float) -> None:
        """Update performance metrics with trade result"""
        self.trade_count += 1
        self.total_profit += profit_loss
        self.last_balance = balance

        if balance > self.peak_balance:
            self.peak_balance = balance

        # Calculate drawdown
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        if result == 'WIN':
            self.win_count += 1
            self.consecutive_losses = 0
        elif result == 'LOSS':
            self.loss_count += 1
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'uptime_seconds': uptime,
            'uptime_formatted': format_duration(int(uptime)),
            'total_trades': self.trade_count,
            'wins': self.win_count,
            'losses': self.loss_count,
            'win_rate': self.win_count / self.trade_count if self.trade_count > 0 else 0,
            'total_profit': self.total_profit,
            'avg_profit_per_trade': self.total_profit / self.trade_count if self.trade_count > 0 else 0,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'current_balance': self.last_balance,
            'peak_balance': self.peak_balance,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.calculate_profit_factor()
        }

    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            trade_logger = TradeLogger()
            trades = trade_logger.get_trade_history()

            gross_profit = sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0)
            gross_loss = abs(sum(t['profit_loss'] for t in trades if t['profit_loss'] < 0))

            return gross_profit / gross_loss if gross_loss > 0 else float('inf')

        except Exception:
            return 0.0

# ==================== UTILITY FUNCTIONS ====================
def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount"""
    symbol = "$" if currency == "USD" else currency + " "
    if amount >= 0:
        return f"+{symbol}{amount:.2f}"
    else:
        return f"-{symbol}{abs(amount):.2f}"

def format_percentage(value: float) -> str:
    """Format percentage with sign"""
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

def calculate_profit_loss(entry_price: float, exit_price: float, amount: float, action: str) -> float:
    """Calculate profit/loss for a trade"""
    if action.upper() == "BUY":
        return amount * (exit_price - entry_price) / entry_price
    elif action.upper() == "SELL":
        return amount * (entry_price - exit_price) / entry_price
    else:
        return 0.0

def validate_trade_parameters(symbol: str, amount: float, action: str) -> bool:
    """Validate trade parameters"""
    if symbol not in AVAILABLE_SYMBOLS:
        logger.error(f"Invalid symbol: {symbol}")
        return False

    if amount <= 0 or amount > 1000:  # Reasonable limits
        logger.error(f"Invalid amount: {amount}")
        return False

    if action not in ['BUY', 'SELL']:
        logger.error(f"Invalid action: {action}")
        return False

    return True

def check_risk_limits(consecutive_losses: int, daily_loss: float, balance: float) -> bool:
    """Check if risk limits are exceeded"""
    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        logger.warning(f"Max consecutive losses reached: {consecutive_losses}")
        return False

    if daily_loss >= MAX_DAILY_LOSS:
        logger.warning(f"Daily loss limit reached: ${daily_loss:.2f}")
        return False

    if balance <= MIN_BALANCE:
        logger.warning(f"Balance too low: ${balance:.2f}")
        return False

    return True

def save_json_data(data: Dict[str, Any], filename: str) -> bool:
    """Save data to JSON file"""
    try:
        filepath = os.path.join(LOG_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON data: {e}")
        return False

def load_json_data(filename: str) -> Optional[Dict[str, Any]]:
    """Load data from JSON file"""
    try:
        filepath = os.path.join(LOG_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        return None

def cleanup_old_logs(days_to_keep: int = 30) -> None:
    """Clean up old log files"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        for filename in os.listdir(LOG_DIR):
            filepath = os.path.join(LOG_DIR, filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                if file_time < cutoff_date:
                    os.remove(filepath)
                    logger.info(f"Removed old log file: {filename}")

    except Exception as e:
        logger.error(f"Error cleaning up logs: {e}")

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import platform

    import psutil

    try:
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:').percent
        }
    except ImportError:
        return {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0
        }

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying functions on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    else:
                        logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {e}")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
            return None
        return wrapper
    return decorator

# ==================== DATA ANALYSIS UTILITIES ====================
def calculate_moving_average(data: List[float], window: int) -> List[float]:
    """Calculate simple moving average"""
    if len(data) < window:
        return []

    return [sum(data[i:i+window]) / window for i in range(len(data) - window + 1)]

def calculate_volatility(prices: List[float]) -> float:
    """Calculate price volatility (standard deviation of returns)"""
    if len(prices) < 2:
        return 0.0

    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

    return variance ** 0.5

def detect_outliers(data: List[float], threshold: float = 2.0) -> List[bool]:
    """Detect outliers using z-score method"""
    if len(data) < 3:
        return [False] * len(data)

    mean_val = sum(data) / len(data)
    std_val = (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5

    if std_val == 0:
        return [False] * len(data)

    z_scores = [(x - mean_val) / std_val for x in data]
    return [abs(z) > threshold for z in z_scores]

# ==================== CONFIGURATION HELPERS ====================
def get_effective_config() -> Dict[str, Any]:
    """Get effective configuration with all settings"""
    return {
        'trading': {
            'symbol': DEFAULT_SYMBOL,
            'amount': TRADE_AMOUNT,
            'currency': CURRENCY,
            'duration': DURATION,
            'paper_trading': PAPER_TRADING
        },
        'risk_management': {
            'max_consecutive_losses': MAX_CONSECUTIVE_LOSSES,
            'max_daily_loss': MAX_DAILY_LOSS,
            'min_balance': MIN_BALANCE,
            'enable_risk_management': ENABLE_RISK_MANAGEMENT
        },
        'ai_settings': {
            'model_type': AI_MODEL_TYPE,
            'confidence_threshold': AI_CONFIDENCE_THRESHOLD,
            'enable_ai_predictions': ENABLE_AI_PREDICTIONS
        },
        'features': {
            'telegram_alerts': ENABLE_TELEGRAM_ALERTS,
            'logging': ENABLE_LOGGING,
            'debug_mode': DEBUG_MODE
        }
    }

# Global instances
trade_logger = TradeLogger()
performance_tracker = PerformanceTracker()

if __name__ == "__main__":
    # Test utilities
    print("🔧 Testing Utilities...")

    # Test logging
    logger.info("Test log message")

    # Test trade logging
    test_trade = {
        'action': 'BUY',
        'symbol': 'R_75',
        'amount': 1.0,
        'price': 1.5,
        'confidence': 0.75,
        'reason': 'Test trade',
        'result': 'WIN',
        'profit_loss': 0.85,
        'balance': 100.0,
        'trade_id': 'test_001'
    }

    trade_logger.log_trade(test_trade)
    print("✅ Trade logging test completed")

    # Test performance tracking
    performance_tracker.update_trade_result('WIN', 0.85, 100.0)
    metrics = performance_tracker.get_metrics()
    print(f"✅ Performance tracking test: Win rate = {metrics['win_rate']:.1%}")

    # Test utility functions
    print(f"✅ Currency format test: {format_currency(123.45)}")
    print(f"✅ Duration format test: {format_duration(3661)}")

    print("🎯 All utility tests completed!")
