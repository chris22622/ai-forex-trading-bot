"""
Advanced Performance Analytics System
Provides comprehensive trading performance analysis and reporting
"""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Enhanced trade record with full analytics"""
    trade_id: str
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    exit_price: Optional[float]
    volume: float
    entry_time: datetime
    exit_time: Optional[datetime]
    profit_loss: Optional[float]
    confidence: float
    strategy: str
    duration_minutes: Optional[float]
    max_drawdown: Optional[float]
    max_profit: Optional[float]
    exit_reason: Optional[str]
    market_conditions: Dict[str, float]


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_profit: float
    total_loss: float
    net_profit: float
    average_win: float
    average_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_drawdown: float
    max_profit: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    expectancy: float
    recovery_factor: float
    profit_per_trade: float


class PerformanceAnalyzer:
    """Advanced performance analysis system"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/trading_analytics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for trade storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    action TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    volume REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    profit_loss REAL,
                    confidence REAL,
                    strategy TEXT,
                    duration_minutes REAL,
                    max_drawdown REAL,
                    max_profit REAL,
                    exit_reason TEXT,
                    market_conditions TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    trades_count INTEGER,
                    net_profit REAL,
                    win_rate REAL,
                    max_drawdown REAL,
                    total_volume REAL
                )
            """)
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.trade_id,
                    trade.symbol,
                    trade.action,
                    trade.entry_price,
                    trade.exit_price,
                    trade.volume,
                    trade.entry_time.isoformat() if trade.entry_time else None,
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.profit_loss,
                    trade.confidence,
                    trade.strategy,
                    trade.duration_minutes,
                    trade.max_drawdown,
                    trade.max_profit,
                    trade.exit_reason,
                    json.dumps(trade.market_conditions)
                ))
            logger.info(f"📊 Trade recorded: {trade.trade_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade: {e}")
    
    def get_trades(self, 
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   symbol: Optional[str] = None) -> List[TradeRecord]:
        """Get trades with optional filtering"""
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND entry_time >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND entry_time <= ?"
                params.append(end_date.isoformat())
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY entry_time DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                trades = []
                
                for row in cursor.fetchall():
                    trade = TradeRecord(
                        trade_id=row[0],
                        symbol=row[1],
                        action=row[2],
                        entry_price=row[3],
                        exit_price=row[4],
                        volume=row[5],
                        entry_time=datetime.fromisoformat(row[6]) if row[6] else None,
                        exit_time=datetime.fromisoformat(row[7]) if row[7] else None,
                        profit_loss=row[8],
                        confidence=row[9],
                        strategy=row[10],
                        duration_minutes=row[11],
                        max_drawdown=row[12],
                        max_profit=row[13],
                        exit_reason=row[14],
                        market_conditions=json.loads(row[15]) if row[15] else {}
                    )
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            logger.error(f"❌ Failed to get trades: {e}")
            return []
    
    def calculate_metrics(self, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        trades = self.get_trades(start_date, end_date)
        completed_trades = [t for t in trades if t.profit_loss is not None]
        
        if not completed_trades:
            return PerformanceMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, profit_factor=0, total_profit=0, total_loss=0,
                net_profit=0, average_win=0, average_loss=0,
                max_consecutive_wins=0, max_consecutive_losses=0,
                max_drawdown=0, max_profit=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0, expectancy=0,
                recovery_factor=0, profit_per_trade=0
            )
        
        # Basic metrics
        profits = [t.profit_loss for t in completed_trades if t.profit_loss > 0]
        losses = [t.profit_loss for t in completed_trades if t.profit_loss < 0]
        
        total_trades = len(completed_trades)
        winning_trades = len(profits)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(profits) if profits else 0
        total_loss = abs(sum(losses)) if losses else 0
        net_profit = total_profit - total_loss
        
        average_win = total_profit / winning_trades if winning_trades > 0 else 0
        average_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in completed_trades:
            if trade.profit_loss > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # Advanced metrics
        returns = [t.profit_loss for t in completed_trades]
        returns_array = np.array(returns)
        
        max_drawdown = self._calculate_max_drawdown(returns)
        max_profit = max(returns) if returns else 0
        
        # Risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
        sortino_ratio = self._calculate_sortino_ratio(returns_array)
        calmar_ratio = abs(net_profit / max_drawdown) if max_drawdown != 0 else 0
        
        expectancy = (win_rate * average_win) - ((1 - win_rate) * average_loss)
        recovery_factor = abs(net_profit / max_drawdown) if max_drawdown != 0 else 0
        profit_per_trade = net_profit / total_trades if total_trades > 0 else 0
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit,
            average_win=average_win,
            average_loss=average_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            max_drawdown=max_drawdown,
            max_profit=max_profit,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            expectancy=expectancy,
            recovery_factor=recovery_factor,
            profit_per_trade=profit_per_trade
        )
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(np.min(drawdown))
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        return np.mean(excess_returns) / downside_deviation if downside_deviation != 0 else 0
    
    def generate_performance_report(self, 
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> Dict:
        """Generate comprehensive performance report"""
        metrics = self.calculate_metrics(start_date, end_date)
        trades = self.get_trades(start_date, end_date)
        
        # Symbol analysis
        symbol_performance = {}
        for trade in trades:
            if trade.symbol not in symbol_performance:
                symbol_performance[trade.symbol] = []
            if trade.profit_loss is not None:
                symbol_performance[trade.symbol].append(trade.profit_loss)
        
        symbol_stats = {}
        for symbol, profits in symbol_performance.items():
            if profits:
                symbol_stats[symbol] = {
                    'trades': len(profits),
                    'profit': sum(profits),
                    'win_rate': len([p for p in profits if p > 0]) / len(profits)
                }
        
        # Time analysis
        hourly_performance = {}
        for trade in trades:
            if trade.entry_time and trade.profit_loss is not None:
                hour = trade.entry_time.hour
                if hour not in hourly_performance:
                    hourly_performance[hour] = []
                hourly_performance[hour].append(trade.profit_loss)
        
        report = {
            'period': {
                'start': start_date.isoformat() if start_date else 'All time',
                'end': end_date.isoformat() if end_date else 'Present'
            },
            'overall_metrics': asdict(metrics),
            'symbol_performance': symbol_stats,
            'hourly_performance': {
                hour: {
                    'trades': len(profits),
                    'net_profit': sum(profits),
                    'avg_profit': sum(profits) / len(profits)
                }
                for hour, profits in hourly_performance.items()
            },
            'risk_analysis': {
                'max_single_loss': min([t.profit_loss for t in trades if t.profit_loss], default=0),
                'max_single_gain': max([t.profit_loss for t in trades if t.profit_loss], default=0),
                'volatility': np.std([t.profit_loss for t in trades if t.profit_loss]) if trades else 0
            }
        }
        
        return report
    
    def export_report(self, 
                     report: Dict,
                     export_path: Path,
                     format: str = 'json') -> bool:
        """Export performance report"""
        try:
            if format.lower() == 'json':
                with open(export_path, 'w') as f:
                    json.dump(report, f, indent=4, default=str)
            
            logger.info(f"📊 Report exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export report: {e}")
            return False


# Global performance analyzer
performance_analyzer = PerformanceAnalyzer()


def record_trade(trade: TradeRecord):
    """Record a trade in the global analyzer"""
    performance_analyzer.record_trade(trade)


def get_performance_metrics(days: int = 30) -> PerformanceMetrics:
    """Get performance metrics for the last N days"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return performance_analyzer.calculate_metrics(start_date, end_date)


def generate_daily_report() -> Dict:
    """Generate daily performance report"""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    return performance_analyzer.generate_performance_report(today, tomorrow)


if __name__ == "__main__":
    # Demo the performance analytics system
    print("📊 Advanced Performance Analytics Demo")
    
    analyzer = PerformanceAnalyzer()
    
    # Create sample trade
    sample_trade = TradeRecord(
        trade_id="DEMO_001",
        symbol="EURUSD",
        action="BUY",
        entry_price=1.1250,
        exit_price=1.1275,
        volume=0.1,
        entry_time=datetime.now() - timedelta(minutes=30),
        exit_time=datetime.now(),
        profit_loss=25.0,
        confidence=0.75,
        strategy="AI_ENSEMBLE",
        duration_minutes=30,
        max_drawdown=5.0,
        max_profit=30.0,
        exit_reason="TAKE_PROFIT",
        market_conditions={"volatility": 0.8, "trend": 1.2}
    )
    
    analyzer.record_trade(sample_trade)
    
    # Generate metrics
    metrics = analyzer.calculate_metrics()
    print(f"📈 Performance metrics: {metrics}")
    
    # Generate report
    report = analyzer.generate_performance_report()
    print(f"📋 Performance report generated with {len(report)} sections")
    
    print("🎯 Performance analytics ready!")
