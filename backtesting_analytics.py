"""
Comprehensive Backtesting Engine with Analytics and Performance Tracking
"""

import pandas as pd
import numpy as np
import json
import sqlite3
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import io
import base64

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️ Matplotlib/Seaborn not available. Chart generation disabled.")

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("⚠️ ReportLab not available. PDF generation disabled.")

class MarketDataRecorder:
    """Records and stores market data for backtesting"""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.logger = self._setup_logger()
        self._initialize_database()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MarketDataRecorder')
        logger.setLevel(logging.INFO)
        return logger
        
    def _initialize_database(self):
        """Initialize SQLite database for market data"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Market ticks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    symbol TEXT,
                    price REAL,
                    quote_id TEXT,
                    pip_size REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Technical indicators table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    symbol TEXT,
                    rsi REAL,
                    ema_fast REAL,
                    ema_slow REAL,
                    macd REAL,
                    macd_signal REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    volume_proxy REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trading signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    symbol TEXT,
                    signal_type TEXT,
                    signal_value TEXT,
                    confidence REAL,
                    method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticks_timestamp ON market_ticks(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_timestamp ON indicators(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON trading_signals(timestamp)')
            
            conn.commit()
            
    def record_tick(self, timestamp: int, symbol: str, price: float, 
                   quote_id: str = "", pip_size: float = 0.00001):
        """Record a market tick"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO market_ticks (timestamp, symbol, price, quote_id, pip_size)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, symbol, price, quote_id, pip_size))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to record tick: {e}")
            
    def record_indicators(self, timestamp: int, symbol: str, indicators: Dict[str, float]):
        """Record technical indicators"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO indicators (timestamp, symbol, rsi, ema_fast, ema_slow, 
                                          macd, macd_signal, bb_upper, bb_lower, volume_proxy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, symbol,
                    indicators.get('rsi'), indicators.get('ema_fast'), indicators.get('ema_slow'),
                    indicators.get('macd'), indicators.get('macd_signal'),
                    indicators.get('bb_upper'), indicators.get('bb_lower'),
                    indicators.get('volume_proxy', 0)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to record indicators: {e}")
            
    def record_signal(self, timestamp: int, symbol: str, signal_type: str, 
                     signal_value: str, confidence: float, method: str):
        """Record trading signal"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trading_signals (timestamp, symbol, signal_type, signal_value, confidence, method)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp, symbol, signal_type, signal_value, confidence, method))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to record signal: {e}")
            
    def get_historical_data(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        """Get historical market data for backtesting"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT t.timestamp, t.price, 
                           i.rsi, i.ema_fast, i.ema_slow, i.macd, i.macd_signal,
                           i.bb_upper, i.bb_lower, i.volume_proxy
                    FROM market_ticks t
                    LEFT JOIN indicators i ON t.timestamp = i.timestamp AND t.symbol = i.symbol
                    WHERE t.symbol = ? AND t.timestamp BETWEEN ? AND ?
                    ORDER BY t.timestamp
                '''
                
                df = pd.read_sql_query(query, conn, params=(symbol, start_time, end_time))
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()

class BacktestingEngine:
    """Advanced backtesting engine for strategy validation"""
    
    def __init__(self, initial_balance: float = 1000.0, trade_amount: float = 1.0):
        self.initial_balance = initial_balance
        self.trade_amount = trade_amount
        self.logger = self._setup_logger()
        
        # Backtesting state
        self.reset_backtest()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('BacktestingEngine')
        logger.setLevel(logging.INFO)
        return logger
        
    def reset_backtest(self):
        """Reset backtesting state"""
        self.current_balance = self.initial_balance
        self.trades: List[Dict[str, Any]] = []
        self.daily_balances: List[Dict[str, Any]] = []
        self.drawdown_history: List[float] = []
        self.equity_curve: List[float] = [self.initial_balance]
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        
    def calculate_trade_outcome(self, entry_price: float, exit_price: float, 
                              action: str, duration: int = 10) -> Dict[str, Any]:
        """Calculate trade outcome based on price movement"""
        
        # Binary options logic
        if action == "BUY":
            is_win = exit_price > entry_price
        elif action == "SELL":
            is_win = exit_price < entry_price
        else:  # HOLD
            return {"status": "no_trade", "payout": 0, "profit_loss": 0}
        
        # Calculate payout (typical binary options: 85% profit on win, total loss on loss)
        if is_win:
            payout = self.trade_amount * 1.85
            profit_loss = payout - self.trade_amount
        else:
            payout = 0
            profit_loss = -self.trade_amount
            
        return {
            "status": "won" if is_win else "lost",
            "payout": payout,
            "profit_loss": profit_loss,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "price_change": exit_price - entry_price,
            "price_change_percent": ((exit_price - entry_price) / entry_price) * 100
        }
    
    def execute_trade(self, timestamp: int, price: float, action: str, 
                     confidence: float, method: str, trade_duration: int = 10) -> Optional[str]:
        """Execute a trade in the backtest"""
        
        if action == "HOLD":
            return None
            
        trade_id = f"bt_{timestamp}_{len(self.trades)}"
        
        # Create trade record
        trade = {
            "trade_id": trade_id,
            "timestamp": timestamp,
            "action": action,
            "entry_price": price,
            "confidence": confidence,
            "method": method,
            "amount": self.trade_amount,
            "duration": trade_duration,
            "status": "open"
        }
        
        # Add to active trades
        self.active_trades[trade_id] = trade
        
        return trade_id
    
    def close_trade(self, trade_id: str, timestamp: int, exit_price: float) -> bool:
        """Close an active trade"""
        
        if trade_id not in self.active_trades:
            return False
            
        trade = self.active_trades[trade_id]
        
        # Calculate outcome
        outcome = self.calculate_trade_outcome(
            trade["entry_price"], exit_price, trade["action"], trade["duration"]
        )
        
        # Update trade record
        trade.update(outcome)
        trade["exit_price"] = exit_price
        trade["exit_timestamp"] = timestamp
        trade["actual_duration"] = timestamp - trade["timestamp"]
        trade["status"] = "closed"
        
        # Update balance
        self.current_balance += outcome["profit_loss"]
        self.equity_curve.append(self.current_balance)
        
        # Track drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        self.drawdown_history.append(current_drawdown)
        
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Move to completed trades
        self.trades.append(trade)
        del self.active_trades[trade_id]
        
        return True
    
    def run_backtest(self, historical_data: pd.DataFrame, strategy_func, 
                    trade_duration: int = 10) -> Dict[str, Any]:
        """Run complete backtest on historical data"""
        
        self.reset_backtest()
        self.logger.info(f"Starting backtest with {len(historical_data)} data points")
        
        for i, row in historical_data.iterrows():
            try:
                timestamp = int(row['timestamp'])
                price = row['price']
                
                # Prepare indicators dictionary
                indicators = {
                    'price': price,
                    'rsi': row.get('rsi'),
                    'ema_fast': row.get('ema_fast'),
                    'ema_slow': row.get('ema_slow'),
                    'macd': row.get('macd'),
                    'macd_signal': row.get('macd_signal'),
                    'bb_upper': row.get('bb_upper'),
                    'bb_lower': row.get('bb_lower'),
                    'volume_proxy': row.get('volume_proxy', 0)
                }
                
                # Get recent price history
                start_idx = max(0, i - 50)
                price_history = historical_data.iloc[start_idx:i+1]['price'].tolist()
                
                # Get strategy decision
                decision = strategy_func(indicators, price_history)
                action = decision.get('prediction', 'HOLD')
                confidence = decision.get('confidence', 0.5)
                method = decision.get('method', 'UNKNOWN')
                
                # Execute trade
                if action != 'HOLD':
                    trade_id = self.execute_trade(timestamp, price, action, confidence, method, trade_duration)
                
                # Check for trade closures
                closed_trades = []
                for trade_id, trade in self.active_trades.items():
                    if timestamp >= trade['timestamp'] + trade['duration']:
                        self.close_trade(trade_id, timestamp, price)
                        closed_trades.append(trade_id)
                
                # Record daily balance
                if i % 100 == 0:  # Every 100 ticks
                    self.daily_balances.append({
                        'timestamp': timestamp,
                        'balance': self.current_balance,
                        'open_trades': len(self.active_trades),
                        'total_trades': len(self.trades)
                    })
                    
            except Exception as e:
                self.logger.error(f"Backtest error at row {i}: {e}")
                continue
        
        # Close any remaining open trades
        final_price = historical_data.iloc[-1]['price']
        final_timestamp = int(historical_data.iloc[-1]['timestamp'])
        
        for trade_id in list(self.active_trades.keys()):
            self.close_trade(trade_id, final_timestamp, final_price)
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics()
        
        self.logger.info(f"Backtest completed: {len(self.trades)} trades, "
                        f"Final balance: ${self.current_balance:.2f}")
        
        return results
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if not self.trades:
            return {"error": "No trades executed"}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.get('profit_loss', 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades
        
        # Profit/Loss metrics
        total_profit = sum(t.get('profit_loss', 0) for t in self.trades)
        average_profit = total_profit / total_trades
        winning_profits = [t['profit_loss'] for t in self.trades if t.get('profit_loss', 0) > 0]
        losing_profits = [t['profit_loss'] for t in self.trades if t.get('profit_loss', 0) < 0]
        
        avg_win = np.mean(winning_profits) if winning_profits else 0
        avg_loss = np.mean(losing_profits) if losing_profits else 0
        profit_factor = abs(sum(winning_profits) / sum(losing_profits)) if losing_profits else float('inf')
        
        # Drawdown metrics
        max_consecutive_losses = self._calculate_max_consecutive_losses()
        
        # Returns
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Time-based metrics
        if self.trades:
            start_time = min(t['timestamp'] for t in self.trades)
            end_time = max(t['exit_timestamp'] for t in self.trades if 'exit_timestamp' in t)
            duration_days = (end_time - start_time) / (24 * 3600)
            trades_per_day = total_trades / max(duration_days, 1)
        else:
            duration_days = 0
            trades_per_day = 0
        
        return {
            # Basic metrics
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            
            # Financial metrics
            "initial_balance": self.initial_balance,
            "final_balance": self.current_balance,
            "total_profit": total_profit,
            "total_return": total_return,
            "average_profit_per_trade": average_profit,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "profit_factor": profit_factor,
            
            # Risk metrics
            "max_drawdown": self.max_drawdown,
            "max_consecutive_losses": max_consecutive_losses,
            "sharpe_ratio": sharpe_ratio,
            
            # Activity metrics
            "duration_days": duration_days,
            "trades_per_day": trades_per_day,
            
            # Additional data
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "daily_balances": self.daily_balances,
            "drawdown_history": self.drawdown_history
        }
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade.get('profit_loss', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive

class PerformanceAnalyzer:
    """Advanced performance analytics and visualization"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('PerformanceAnalyzer')
        logger.setLevel(logging.INFO)
        return logger
    
    def generate_performance_report(self, backtest_results: Dict[str, Any], 
                                  output_dir: str = "reports") -> Dict[str, str]:
        """Generate comprehensive performance report"""
        
        os.makedirs(output_dir, exist_ok=True)
        report_files = {}
        
        # Generate charts
        if HAS_PLOTTING:
            chart_files = self.generate_charts(backtest_results, output_dir)
            report_files.update(chart_files)
        
        # Generate PDF report
        if HAS_REPORTLAB:
            pdf_file = self.generate_pdf_report(backtest_results, output_dir)
            report_files['pdf_report'] = pdf_file
        
        # Generate CSV export
        csv_file = self.export_trades_csv(backtest_results, output_dir)
        report_files['trades_csv'] = csv_file
        
        # Generate JSON summary
        json_file = self.export_summary_json(backtest_results, output_dir)
        report_files['summary_json'] = json_file
        
        return report_files
    
    def generate_charts(self, results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Generate performance charts"""
        
        if not HAS_PLOTTING:
            return {}
        
        chart_files = {}
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Equity Curve
        fig, ax = plt.subplots(figsize=(12, 6))
        equity_curve = results.get('equity_curve', [])
        if equity_curve:
            ax.plot(range(len(equity_curve)), equity_curve, linewidth=2, color='blue')
            ax.set_title('Equity Curve', fontsize=16, fontweight='bold')
            ax.set_xlabel('Trade Number')
            ax.set_ylabel('Balance ($)')
            ax.grid(True, alpha=0.3)
            
            # Add performance text
            final_balance = equity_curve[-1]
            initial_balance = equity_curve[0]
            total_return = (final_balance - initial_balance) / initial_balance * 100
            ax.text(0.02, 0.98, f'Total Return: {total_return:.1f}%', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        equity_file = os.path.join(output_dir, 'equity_curve.png')
        plt.savefig(equity_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files['equity_curve'] = equity_file
        
        # 2. Drawdown Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        drawdown = results.get('drawdown_history', [])
        if drawdown:
            ax.fill_between(range(len(drawdown)), drawdown, 0, 
                           color='red', alpha=0.3, label='Drawdown')
            ax.plot(range(len(drawdown)), drawdown, color='red', linewidth=1)
            ax.set_title('Drawdown Analysis', fontsize=16, fontweight='bold')
            ax.set_xlabel('Trade Number')
            ax.set_ylabel('Drawdown (%)')
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            
            max_dd = max(drawdown) if drawdown else 0
            ax.text(0.02, 0.98, f'Max Drawdown: {max_dd:.1%}', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        drawdown_file = os.path.join(output_dir, 'drawdown.png')
        plt.savefig(drawdown_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files['drawdown'] = drawdown_file
        
        # 3. Trade Distribution
        trades = results.get('trades', [])
        if trades:
            profits = [t.get('profit_loss', 0) for t in trades]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram of profits/losses
            ax1.hist(profits, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax1.set_title('Profit/Loss Distribution')
            ax1.set_xlabel('Profit/Loss ($)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Win/Loss pie chart
            wins = len([p for p in profits if p > 0])
            losses = len([p for p in profits if p <= 0])
            
            ax2.pie([wins, losses], labels=['Wins', 'Losses'], 
                   autopct='%1.1f%%', startangle=90,
                   colors=['green', 'red'], explode=(0.05, 0))
            ax2.set_title('Win/Loss Ratio')
        
            plt.tight_layout()
            distribution_file = os.path.join(output_dir, 'trade_distribution.png')
            plt.savefig(distribution_file, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files['trade_distribution'] = distribution_file
        
        # 4. Performance Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create performance matrix
        metrics = {
            'Win Rate': results.get('win_rate', 0),
            'Profit Factor': min(results.get('profit_factor', 0), 5),  # Cap at 5 for visualization
            'Sharpe Ratio': results.get('sharpe_ratio', 0),
            'Max Drawdown': results.get('max_drawdown', 0),
            'Total Return': results.get('total_return', 0),
        }
        
        # Normalize values for heatmap
        normalized_metrics = {}
        for key, value in metrics.items():
            if key == 'Max Drawdown':
                normalized_metrics[key] = 1 - min(value, 1)  # Invert drawdown (lower is better)
            else:
                normalized_metrics[key] = min(max(value, 0), 1)  # Clamp between 0 and 1
        
        # Create heatmap
        data = np.array(list(normalized_metrics.values())).reshape(1, -1)
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   xticklabels=list(normalized_metrics.keys()),
                   yticklabels=['Performance'], ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title('Performance Heatmap')
        
        plt.tight_layout()
        heatmap_file = os.path.join(output_dir, 'performance_heatmap.png')
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files['performance_heatmap'] = heatmap_file
        
        return chart_files
    
    def generate_pdf_report(self, results: Dict[str, Any], output_dir: str) -> str:
        """Generate PDF performance report"""
        
        if not HAS_REPORTLAB:
            return ""
        
        pdf_file = os.path.join(output_dir, f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        story.append(Paragraph("Trading Bot Performance Report", title_style))
        story.append(Spacer(1, 20))
        
        # Summary Table
        summary_data = [
            ['Metric', 'Value'],
            ['Total Trades', str(results.get('total_trades', 0))],
            ['Win Rate', f"{results.get('win_rate', 0):.1%}"],
            ['Total Return', f"{results.get('total_return', 0):.1%}"],
            ['Profit Factor', f"{results.get('profit_factor', 0):.2f}"],
            ['Max Drawdown', f"{results.get('max_drawdown', 0):.1%}"],
            ['Sharpe Ratio', f"{results.get('sharpe_ratio', 0):.2f}"],
            ['Final Balance', f"${results.get('final_balance', 0):.2f}"],
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Performance Summary", styles['Heading2']))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Add charts if available
        chart_files = {
            'equity_curve.png': 'Equity Curve',
            'drawdown.png': 'Drawdown Analysis',
            'trade_distribution.png': 'Trade Distribution',
            'performance_heatmap.png': 'Performance Heatmap'
        }
        
        for chart_file, chart_title in chart_files.items():
            chart_path = os.path.join(output_dir, chart_file)
            if os.path.exists(chart_path):
                story.append(Paragraph(chart_title, styles['Heading2']))
                img = Image(chart_path, width=6*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 20))
        
        # Recent Trades Table
        if results.get('trades'):
            recent_trades = results['trades'][-10:]  # Last 10 trades
            trades_data = [['Trade ID', 'Action', 'Entry Price', 'Exit Price', 'P&L', 'Result']]
            
            for trade in recent_trades:
                trades_data.append([
                    trade.get('trade_id', '')[:15],  # Truncate long IDs
                    trade.get('action', ''),
                    f"{trade.get('entry_price', 0):.4f}",
                    f"{trade.get('exit_price', 0):.4f}",
                    f"{trade.get('profit_loss', 0):.2f}",
                    trade.get('status', '')
                ])
            
            trades_table = Table(trades_data)
            trades_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Recent Trades", styles['Heading2']))
            story.append(trades_table)
        
        # Build PDF
        doc.build(story)
        return pdf_file
    
    def export_trades_csv(self, results: Dict[str, Any], output_dir: str) -> str:
        """Export trades to CSV"""
        csv_file = os.path.join(output_dir, f'trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        trades = results.get('trades', [])
        if trades:
            df = pd.DataFrame(trades)
            df.to_csv(csv_file, index=False)
        
        return csv_file
    
    def export_summary_json(self, results: Dict[str, Any], output_dir: str) -> str:
        """Export summary to JSON"""
        json_file = os.path.join(output_dir, f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        # Remove non-serializable data
        summary = results.copy()
        summary.pop('trades', None)
        summary.pop('equity_curve', None)
        summary.pop('daily_balances', None)
        summary.pop('drawdown_history', None)
        
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return json_file

# Example usage and testing
if __name__ == "__main__":
    # Example strategy function
    def simple_rsi_strategy(indicators: Dict[str, Any], price_history: List[float]) -> Dict[str, Any]:
        rsi = indicators.get('rsi', 50)
        
        if rsi < 30:
            return {"prediction": "BUY", "confidence": 0.8, "method": "RSI_OVERSOLD"}
        elif rsi > 70:
            return {"prediction": "SELL", "confidence": 0.8, "method": "RSI_OVERBOUGHT"}
        else:
            return {"prediction": "HOLD", "confidence": 0.3, "method": "RSI_NEUTRAL"}
    
    # Test backtesting engine
    print("Testing Backtesting Engine...")
    
    # Create sample data
    np.random.seed(42)
    n_points = 1000
    timestamps = range(n_points)
    base_price = 100
    prices = [base_price]
    
    for i in range(1, n_points):
        change = np.random.normal(0, 0.1)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create sample indicators
    rsi_values = [50 + 20 * np.sin(i / 50) + np.random.normal(0, 5) for i in range(n_points)]
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'rsi': rsi_values
    })
    
    # Run backtest
    engine = BacktestingEngine(initial_balance=1000, trade_amount=10)
    results = engine.run_backtest(data, simple_rsi_strategy)
    
    print(f"Backtest Results:")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Total Return: {results['total_return']:.1%}")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    
    # Generate reports
    analyzer = PerformanceAnalyzer()
    report_files = analyzer.generate_performance_report(results)
    print(f"Reports generated: {list(report_files.keys())}")
