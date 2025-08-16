"""
Enhanced Streamlit Dashboard for Advanced Trading Bot Analytics
"""

import json
import pathlib
import queue
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Try to import our advanced modules
try:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from advanced_config import ConfigManager, get_config
    from performance_analytics import PerformanceAnalyzer, get_performance_metrics
except ImportError as e:
    st.error(f"Could not import advanced modules: {e}")
    ConfigManager = None
    PerformanceAnalyzer = None

# --- App Setup ---
st.set_page_config(
    page_title="🚀 AI Forex Trading Bot — Advanced Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚀 AI Forex Trading Bot</h1>', unsafe_allow_html=True)
st.markdown(
    '<h3 style="text-align: center; color: #666;">Advanced Analytics & Control Dashboard</h3>',
    unsafe_allow_html=True
)

# Paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "ui_runtime.log"

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
if "mode" not in st.session_state:
    st.session_state.mode = "Demo"
if "log_q" not in st.session_state:
    st.session_state.log_q = queue.Queue()
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["time", "price", "action", "profit"])
if "config_manager" not in st.session_state and ConfigManager:
    st.session_state.config_manager = ConfigManager()
if "performance_analyzer" not in st.session_state and PerformanceAnalyzer:
    st.session_state.performance_analyzer = PerformanceAnalyzer()

stop_event = threading.Event()

# Sidebar Navigation
st.sidebar.title("🎛️ Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["🎮 Trading Control", "📊 Performance Analytics", "⚙️ Configuration", "📈 Live Charts", "📋 Reports"]
)

# --- Logging helper ---
def log(msg: str):
    line = f"[{datetime.utcnow().isoformat()}] {msg}"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        st.session_state.log_q.put_nowait(line)
    except queue.Full:
        pass

# --- Demo engine (enhanced) ---
def enhanced_demo_engine(stop: threading.Event, config: Dict):
    """Enhanced demo engine with realistic trading simulation"""
    log("🚀 Enhanced demo engine starting...")
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    base_prices = {"EURUSD": 1.1000, "GBPUSD": 1.3000, "USDJPY": 110.0, "AUDUSD": 0.7500, "USDCAD": 1.2500}
    
    trade_count = 0
    session_profit = 0
    
    while not stop.is_set():
        try:
            # Simulate market conditions
            symbol = random.choice(symbols)
            base_price = base_prices[symbol]
            
            # Random walk price movement
            price_change = random.gauss(0, 0.001) * base_price
            current_price = base_price + price_change
            base_prices[symbol] = current_price
            
            # Generate trading signal with enhanced logic
            confidence = random.random()
            volatility = abs(price_change / base_price) * 100
            
            if confidence > config.get('confidence_threshold', 0.65) and trade_count < config.get('max_concurrent_trades', 5):
                action = "BUY" if random.random() > 0.5 else "SELL"
                
                # Simulate trade execution
                entry_price = current_price
                volume = random.uniform(0.01, 0.1)
                
                # Simulate trade outcome based on confidence
                success_probability = confidence * 0.8 + 0.1  # 10-90% success rate
                profit_base = volume * 1000  # Base profit calculation
                
                if random.random() < success_probability:
                    profit = profit_base * random.uniform(0.5, 2.0)
                    exit_reason = "TAKE_PROFIT"
                else:
                    profit = -profit_base * random.uniform(0.3, 1.0)
                    exit_reason = "STOP_LOSS"
                
                session_profit += profit
                trade_count += 1
                
                # Add to data
                new_row = pd.DataFrame({
                    "time": [datetime.now()],
                    "price": [current_price],
                    "action": [action],
                    "profit": [profit],
                    "symbol": [symbol],
                    "confidence": [confidence],
                    "volume": [volume],
                    "exit_reason": [exit_reason]
                })
                
                                st.session_state.data = pd.concat(
                    [st.session_state.data,
                    new_row],
                    ignore_index=True
                )
                
                f"💰 {action}"
                f"{symbol} @ {current_price:.4f} | Profit: ${profit:+.2f} | Confidence: {confidence:.1%}"
            
            time.sleep(random.uniform(2, 8))  # Variable delay
            
        except Exception as e:
            log(f"❌ Demo engine error: {e}")
            time.sleep(5)
    
    log(f"🏁 Demo session ended. Total trades: {trade_count}, Session P&L: ${session_profit:+.2f}")

# --- Live engine placeholder ---
def live_engine(stop: threading.Event):
    """Live trading engine placeholder"""
    log("🔴 Live engine starting (placeholder mode)")
    log("⚠️ Live trading requires MetaTrader5 setup and configuration")
    
    while not stop.is_set():
        log("📡 Live engine monitoring... (demo mode)")
        time.sleep(30)

# --- Page: Trading Control ---
if page == "🎮 Trading Control":
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("🎮 Trading Control Panel")
        
        # Mode selection
                mode = st.selectbox(
            "Trading Mode",
            ["Demo",
            "Live"],
            index=0 if st.session_state.mode == "Demo" else 1
        )
        st.session_state.mode = mode
        
        # Trading parameters
        with st.expander("🎯 Trading Parameters"):
            col_a, col_b = st.columns(2)
            with col_a:
                confidence_threshold = st.slider("Confidence Threshold", 0.3, 0.95, 0.65, 0.05)
                max_trades = st.slider("Max Concurrent Trades", 1, 10, 5)
            with col_b:
                profit_target = st.number_input("Profit Target ($)", 1.0, 100.0, 10.0, 1.0)
                stop_loss = st.number_input("Stop Loss ($)", 1.0, 50.0, 5.0, 1.0)
    
    with col2:
        st.subheader("📊 Session Status")
        
        # Status indicators
        status_color = "status-running" if st.session_state.running else "status-stopped"
        status_text = "🟢 RUNNING" if st.session_state.running else "🔴 STOPPED"
        st.markdown(f'<p class="{status_color}">{status_text}</p>', unsafe_allow_html=True)
        
        st.metric("Mode", st.session_state.mode)
        st.metric("Trades Today", len(st.session_state.data))
    
    with col3:
        st.subheader("💰 Quick Stats")
        
        if len(st.session_state.data) > 0:
            total_profit = st.session_state.data['profit'].sum()
            win_rate = len(st.session_state.data[st.session_state.data['profit'] > 0]) / len(st.session_state.data) * 100
            
            st.metric("Session P&L", f"${total_profit:+.2f}")
            st.metric("Win Rate", f"{win_rate:.1f}%")
        else:
            st.metric("Session P&L", "$0.00")
            st.metric("Win Rate", "0%")
    
    # Control buttons
    st.subheader("🕹️ Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Trading", type="primary", disabled=st.session_state.running):
            st.session_state.running = True
            st.session_state.data = st.session_state.data.iloc[0:0]  # Clear data
            
            config = {
                'confidence_threshold': confidence_threshold,
                'max_concurrent_trades': max_trades,
                'profit_target': profit_target,
                'stop_loss': stop_loss
            }
            
            if mode == "Demo":
                                t = threading.Thread(
                    target=enhanced_demo_engine,
                    args=(stop_event, config),
                    daemon=True
                )
            else:
                t = threading.Thread(target=live_engine, args=(stop_event,), daemon=True)
            
            t.start()
            st.success(f"🚀 {mode} mode started!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Trading", disabled=not st.session_state.running):
            st.session_state.running = False
            stop_event.set()
            st.warning("⏹️ Trading stopped!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Session"):
            st.session_state.running = False
            stop_event.set()
            st.session_state.data = pd.DataFrame(columns=["time", "price", "action", "profit"])
            st.info("🔄 Session reset!")
            st.rerun()

# --- Page: Performance Analytics ---
elif page == "📊 Performance Analytics":
    st.subheader("📊 Advanced Performance Analytics")
    
    if len(st.session_state.data) > 0:
        df = st.session_state.data
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trades = len(df)
            st.metric("Total Trades", total_trades)
        
        with col2:
            total_profit = df['profit'].sum()
            st.metric("Total P&L", f"${total_profit:+.2f}")
        
        with col3:
            winning_trades = len(df[df['profit'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col4:
            avg_profit = df['profit'].mean()
            st.metric("Avg Profit/Trade", f"${avg_profit:+.2f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Cumulative P&L")
            df['cumulative_profit'] = df['profit'].cumsum()
            
            fig = px.line(df, x='time', y='cumulative_profit', 
                         title="Cumulative Profit/Loss Over Time")
            fig.update_traces(line_color='#1f77b4', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Trade Distribution")
            
            # Profit distribution histogram
            fig = px.histogram(df, x='profit', nbins=20, 
                             title="Profit Distribution", 
                             color_discrete_sequence=['#2ca02c'])
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Symbol performance
        if 'symbol' in df.columns:
            st.subheader("🌍 Symbol Performance")
            symbol_perf = df.groupby('symbol').agg({
                'profit': ['sum', 'count', 'mean']
            }).round(2)
            symbol_perf.columns = ['Total P&L', 'Trades', 'Avg P&L']
            st.dataframe(symbol_perf, use_container_width=True)
    
    else:
        st.info("📊 No trading data available. Start a trading session to see analytics.")

# --- Page: Configuration ---
elif page == "⚙️ Configuration":
    st.subheader("⚙️ Advanced Configuration")
    
    if ConfigManager and st.session_state.config_manager:
        config = st.session_state.config_manager.get_config()
        
        with st.expander("🎯 Risk Management", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                                max_daily_loss = st.number_input(
                    "Max Daily Loss ($)",
                    10.0,
                    1000.0,
                    config.max_daily_loss,
                    10.0
                )
                                max_concurrent = st.slider(
                    "Max Concurrent Trades",
                    1,
                    20,
                    config.max_concurrent_trades
                )
                                position_size = st.slider(
                    "Position Size (%)",
                    0.1,
                    10.0,
                    config.position_size_percent,
                    0.1
                )
            
            with col2:
                stop_loss = st.slider("Stop Loss (%)", 0.1, 5.0, config.stop_loss_percent, 0.1)
                                take_profit = st.slider(
                    "Take Profit (%)",
                    0.5,
                    10.0,
                    config.take_profit_percent,
                    0.1
                )
                                confidence_thresh = st.slider(
                    "Confidence Threshold",
                    0.3,
                    0.95,
                    config.confidence_threshold,
                    0.05
                )
        
        with st.expander("🤖 AI & Strategy Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                                retrain_freq = st.number_input(
                    "Model Retrain Frequency",
                    10,
                    1000,
                    config.ai_model_retrain_frequency,
                    10
                )
                                ensemble_weights = st.checkbox(
                    "Dynamic Ensemble Weights",
                    config.ensemble_weight_adjustment
                )
            
            with col2:
                                confidence_cal = st.checkbox(
                    "Confidence Calibration",
                    config.confidence_calibration
                )
                                adaptive_sizing = st.checkbox(
                    "Adaptive Position Sizing",
                    config.adaptive_position_sizing
                )
        
        with st.expander("📱 Notifications"):
            telegram_enabled = st.checkbox("Telegram Notifications", config.telegram_enabled)
            notification_levels = st.multiselect(
                "Notification Levels",
                ["ERROR", "WARNING", "INFO", "TRADE", "PROFIT", "LOSS"],
                config.notification_levels
            )
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary"):
            updates = {
                'max_daily_loss': max_daily_loss,
                'max_concurrent_trades': max_concurrent,
                'position_size_percent': position_size,
                'stop_loss_percent': stop_loss,
                'take_profit_percent': take_profit,
                'confidence_threshold': confidence_thresh,
                'ai_model_retrain_frequency': retrain_freq,
                'ensemble_weight_adjustment': ensemble_weights,
                'confidence_calibration': confidence_cal,
                'adaptive_position_sizing': adaptive_sizing,
                'telegram_enabled': telegram_enabled,
                'notification_levels': notification_levels
            }
            
            if st.session_state.config_manager.update_config(updates):
                st.success("✅ Configuration saved successfully!")
            else:
                st.error("❌ Failed to save configuration!")
    
    else:
        st.warning("⚠️ Advanced configuration not available. Using basic settings.")

# --- Page: Live Charts ---
elif page == "📈 Live Charts":
    st.subheader("📈 Real-Time Trading Charts")
    
    if len(st.session_state.data) > 0:
        df = st.session_state.data.copy()
        
        # Main price chart with trades
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Price Action & Trades", "Profit/Loss"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price line
        if 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol]
                fig.add_trace(
                    go.Scatter(x=symbol_data['time'], y=symbol_data['price'],
                             mode='lines', name=f'{symbol} Price'),
                    row=1, col=1
                )
        else:
            fig.add_trace(
                go.Scatter(x=df['time'], y=df['price'],
                         mode='lines', name='Price'),
                row=1, col=1
            )
        
        # Trade markers
        buy_trades = df[df['action'] == 'BUY']
        sell_trades = df[df['action'] == 'SELL']
        
        if len(buy_trades) > 0:
            fig.add_trace(
                go.Scatter(x=buy_trades['time'], y=buy_trades['price'],
                         mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                         name='Buy Trades'),
                row=1, col=1
            )
        
        if len(sell_trades) > 0:
            fig.add_trace(
                go.Scatter(x=sell_trades['time'], y=sell_trades['price'],
                         mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
                         name='Sell Trades'),
                row=1, col=1
            )
        
        # Profit chart
        df['cumulative_profit'] = df['profit'].cumsum()
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['cumulative_profit'],
                     mode='lines', name='Cumulative P&L', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title="Live Trading Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent trades table
        st.subheader("🔄 Recent Trades")
        recent_trades = df.tail(10)[['time', 'action', 'price', 'profit']].copy()
        recent_trades['time'] = recent_trades['time'].dt.strftime('%H:%M:%S')
        recent_trades['profit'] = recent_trades['profit'].apply(lambda x: f"${x:+.2f}")
        st.dataframe(recent_trades, use_container_width=True)
    
    else:
                st.info("📈 No trading data available for charts. Start "
        "a trading session to see live charts.")

# --- Page: Reports ---
elif page == "📋 Reports":
    st.subheader("📋 Trading Reports & Analysis")
    
    # Report generation
        report_type = st.selectbox(
        "Report Type",
        ["Daily Summary",
        "Weekly Analysis",
        "Monthly Report",
        "Custom Period"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if report_type == "Custom Period":
            start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
            end_date = st.date_input("End Date", datetime.now().date())
    
    with col2:
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "PDF"])
    
    if st.button("📊 Generate Report", type="primary"):
        if len(st.session_state.data) > 0:
            # Generate basic report
            df = st.session_state.data
            
            report_data = {
                "report_type": report_type,
                "generated_at": datetime.now().isoformat(),
                "period": {
                    "start": start_date.isoformat() if report_type == "Custom Period" else "N/A",
                    "end": end_date.isoformat() if report_type == "Custom Period" else "N/A"
                },
                "summary": {
                    "total_trades": len(df),
                    "winning_trades": len(df[df['profit'] > 0]),
                    "losing_trades": len(df[df['profit'] < 0]),
                    "total_profit": float(df['profit'].sum()),
                    "win_rate": float(len(df[df['profit'] > 0]) / len(df) * 100) if len(df) > 0 else 0,
                    "average_profit": float(df['profit'].mean()),
                    "max_profit": float(df['profit'].max()),
                    "max_loss": float(df['profit'].min())
                }
            }
            
            # Display report
            st.subheader("📊 Generated Report")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trades", report_data["summary"]["total_trades"])
                st.metric("Win Rate", f"{report_data['summary']['win_rate']:.1f}%")
            
            with col2:
                st.metric("Total P&L", f"${report_data['summary']['total_profit']:+.2f}")
                st.metric("Avg P&L", f"${report_data['summary']['average_profit']:+.2f}")
            
            with col3:
                st.metric("Best Trade", f"${report_data['summary']['max_profit']:+.2f}")
                st.metric("Worst Trade", f"${report_data['summary']['max_loss']:+.2f}")
            
            # Download report
            if export_format == "JSON":
                st.download_button(
                    "📥 Download Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.warning("⚠️ No trading data available for report generation.")

# --- Footer ---
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🤖 AI Forex Trading Bot v2.0**")

with col2:
    st.markdown("**📊 Real-time Analytics**")

with col3:
    if st.button("📖 Documentation"):
        st.markdown("📚 [View Documentation](https://github.com/chris22622/ai-forex-trading-bot)")

# Auto-refresh for live data
if st.session_state.running:
    time.sleep(1)
    st.rerun()
