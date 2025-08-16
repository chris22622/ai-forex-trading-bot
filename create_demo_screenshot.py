import time
import asyncio
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Create a demo chart that looks like the live dashboard
def create_demo_chart():
    # Generate sample trading data
    dates = pd.date_range(start='2025-08-15 10:00', periods=100, freq='1min')
    prices = []
    price = 1.0850  # Starting EUR/USD price
    
    for i in range(100):
        # Simulate price movement
        change = (hash(str(i)) % 21 - 10) / 10000  # Random-ish movement
        price += change
        prices.append(round(price, 5))
    
    df = pd.DataFrame({
        'time': dates,
        'price': prices
    })
    
    # Create professional trading chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('EUR/USD Live Price', 'Trading Signals'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['price'],
            mode='lines',
            name='EUR/USD',
            line=dict(color='#00ff88', width=2)
        ),
        row=1, col=1
    )
    
    # Add some buy/sell signals
    buy_signals = df.iloc[::15]  # Every 15th point
    sell_signals = df.iloc[7::15]  # Offset sell signals
    
    fig.add_trace(
        go.Scatter(
            x=buy_signals['time'],
            y=buy_signals['price'],
            mode='markers',
            name='BUY Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=sell_signals['time'],
            y=sell_signals['price'],
            mode='markers',
            name='SELL Signal',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ),
        row=1, col=1
    )
    
    # Volume/activity bars
    fig.add_trace(
        go.Bar(
            x=df['time'][::5],
            y=[abs(hash(str(i)) % 100) for i in range(0, len(df), 5)],
            name='Activity',
            marker_color='rgba(55, 128, 191, 0.7)'
        ),
        row=2, col=1
    )
    
    # Styling
    fig.update_layout(
        title={
            'text': '🤖 AI Forex Trading Bot - Live Dashboard',
            'x': 0.5,
            'font': {'size': 20, 'color': '#2E8B57'}
        },
        template='plotly_dark',
        showlegend=True,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# Generate the demo chart
if __name__ == "__main__":
    print("🎨 Creating demo dashboard screenshot...")
    fig = create_demo_chart()
    
    # Save as high-quality image
    pio.write_image(fig, "docs/dashboard_demo.png", 
                    width=1200, height=600, scale=2)
    print("✅ Demo chart saved as docs/dashboard_demo.png")
    
    # Also save as HTML for interactive demo
    pio.write_html(fig, "docs/dashboard_demo.html")
    print("✅ Interactive demo saved as docs/dashboard_demo.html")
