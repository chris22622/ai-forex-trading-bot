"""
Automated Demo GIF Creator
Creates a simulated animated demo of the trading bot
"""
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta
import os

def create_animated_demo():
    """Create an animated chart that simulates the live demo"""
    
    print("🎬 Creating animated demo simulation...")
    
    # Generate time series data
    start_time = datetime.now()
    times = [start_time + timedelta(seconds=i*2) for i in range(50)]
    
    # Simulate realistic price movement
    base_price = 1.0850
    prices = []
    for i in range(50):
        # Create realistic price movement with some trend
        noise = (hash(f"price_{i}") % 201 - 100) / 100000  # Small random movement
        trend = 0.0001 if i > 20 else -0.0001  # Slight trend change
        base_price += trend + noise
        prices.append(round(base_price, 5))
    
    # Create frames for animation
    frames = []
    for i in range(10, 50, 5):  # Every 5th frame for smooth animation
        frame_data = {
            'time': times[:i],
            'price': prices[:i]
        }
        
        # Create frame
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=frame_data['time'],
            y=frame_data['price'],
            mode='lines+markers',
            name='EUR/USD Live',
            line=dict(color='#00ff88', width=3),
            marker=dict(size=4, color='#00ff88')
        ))
        
        # Add buy/sell signals occasionally
        if i > 20 and i % 10 == 0:
            signal_time = frame_data['time'][-1]
            signal_price = frame_data['price'][-1]
            signal_type = 'BUY' if i % 20 == 0 else 'SELL'
            color = 'green' if signal_type == 'BUY' else 'red'
            symbol = 'triangle-up' if signal_type == 'BUY' else 'triangle-down'
            
            fig.add_trace(go.Scatter(
                x=[signal_time],
                y=[signal_price],
                mode='markers',
                name=f'{signal_type} Signal',
                marker=dict(size=15, color=color, symbol=symbol),
                showlegend=False
            ))
        
        # Style the chart
        fig.update_layout(
            title={
                'text': f'🤖 AI Forex Trading Bot - Live Demo (Frame {i//5})',
                'x': 0.5,
                'font': {'size': 18, 'color': '#2E8B57'}
            },
            xaxis_title="Time",
            yaxis_title="EUR/USD Price",
            template='plotly_dark',
            showlegend=True,
            height=400,
            width=800,
            margin=dict(l=50, r=50, t=60, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.8)'
        )
        
        # Save frame
        frame_filename = f"docs/frame_{i:02d}.png"
        pio.write_image(fig, frame_filename, width=800, height=400, scale=2)
        frames.append(frame_filename)
        print(f"  📸 Created frame {i//5}/8")
    
    return frames

def create_demo_mockup():
    """Create a static mockup that looks like the actual dashboard"""
    
    # Create a more realistic demo chart
    fig = go.Figure()
    
    # Generate realistic trading data
    times = pd.date_range(start='2025-08-15 14:30', periods=60, freq='30s')
    base_price = 1.0852
    prices = []
    
    for i in range(60):
        # Realistic intraday movement
        market_noise = (hash(f"tick_{i}") % 21 - 10) / 100000
        momentum = 0.0002 * (i % 10 - 5) / 10  # Some momentum patterns
        base_price += market_noise + momentum
        prices.append(round(base_price, 5))
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=times,
        y=prices,
        mode='lines',
        name='EUR/USD Live',
        line=dict(color='#00ff88', width=2),
        hovertemplate='Time: %{x}<br>Price: $%{y:.5f}<extra></extra>'
    ))
    
    # Add trading signals
    buy_times = times[::15]  # Every 15th point
    buy_prices = [prices[i] for i in range(0, len(prices), 15)]
    
    sell_times = times[7::15]  # Offset sell signals
    sell_prices = [prices[i] for i in range(7, len(prices), 15) if i < len(prices)]
    
    fig.add_trace(go.Scatter(
        x=buy_times,
        y=buy_prices,
        mode='markers',
        name='AI BUY Signal',
        marker=dict(color='#00ff00', size=12, symbol='triangle-up', line=dict(color='white', width=1)),
        hovertemplate='BUY Signal<br>Price: $%{y:.5f}<br>Confidence: 78%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=sell_times[:len(sell_prices)],
        y=sell_prices,
        mode='markers',
        name='AI SELL Signal',
        marker=dict(color='#ff4444', size=12, symbol='triangle-down', line=dict(color='white', width=1)),
        hovertemplate='SELL Signal<br>Price: $%{y:.5f}<br>Confidence: 72%<extra></extra>'
    ))
    
    # Style for professional look
    fig.update_layout(
        title={
            'text': '🤖 AI Forex Trading Bot - Demo Mode Active',
            'x': 0.5,
            'font': {'size': 20, 'color': '#2E8B57', 'family': 'Arial, sans-serif'}
        },
        xaxis=dict(
            title="Time (Live Feed)",
            gridcolor='rgba(128,128,128,0.3)',
            showgrid=True,
            tickformat='%H:%M:%S'
        ),
        yaxis=dict(
            title="EUR/USD Price",
            gridcolor='rgba(128,128,128,0.3)',
            showgrid=True,
            tickformat='.5f'
        ),
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        width=1000,
        margin=dict(l=60, r=60, t=80, b=60),
        paper_bgcolor='rgba(17,17,17,1)',
        plot_bgcolor='rgba(0,0,0,0.8)',
        hovermode='x unified'
    )
    
    # Add annotations for status
    fig.add_annotation(
        x=times[-1],
        y=max(prices),
        text="✅ DEMO MODE<br>📊 Live Chart<br>🤖 AI Active",
        showarrow=True,
        arrowhead=2,
        arrowcolor="green",
        bgcolor="rgba(0,255,0,0.1)",
        bordercolor="green",
        borderwidth=1,
        font=dict(color="lightgreen", size=10)
    )
    
    return fig

if __name__ == "__main__":
    print("🎨 Creating professional demo assets...")
    
    # Create the main demo mockup
    demo_fig = create_demo_mockup()
    
    # Save as high-quality PNG for README
    pio.write_image(demo_fig, "docs/demo_dashboard.png", 
                    width=1000, height=500, scale=2)
    print("✅ Demo dashboard saved as docs/demo_dashboard.png")
    
    # Save as interactive HTML
    pio.write_html(demo_fig, "docs/demo_interactive.html",
                   include_plotlyjs='cdn')
    print("✅ Interactive demo saved as docs/demo_interactive.html")
    
    # Create animated frames (optional)
    print("\n🎬 Creating animation frames...")
    frames = create_animated_demo()
    print(f"✅ Created {len(frames)} animation frames")
    
    print("\n🎯 Demo assets ready!")
    print("   📸 Static demo: docs/demo_dashboard.png") 
    print("   🌐 Interactive: docs/demo_interactive.html")
    print("   🎬 Animation frames: docs/frame_*.png")
    print("\nNext: You can manually combine frames into GIF or use the static PNG!")
