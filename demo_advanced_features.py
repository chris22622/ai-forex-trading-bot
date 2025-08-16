#!/usr/bin/env python3
"""
Advanced Features Demo Script
Demonstrates the new enhanced capabilities of the AI Forex Trading Bot
"""

import sys
from pathlib import Path
import time
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from advanced_config import ConfigManager, get_config
    from performance_analytics import PerformanceAnalyzer, TradeRecord
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def demo_advanced_config():
    """Demonstrate advanced configuration management"""
    print("🔧 Advanced Configuration Demo")
    print("=" * 50)
    
    # Get current config
    config = get_config()
    print(f"📊 Current confidence threshold: {config.confidence_threshold}")
    print(f"💰 Max daily loss: ${config.max_daily_loss}")
    print(f"📈 Position size: {config.position_size_percent}%")
    
    # Test validation
    errors = config.validate()
    if errors:
        print(f"❌ Validation errors: {errors}")
    else:
        print("✅ Configuration is valid")
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Test update
    print("\n🔄 Testing configuration update...")
    success = config_manager.update_config({
        'confidence_threshold': 0.75,
        'max_concurrent_trades': 3
    })
    
    if success:
        print("✅ Configuration updated successfully")
        updated_config = get_config()
        print(f"📊 New confidence threshold: {updated_config.confidence_threshold}")
    else:
        print("❌ Configuration update failed")
    
    print()

def demo_performance_analytics():
    """Demonstrate performance analytics system"""
    print("📊 Performance Analytics Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = PerformanceAnalyzer()
    
    # Create sample trades
    sample_trades = [
        TradeRecord(
            trade_id="DEMO_001",
            symbol="EURUSD",
            action="BUY",
            entry_price=1.1250,
            exit_price=1.1275,
            volume=0.1,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            profit_loss=25.0,
            confidence=0.75,
            strategy="AI_ENSEMBLE",
            duration_minutes=30,
            max_drawdown=5.0,
            max_profit=30.0,
            exit_reason="TAKE_PROFIT",
            market_conditions={"volatility": 0.8, "trend": 1.2}
        ),
        TradeRecord(
            trade_id="DEMO_002",
            symbol="GBPUSD",
            action="SELL",
            entry_price=1.3000,
            exit_price=1.2980,
            volume=0.1,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            profit_loss=-20.0,
            confidence=0.65,
            strategy="AI_ENSEMBLE",
            duration_minutes=45,
            max_drawdown=25.0,
            max_profit=5.0,
            exit_reason="STOP_LOSS",
            market_conditions={"volatility": 1.2, "trend": -0.5}
        ),
        TradeRecord(
            trade_id="DEMO_003",
            symbol="USDJPY",
            action="BUY",
            entry_price=110.50,
            exit_price=110.75,
            volume=0.1,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            profit_loss=22.7,
            confidence=0.80,
            strategy="AI_ENSEMBLE",
            duration_minutes=25,
            max_drawdown=3.0,
            max_profit=25.0,
            exit_reason="TAKE_PROFIT",
            market_conditions={"volatility": 0.6, "trend": 1.5}
        )
    ]
    
    # Record trades
    print("📝 Recording sample trades...")
    for trade in sample_trades:
        analyzer.record_trade(trade)
    
    # Calculate metrics
    print("📈 Calculating performance metrics...")
    metrics = analyzer.calculate_metrics()
    
    print(f"📊 Total trades: {metrics.total_trades}")
    print(f"🎯 Win rate: {metrics.win_rate:.1%}")
    print(f"💰 Net profit: ${metrics.net_profit:.2f}")
    print(f"📉 Max drawdown: ${metrics.max_drawdown:.2f}")
    print(f"⚡ Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    print(f"🛡️ Sortino ratio: {metrics.sortino_ratio:.2f}")
    
    # Generate report
    print("\n📋 Generating performance report...")
    report = analyzer.generate_performance_report()
    
    print("✅ Report generated with the following sections:")
    for section in report.keys():
        print(f"  • {section}")
    
    print()

def demo_enhanced_dashboard_info():
    """Show information about the enhanced dashboard"""
    print("🎛️ Enhanced Dashboard Features")
    print("=" * 50)
    
    dashboard_features = [
        "🎮 Trading Control - Start/stop trading with real-time controls",
        "📊 Performance Analytics - Comprehensive trading metrics and charts", 
        "⚙️ Configuration - Dynamic parameter adjustment with validation",
        "📈 Live Charts - Real-time price action and trade visualization",
        "📋 Reports - Automated report generation with export options"
    ]
    
    print("The enhanced Streamlit dashboard includes:")
    for feature in dashboard_features:
        print(f"  • {feature}")
    
    print("\n🚀 To launch the enhanced dashboard:")
    print("   streamlit run ui/enhanced_dashboard.py")
    
    print("\n🎯 Features include:")
    print("   • Multi-page interface with navigation")
    print("   • Real-time trade monitoring") 
    print("   • Interactive Plotly charts")
    print("   • Advanced performance analytics")
    print("   • Dynamic configuration management")
    print("   • Automated report generation")
    
    print()

def main():
    """Run the complete advanced features demo"""
    print("🚀 AI Forex Trading Bot - Advanced Features Demo")
    print("=" * 60)
    print(f"🕐 Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Demo 1: Advanced Configuration
        demo_advanced_config()
        
        # Demo 2: Performance Analytics
        demo_performance_analytics()
        
        # Demo 3: Enhanced Dashboard Info
        demo_enhanced_dashboard_info()
        
        print("🎉 Advanced Features Demo Complete!")
        print("=" * 60)
        print("✅ All systems operational and ready for advanced trading")
        print()
        print("🔗 Next Steps:")
        print("   1. Launch enhanced dashboard: streamlit run ui/enhanced_dashboard.py")
        print("   2. Configure your trading parameters")
        print("   3. Start demo trading to test features")
        print("   4. Monitor performance with advanced analytics")
        print("   5. Generate reports and optimize your strategy")
        print()
        print("📚 For more information, check the updated README.md")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
