#!/usr/bin/env python3
"""
Enhanced Dashboard Launcher
Quick launcher for the advanced Streamlit dashboard
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def check_streamlit():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit if not present"""
    print("📦 Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Streamlit")
        return False

def main():
    """Launch the enhanced dashboard"""
    print("🚀 Enhanced Dashboard Launcher")
    print("=" * 40)
    
    # Check if streamlit is installed
    if not check_streamlit():
        print("⚠️ Streamlit not found. Installing...")
        if not install_streamlit():
            print("❌ Cannot proceed without Streamlit")
            sys.exit(1)
    
    # Get dashboard path
    dashboard_path = Path(__file__).parent / "ui" / "enhanced_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at: {dashboard_path}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    
    print("🎛️ Launching Enhanced Dashboard...")
    print("🌐 Dashboard will open in your default browser")
    print("📝 Check the terminal for the local URL if it doesn't open automatically")
    print()
    print("⚠️ To stop the dashboard, press Ctrl+C in this terminal")
    print()
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.headless", "false",
            "--server.runOnSave", "true"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\n🔧 Try running manually:")
        print(f"   streamlit run {dashboard_path}")

if __name__ == "__main__":
    main()
