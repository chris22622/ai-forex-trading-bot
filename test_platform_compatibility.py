#!/usr/bin/env python3
"""
Platform compatibility test for CI/CD pipeline
Tests that the bot can import safely on any platform
"""

import os
import sys

def test_mt5_import():
    """Test safe MT5 import"""
    try:
        import MetaTrader5 as mt5  # type: ignore
        print("✅ MetaTrader5 import successful")
        return True
    except Exception as e:
        print(f"⚠️ MetaTrader5 import failed (expected on Linux): {e}")
        return False

def test_main_import():
    """Test that main.py imports safely"""
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        # This should work on any platform now
        import main  # type: ignore
        print("✅ main.py import successful")
        return True
    except Exception as e:
        print(f"❌ main.py import failed: {e}")
        return False

def test_mt5_integration_import():
    """Test that mt5_integration.py imports safely"""
    try:
        # Add src to path  
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        # This should work on any platform now
        import mt5_integration  # type: ignore
        print("✅ mt5_integration.py import successful")
        return True
    except Exception as e:
        print(f"❌ mt5_integration.py import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing platform compatibility...")
    print(f"Platform: {os.name}")
    
    mt5_ok = test_mt5_import()
    main_ok = test_main_import()
    mt5_int_ok = test_mt5_integration_import()
    
    if main_ok and mt5_int_ok:
        print("✅ Platform compatibility test PASSED")
        exit(0)
    else:
        print("❌ Platform compatibility test FAILED")
        exit(1)
