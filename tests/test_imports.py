"""
Test file to verify all imports work correctly.
This ensures the basic structure and dependencies are functioning.
"""

import os
import sys

import pytest

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_core_imports():
    """Test that core dependencies can be imported."""
    try:
        import joblib
        import numpy
        import pandas
        import sklearn

        assert True
    except ImportError as e:
        pytest.fail(f"Core dependency import failed: {e}")


def test_trading_imports():
    """Test that trading-related dependencies can be imported."""
    try:
        import MetaTrader5
        import ta

        assert True
    except ImportError as e:
        pytest.fail(f"Trading dependency import failed: {e}")


def test_communication_imports():
    """Test that communication dependencies can be imported."""
    try:
        import requests
        import telegram
        import websockets

        assert True
    except ImportError as e:
        pytest.fail(f"Communication dependency import failed: {e}")


def test_project_imports():
    """Test that project modules can be imported."""
    try:
        import config
        import main

        assert True
    except ImportError as e:
        pytest.fail(f"Project module import failed: {e}")


def test_config_structure():
    """Test that config has required attributes."""
    import config

    # Check that essential config attributes exist
    required_attrs = [
        "TRADE_AMOUNT",
        "RISK_PERCENTAGE",
        "MT5_LOGIN",
        "MT5_PASSWORD",
        "MT5_SERVER",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
    ]

    for attr in required_attrs:
        assert hasattr(config, attr), f"Config missing required attribute: {attr}"


if __name__ == "__main__":
    pytest.main([__file__])
