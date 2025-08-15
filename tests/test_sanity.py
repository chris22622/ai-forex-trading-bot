"""
Basic sanity tests for the AI Forex Trading Bot.
These tests verify basic functionality without requiring live connections.
"""

import os
import sys

import pytest

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_python_version():
    """Ensure we're running on a supported Python version."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"


def test_config_exists():
    """Test that config file exists and has basic structure."""
    try:
        import config

        # Basic smoke test - config should load without errors
        assert hasattr(config, "__name__")
    except ImportError:
        pytest.fail("Config module could not be imported")


def test_main_exists():
    """Test that main module exists."""
    try:
        import main

        # Basic smoke test - main should load without errors
        assert hasattr(main, "__name__")
    except ImportError:
        pytest.fail("Main module could not be imported")


def test_environment_variables():
    """Test that environment variables are properly handled."""
    # Test that os.getenv works as expected
    import os

    # This should not raise an exception
    test_var = os.getenv("NON_EXISTENT_VAR", "default_value")
    assert test_var == "default_value"


def test_basic_math():
    """Test basic mathematical operations for trading calculations."""
    # Test percentage calculations
    assert 0.01 * 1000 == 10  # 1% of 1000
    assert 0.02 * 50000 == 1000  # 2% of 50000

    # Test risk calculations
    account_balance = 10000
    risk_percentage = 0.02
    max_risk = account_balance * risk_percentage
    assert max_risk == 200


def test_directory_structure():
    """Test that the project has the expected directory structure."""
    project_root = os.path.join(os.path.dirname(__file__), "..")

    # Check that key directories exist
    expected_dirs = ["src", "tests", "docs", ".github"]
    for dir_name in expected_dirs:
        dir_path = os.path.join(project_root, dir_name)
        assert os.path.exists(dir_path), f"Missing directory: {dir_name}"


if __name__ == "__main__":
    pytest.main([__file__])
