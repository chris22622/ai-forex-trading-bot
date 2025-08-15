#!/usr/bin/env python3
"""
Deriv Trading Bot Setup Validator
Checks all requirements for the Godlike Continuous Indices Trading Bot
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.9+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.9+")
        return False

def check_required_packages():
    """Check if all required packages are installed"""
    required_packages = [
        'websockets',
        'pandas', 
        'numpy',
        'ta',
        'asyncio',
        'requests',
        'pytz',
        'telegram'
    ]
    
    missing_packages: list[str] = []
    
    for package in required_packages:
        try:
            if package == 'telegram':
                importlib.import_module('telegram')
            else:
                importlib.import_module(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    return missing_packages

def check_config_files():
    """Check if config files exist"""
    config_files = ['config.py']
    missing_files: list[str] = []
    
    for file in config_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    return missing_files

def create_directories():
    """Create necessary directories"""
    directories = ['logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory '{directory}' - Ready")

def install_missing_packages(missing_packages: list[str]) -> bool:
    """Install missing packages"""
    if not missing_packages:
        return True
    
    print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
    
    # Map package names to pip install names
    pip_packages: list[str] = []
    for package in missing_packages:
        if package == 'telegram':
            pip_packages.append('python-telegram-bot')
        else:
            pip_packages.append(package)
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + pip_packages)
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Deriv Trading Bot Setup Validator")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup validation failed - Update Python to 3.9+")
        return False
    
    # Check packages
    missing_packages = check_required_packages()
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Found {len(missing_packages)} missing packages")
        if not install_missing_packages(missing_packages):
            print("\n❌ Setup validation failed - Could not install required packages")
            return False
    
    # Create directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    # Check config files
    print("\n🔧 Checking configuration files...")
    missing_files = check_config_files()
    
    if missing_files:
        print(f"\n⚠️  Missing config files: {', '.join(missing_files)}")
        print("These will be created automatically when you run the bot setup.")
    
    print("\n" + "=" * 50)
    print("✅ Setup validation completed successfully!")
    print("🎯 Ready to create the Godlike Deriv Trading Bot!")
    
    return True

if __name__ == "__main__":
    main()