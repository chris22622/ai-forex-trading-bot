# Scripts Directory

This directory contains utility scripts, test files, and development tools that are not part of the main trading bot application.

## Script Categories

### 🔧 Development & Testing
- `test_*.py` - Various test scripts for different components
- `debug_*.py` - Debugging utilities
- `check_*.py` - Status checking scripts

### 🚀 Launch Scripts
- `*.bat` / `*.ps1` - Windows batch/PowerShell launch scripts
- `launch_*.py` - Python launcher utilities

### 🎮 Demo & Examples
- `demo_*.py` - Demo trading scripts
- `*_demo.py` - Example implementations

### 🔨 Configuration & Fixes
- `config_*.py` - Configuration backup and utility files
- `fix_*.py` - Bug fix and repair scripts
- `*_backup*.py` - Backup versions of files

### 📊 Analysis & Monitoring
- `*_status.py` - Status monitoring scripts
- `*_check.py` - Health check utilities

## Usage

These scripts are primarily for development, testing, and troubleshooting. The main trading bot application is in the `src/` directory.

To run any script:
```bash
cd scripts
python script_name.py
```

**Note**: Many of these scripts may require configuration or may be specific to certain development phases. Use with caution in production environments.
