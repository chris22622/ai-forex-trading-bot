"""
Advanced Configuration Management System
Provides dynamic configuration with validation and hot-reloading
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Advanced trading configuration with validation"""
    
    # Risk Management
    max_daily_loss: float = 100.0
    max_concurrent_trades: int = 5
    position_size_percent: float = 2.0
    stop_loss_percent: float = 1.5
    take_profit_percent: float = 3.0
    
    # Trading Parameters
    confidence_threshold: float = 0.65
    min_spread_points: float = 0.5
    max_spread_points: float = 5.0
    trade_timeout_minutes: int = 30
    
    # AI Model Settings
    ai_model_retrain_frequency: int = 100  # trades
    ensemble_weight_adjustment: bool = True
    confidence_calibration: bool = True
    
    # Risk Scaling
    adaptive_position_sizing: bool = True
    volatility_adjustment: bool = True
    drawdown_protection: bool = True
    
    # Notification Settings
    telegram_enabled: bool = True
    notification_levels: List[str] = field(default_factory=lambda: ["ERROR", "TRADE", "PROFIT"])
    
    # Advanced Features
    paper_trading_mode: bool = False
    backtesting_enabled: bool = False
    performance_analytics: bool = True
    
    def validate(self) -> List[str]:
        """Validate configuration parameters"""
        errors = []
        
        if self.max_daily_loss <= 0:
            errors.append("max_daily_loss must be positive")
        
        if self.max_concurrent_trades < 1 or self.max_concurrent_trades > 20:
            errors.append("max_concurrent_trades must be between 1 and 20")
        
        if not 0.1 <= self.position_size_percent <= 10.0:
            errors.append("position_size_percent must be between 0.1 and 10.0")
        
        if not 0.3 <= self.confidence_threshold <= 0.95:
            errors.append("confidence_threshold must be between 0.3 and 0.95")
        
        if self.stop_loss_percent <= 0 or self.take_profit_percent <= 0:
            errors.append("stop_loss_percent and take_profit_percent must be positive")
        
        if self.take_profit_percent <= self.stop_loss_percent:
            errors.append("take_profit_percent should be greater than stop_loss_percent")
        
        return errors


class ConfigManager:
    """Advanced configuration manager with hot-reloading and validation"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/advanced_config.json")
        self.config = TradingConfig()
        self.watchers = []
        self._last_modified = 0
        
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config or create default
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                # Update config with loaded data
                for key, value in data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                # Validate loaded config
                errors = self.config.validate()
                if errors:
                    logger.warning(f"Configuration validation errors: {errors}")
                    return False
                
                self._last_modified = self.config_path.stat().st_mtime
                logger.info("✅ Configuration loaded successfully")
                return True
            else:
                # Create default config file
                self.save_config()
                logger.info("📝 Created default configuration file")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_dict = {
                key: getattr(self.config, key)
                for key in self.config.__dataclass_fields__.keys()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
            
            self._last_modified = self.config_path.stat().st_mtime
            logger.info("💾 Configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            # Apply updates
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            # Validate updated config
            errors = self.config.validate()
            if errors:
                logger.error(f"Configuration validation failed: {errors}")
                return False
            
            # Save updated config
            return self.save_config()
            
        except Exception as e:
            logger.error(f"❌ Failed to update configuration: {e}")
            return False
    
    def check_for_updates(self) -> bool:
        """Check if configuration file has been modified"""
        try:
            if self.config_path.exists():
                current_mtime = self.config_path.stat().st_mtime
                if current_mtime > self._last_modified:
                    logger.info("🔄 Configuration file modified, reloading...")
                    return self.load_config()
            return False
        except Exception as e:
            logger.error(f"❌ Error checking for config updates: {e}")
            return False
    
    def get_config(self) -> TradingConfig:
        """Get current configuration (with auto-reload check)"""
        self.check_for_updates()
        return self.config
    
    def get_risk_parameters(self) -> Dict[str, float]:
        """Get risk management parameters"""
        return {
            'max_daily_loss': self.config.max_daily_loss,
            'max_concurrent_trades': self.config.max_concurrent_trades,
            'position_size_percent': self.config.position_size_percent,
            'stop_loss_percent': self.config.stop_loss_percent,
            'take_profit_percent': self.config.take_profit_percent,
        }
    
    def get_trading_parameters(self) -> Dict[str, Union[float, int]]:
        """Get trading parameters"""
        return {
            'confidence_threshold': self.config.confidence_threshold,
            'min_spread_points': self.config.min_spread_points,
            'max_spread_points': self.config.max_spread_points,
            'trade_timeout_minutes': self.config.trade_timeout_minutes,
        }
    
    def export_config(self, export_path: Path) -> bool:
        """Export current configuration to specified path"""
        try:
            config_dict = {
                key: getattr(self.config, key)
                for key in self.config.__dataclass_fields__.keys()
            }
            
            with open(export_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
            
            logger.info(f"📤 Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export configuration: {e}")
            return False


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> TradingConfig:
    """Get global configuration instance"""
    return config_manager.get_config()


def update_config(updates: Dict[str, Any]) -> bool:
    """Update global configuration"""
    return config_manager.update_config(updates)


def get_risk_parameters() -> Dict[str, float]:
    """Get current risk management parameters"""
    return config_manager.get_risk_parameters()


def get_trading_parameters() -> Dict[str, Union[float, int]]:
    """Get current trading parameters"""
    return config_manager.get_trading_parameters()


if __name__ == "__main__":
    # Demo the configuration system
    print("🔧 Advanced Configuration System Demo")
    
    config = get_config()
    print(f"📊 Current config: {config}")
    
    # Test validation
    errors = config.validate()
    if errors:
        print(f"❌ Validation errors: {errors}")
    else:
        print("✅ Configuration is valid")
    
    # Test update
    success = update_config({
        'confidence_threshold': 0.75,
        'max_concurrent_trades': 3
    })
    print(f"📝 Config update: {'✅ Success' if success else '❌ Failed'}")
    
    print("🎯 Configuration system ready!")
