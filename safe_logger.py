"""
Safe logging utility for Windows compatibility
Removes emoji characters that cause UnicodeEncodeError on Windows cp1252 encoding
"""

import re
import logging
from typing import Any

def safe_log_message(message: str) -> str:
    """
    Remove emoji and other Unicode characters that cause issues on Windows
    """
    # Remove emoji and other problematic Unicode characters
    # This regex removes most emoji characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "]+", 
        flags=re.UNICODE
    )
    
    # Replace emojis with text equivalents
    replacements = {
        '🚀': '[START]',
        '🔄': '[LOADING]',
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '🔧': '[SETUP]',
        '💰': '[MONEY]',
        '📊': '[CHART]',
        '🎯': '[TARGET]',
        '🧠': '[AI]',
        '📱': '[MOBILE]',
        '🔐': '[AUTH]',
        '🌐': '[NETWORK]',
        '🔥': '[FIRE]',
        '🏢': '[BUILDING]',
        '💡': '[IDEA]',
        '🇩🇪': '[DE]',
        '🇳🇱': '[NL]',
        '🇬🇧': '[GB]',
        '🇸🇬': '[SG]',
        '🇺🇸': '[US]',
        '💾': '[SAVE]',
        '⏹️': '[STOP]',
        '🧪': '[TEST]',
        '📝': '[NOTE]',
        '🎮': '[GAME]',
        '🚨': '[ALARM]',
        '⏰': '[TIME]',
        '🔌': '[PLUG]',
        '📈': '[UP]',
        '📉': '[DOWN]',
        '💎': '[DIAMOND]',
        '🤖': '[BOT]',
        '🎨': '[ART]',
        '📸': '[CAMERA]',
        '🎵': '[MUSIC]',
        '🔔': '[BELL]',
        '🚪': '[DOOR]',
        '🏃': '[RUN]',
        '💻': '[COMPUTER]',
        '📺': '[TV]',
        '🎬': '[MOVIE]',
        '🎪': '[CIRCUS]',
        '🎭': '[THEATER]',
        '🎨': '[PALETTE]',
        '🎸': '[GUITAR]',
        '🥳': '[PARTY]',
        '😎': '[COOL]',
        '😊': '[HAPPY]',
        '😢': '[SAD]',
        '😡': '[ANGRY]',
        '🤔': '[THINKING]',
        '👍': '[THUMBS_UP]',
        '👎': '[THUMBS_DOWN]',
        '👌': '[OK]',
        '✨': '[SPARKLE]',
        '⭐': '[STAR]',
        '🌟': '[STAR]',
        '🔴': '[RED]',
        '🟢': '[GREEN]',
        '🟡': '[YELLOW]',
        '🔵': '[BLUE]',
        '⚫': '[BLACK]',
        '⚪': '[WHITE]',
        '🟠': '[ORANGE]',
        '🟣': '[PURPLE]',
        '🟤': '[BROWN]',
    }
    
    # Replace known emojis first
    safe_message = message
    for emoji, replacement in replacements.items():
        safe_message = safe_message.replace(emoji, replacement)
    
    # Remove any remaining emoji/unicode characters
    safe_message = emoji_pattern.sub('', safe_message)
    
    # Clean up extra spaces
    safe_message = ' '.join(safe_message.split())
    
    return safe_message

class SafeLogger:
    """
    Safe logger wrapper that ensures Windows compatibility
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        safe_message = safe_log_message(str(message))
        self.logger.info(safe_message, *args, **kwargs)
    
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        safe_message = safe_log_message(str(message))
        self.logger.error(safe_message, *args, **kwargs)
    
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        safe_message = safe_log_message(str(message))
        self.logger.warning(safe_message, *args, **kwargs)
    
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        safe_message = safe_log_message(str(message))
        self.logger.debug(safe_message, *args, **kwargs)
    
    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        safe_message = safe_log_message(str(message))
        self.logger.critical(safe_message, *args, **kwargs)

def get_safe_logger(name: str) -> SafeLogger:
    """
    Get a safe logger instance
    """
    logger = logging.getLogger(name)
    return SafeLogger(logger)
