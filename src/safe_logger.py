"""
Safe logging utility for Windows compatibility
Removes emoji characters that cause UnicodeEncodeError on Windows cp1252 encoding
"""

import logging
import re
from typing import Any


def safe_log_message(message: str) -> str:
    """
    Remove emoji and other Unicode characters that cause issues on Windows
    """
    # Remove emoji and other problematic Unicode characters
    # This regex removes most emoji characters
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"  # dingbats
        "\U000024c2-\U0001f251"  # enclosed characters
        "\U0001f900-\U0001f9ff"  # supplemental symbols
        "\U0001fa70-\U0001faff"  # symbols and pictographs extended-A
        "]+",
        flags=re.UNICODE,
    )

    # Replace emojis with text equivalents
    replacements = {
        "🚀": "[START]",
        "🔄": "[LOADING]",
        "✅": "[SUCCESS]",
        "❌": "[ERROR]",
        "⚠️": "[WARNING]",
        "🔧": "[SETUP]",
        "💰": "[MONEY]",
        "📊": "[CHART]",
        "🎯": "[TARGET]",
        "🧠": "[AI]",
        "📱": "[MOBILE]",
        "🔐": "[AUTH]",
        "🌐": "[NETWORK]",
        "🔥": "[FIRE]",
        "🏢": "[BUILDING]",
        "💡": "[IDEA]",
        "🇩🇪": "[DE]",
        "🇳🇱": "[NL]",
        "🇬🇧": "[GB]",
        "🇸🇬": "[SG]",
        "🇺🇸": "[US]",
        "💾": "[SAVE]",
        "⏹️": "[STOP]",
        "🧪": "[TEST]",
        "📝": "[NOTE]",
        "🎮": "[GAME]",
        "🚨": "[ALARM]",
        "⏰": "[TIME]",
        "🔌": "[PLUG]",
        "📈": "[UP]",
        "📉": "[DOWN]",
        "💎": "[DIAMOND]",
        "🤖": "[BOT]",
        "🎨": "[ART]",
        "📸": "[CAMERA]",
        "🎵": "[MUSIC]",
        "🔔": "[BELL]",
        "🚪": "[DOOR]",
        "🏃": "[RUN]",
        "💻": "[COMPUTER]",
        "📺": "[TV]",
        "🎬": "[MOVIE]",
        "🎪": "[CIRCUS]",
        "🎭": "[THEATER]",
        "": "[GUITAR]",
        "🥳": "[PARTY]",
        "😎": "[COOL]",
        "😊": "[HAPPY]",
        "😢": "[SAD]",
        "😡": "[ANGRY]",
        "🤔": "[THINKING]",
        "👍": "[THUMBS_UP]",
        "👎": "[THUMBS_DOWN]",
        "👌": "[OK]",
        "✨": "[SPARKLE]",
        "⭐": "[STAR]",
        "🌟": "[STAR]",
        "🔴": "[RED]",
        "🟢": "[GREEN]",
        "🟡": "[YELLOW]",
        "🔵": "[BLUE]",
        "⚫": "[BLACK]",
        "⚪": "[WHITE]",
        "🟠": "[ORANGE]",
        "🟣": "[PURPLE]",
        "🟤": "[BROWN]",
    }

    # Replace known emojis first
    safe_message = message
    for emoji, replacement in replacements.items():
        safe_message = safe_message.replace(emoji, replacement)

    # Remove any remaining emoji/unicode characters
    safe_message = emoji_pattern.sub("", safe_message)

    # Clean up extra spaces
    safe_message = " ".join(safe_message.split())

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
