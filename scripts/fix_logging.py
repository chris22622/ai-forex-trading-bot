"""
Quick fix script for Unicode logging issues
"""

import re


def fix_logging_file(filepath):
    """Fix logging issues in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace self.logger with logger
        content = re.sub(r'self\.logger\.(info|error|warning|debug|critical)', r'logger.\1', content)

        # Remove the self.logger assignment line
        content = re.sub(r'\s*self\.logger = self\._setup_logger\(\)\s*\n', '', content)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Fixed: {filepath}")

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")

if __name__ == "__main__":
    fix_logging_file("ai_integration_system.py")
    print("Logging fixes applied")
