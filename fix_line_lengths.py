#!/usr/bin/env python3
"""
Automatic E501 Line Length Fixer
Fixes all lines over 100 characters in Python files
"""

import os
import re
import sys
from pathlib import Path

def fix_long_lines_in_file(file_path: str) -> bool:
    """Fix E501 errors in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        
        for i, line in enumerate(lines):
            line_no = i + 1
            if len(line.rstrip()) > 100:
                fixed_line = fix_long_line(line)
                if fixed_line != line:
                    print(f"Fixed line {line_no} in {file_path}")
                    modified = True
                    if isinstance(fixed_line, list):
                        new_lines.extend(fixed_line)
                    else:
                        new_lines.append(fixed_line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def fix_long_line(line: str) -> str:
    """Fix a single long line"""
    stripped = line.rstrip()
    indent = len(line) - len(line.lstrip())
    base_indent = ' ' * indent
    
    # Skip certain types of lines that are hard to split
    if any(pattern in stripped for pattern in [
        'http://', 'https://', 'ftp://', 'www.',
        'SELECT ', 'INSERT ', 'UPDATE ', 'DELETE ',
        'CREATE ', 'ALTER ', 'DROP '
    ]):
        return line
    
    # Handle f-strings
    if 'f"' in stripped or "f'" in stripped:
        return fix_f_string_line(line, base_indent)
    
    # Handle logger statements
    if any(pattern in stripped for pattern in ['logger.error', 'logger.warning', 'logger.info', 'logger.debug']):
        return fix_logger_line(line, base_indent)
    
    # Handle function calls with long parameter lists
    if '(' in stripped and ')' in stripped and '=' in stripped:
        return fix_function_call(line, base_indent)
    
    # Handle long string literals
    if ('"' in stripped and stripped.count('"') >= 2) or ("'" in stripped and stripped.count("'") >= 2):
        return fix_string_literal(line, base_indent)
    
    # Handle dictionary/list literals
    if any(char in stripped for char in ['{', '[']) and any(char in stripped for char in ['}', ']']):
        return fix_dict_or_list(line, base_indent)
    
    # Default: try to split at logical operators or commas
    return fix_generic_line(line, base_indent)

def fix_f_string_line(line: str, base_indent: str) -> str:
    """Fix long f-string lines"""
    stripped = line.rstrip()
    
    # Find f-string pattern
    if 'f"' in stripped:
        # Split f-string at logical points
        if '{' in stripped and '}' in stripped:
            # Try to split after variables in f-strings
            pattern = r'(f"[^"]*?)(\{[^}]+\})([^"]*")'
            match = re.search(pattern, stripped)
            if match and len(stripped) > 100:
                before = match.group(1)
                var = match.group(2)
                after = match.group(3)
                
                if len(before + var) < 90:
                    return [
                        base_indent + before + var + '"\n',
                        base_indent + 'f"' + after[1:] + '\n'
                    ]
    
    # Fallback to generic string fixing
    return fix_string_literal(line, base_indent)

def fix_logger_line(line: str, base_indent: str) -> str:
    """Fix long logger statement lines"""
    stripped = line.rstrip()
    
    # Handle logger with f-strings
    if 'f"' in stripped or "f'" in stripped:
        logger_match = re.match(r'(\s*logger\.\w+\(\s*)(f["\'][^"\']*["\'])\s*\)', stripped)
        if logger_match:
            logger_part = logger_match.group(1)
            f_string = logger_match.group(2)
            
            # Split the f-string
            if len(f_string) > 80:
                # Try to find good break points in f-strings
                if '{' in f_string and '}' in f_string:
                    # Split at variable boundaries
                    parts = re.split(r'(\{[^}]+\})', f_string)
                    if len(parts) > 2:
                        mid_point = len(parts) // 2
                        first_half = ''.join(parts[:mid_point])
                        second_half = 'f"' + ''.join(parts[mid_point:])[2:]  # Remove f" from start
                        
                        return [
                            base_indent + logger_part + '\n',
                            base_indent + '    ' + first_half + '"\n',
                            base_indent + '    ' + second_half + '\n',
                            base_indent + ')\n'
                        ]
            
            # Simple split
            return [
                base_indent + logger_part + '\n',
                base_indent + '    ' + f_string + '\n',
                base_indent + ')\n'
            ]
    
    return line

def fix_string_literal(line: str, base_indent: str) -> str:
    """Fix long string literal lines"""
    stripped = line.rstrip()
    
    # Handle concatenated strings
    if ' + ' in stripped and ('"' in stripped or "'" in stripped):
        parts = stripped.split(' + ')
        if len(parts) > 1:
            result = [base_indent + parts[0] + ' +\n']
            for part in parts[1:-1]:
                result.append(base_indent + '    ' + part + ' +\n')
            result.append(base_indent + '    ' + parts[-1] + '\n')
            return result
    
    # Try to split long strings at logical points
    quote_char = '"' if '"' in stripped else "'"
    if quote_char in stripped:
        parts = stripped.split(quote_char)
        if len(parts) >= 3:
            # Try to split the string content
            string_content = parts[1]
            if len(string_content) > 60:
                # Split at spaces or common separators
                for sep in [' → ', ' - ', ': ', ', ', ' ']:
                    if sep in string_content:
                        split_point = string_content.find(sep, len(string_content) // 2)
                        if split_point > 0:
                            first_part = string_content[:split_point + len(sep)]
                            second_part = string_content[split_point + len(sep):]
                            
                            return [
                                                                base_indent +
                                    parts[0] +
                                    quote_char +
                                    first_part +
                                    quote_char +
                                    '\n',
                                                                base_indent +
                                    quote_char +
                                    second_part +
                                    quote_char +
                                    ''.join(parts[2:]) +
                                    '\n'
                            ]
    
    return line

def fix_function_call(line: str, base_indent: str) -> str:
    """Fix long function call lines"""
    stripped = line.rstrip()
    
    # Handle function calls with multiple parameters
    if '(' in stripped and ')' in stripped and ',' in stripped:
        func_match = re.match(r'(\s*[^(]+\()(.*)\)', stripped)
        if func_match:
            func_start = func_match.group(1)
            params = func_match.group(2)
            
            # Split parameters
            param_list = []
            current_param = ""
            paren_count = 0
            quote_char = None
            
            for char in params:
                if char in ['"', "'"]:
                    if quote_char is None:
                        quote_char = char
                    elif quote_char == char:
                        quote_char = None
                elif char == '(' and quote_char is None:
                    paren_count += 1
                elif char == ')' and quote_char is None:
                    paren_count -= 1
                elif char == ',' and paren_count == 0 and quote_char is None:
                    param_list.append(current_param.strip())
                    current_param = ""
                    continue
                
                current_param += char
            
            if current_param.strip():
                param_list.append(current_param.strip())
            
            if len(param_list) > 1:
                result = [base_indent + func_start + '\n']
                for i, param in enumerate(param_list):
                    ending = ',' if i < len(param_list) - 1 else ''
                    result.append(base_indent + '    ' + param + ending + '\n')
                result.append(base_indent + ')\n')
                return result
    
    return line

def fix_dict_or_list(line: str, base_indent: str) -> str:
    """Fix long dictionary or list lines"""
    stripped = line.rstrip()
    
    # Handle dictionary/list with multiple items
    if '{' in stripped and '}' in stripped and ',' in stripped:
        # Simple split at commas
        parts = [part.strip() for part in stripped.split(',')]
        if len(parts) > 1:
            result = []
            for i, part in enumerate(parts):
                if i == 0:
                    result.append(base_indent + part + ',\n')
                elif i == len(parts) - 1:
                    result.append(base_indent + '    ' + part + '\n')
                else:
                    result.append(base_indent + '    ' + part + ',\n')
            return result
    
    return line

def fix_generic_line(line: str, base_indent: str) -> str:
    """Generic line fixing for other cases"""
    stripped = line.rstrip()
    
    # Try to split at logical operators
    for op in [' and ', ' or ', ' if ', ' else ', ' == ', ' != ', ' >= ', ' <= ', ' + ', ' - ']:
        if op in stripped:
            split_point = stripped.find(op, len(stripped) // 2)
            if split_point > 0:
                first_part = stripped[:split_point]
                second_part = stripped[split_point:]
                return [
                    base_indent + first_part + '\n',
                    base_indent + '    ' + second_part.lstrip() + '\n'
                ]
    
    # Try to split at parentheses
    if '(' in stripped and len(stripped) > 100:
        paren_pos = stripped.find('(')
        if paren_pos > 0 and paren_pos < len(stripped) - 20:
            return [
                base_indent + stripped[:paren_pos + 1] + '\n',
                base_indent + '    ' + stripped[paren_pos + 1:] + '\n'
            ]
    
    return line

def main():
    """Main function to fix all E501 errors in the project"""
    project_root = Path(__file__).parent
    
    # Files to process (mentioned in the prompt)
    target_files = [
        "src/main.py",
        "src/mt5_integration.py", 
        "scripts/ultimate_bulletproof_bot.py",
        "scripts/utils.py"
    ]
    
    # Also process other important files
    additional_patterns = [
        "src/*.py",
        "scripts/*.py",
        "ui/*.py",
        "*.py"
    ]
    
    all_files = set()
    
    # Add target files
    for file_path in target_files:
        full_path = project_root / file_path
        if full_path.exists():
            all_files.add(str(full_path))
    
    # Add files from patterns
    for pattern in additional_patterns:
        for file_path in project_root.glob(pattern):
            if file_path.is_file() and file_path.suffix == '.py':
                # Skip virtual environment and __pycache__ directories
                if not any(part in str(file_path) for part in ['.venv', '__pycache__', '.git']):
                    all_files.add(str(file_path))
    
    print(f"Processing {len(all_files)} Python files...")
    
    fixed_files = 0
    for file_path in sorted(all_files):
        if fix_long_lines_in_file(file_path):
            fixed_files += 1
    
    print(f"✅ Fixed E501 errors in {fixed_files} files")
    
    # Run a quick check to see remaining issues
    print("Running final lint check...")
    import subprocess
    try:
        result = subprocess.run([
            sys.executable, "-m", "flake8", "--select=E501", "--max-line-length=100", "."
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("🎉 All E501 errors fixed!")
        else:
            # Count remaining errors
            remaining_lines = [line for line in result.stdout.split('\n') if 'E501' in line and not '.venv' in line]
            print(f"⚠️ {len(remaining_lines)} E501 errors remain (excluding .venv)")
            
            # Show first few remaining errors
            for line in remaining_lines[:5]:
                print(f"  {line}")
            if len(remaining_lines) > 5:
                print(f"  ... and {len(remaining_lines) - 5} more")
                
    except Exception as e:
        print(f"Error running lint check: {e}")

if __name__ == "__main__":
    main()
