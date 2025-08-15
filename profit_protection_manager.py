#!/usr/bin/env python3
"""
🛡️ Profit Protection Manager
Simple utility to check and manage profit protection system
"""

import sys
import os

# Add the directory to path so we can import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import check_profit_protection_status, reset_profit_protection_manual
    
    def main():
        print("🛡️ PROFIT PROTECTION MANAGER")
        print("=" * 40)
        
        while True:
            print("\nOptions:")
            print("1. Check profit protection status")
            print("2. Reset profit protection (resume trading)")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\n" + "="*40)
                check_profit_protection_status()
                print("="*40)
                
            elif choice == "2":
                print("\n⚠️  WARNING: This will reset profit protection and resume trading!")
                confirm = input("Are you sure? (yes/no): ").strip().lower()
                
                if confirm in ['yes', 'y']:
                    print("\n" + "="*40)
                    reset_profit_protection_manual()
                    print("="*40)
                else:
                    print("❌ Reset cancelled")
                    
            elif choice == "3":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Error importing from main.py: {e}")
    print("Make sure the trading bot is in the same directory.")
    sys.exit(1)
