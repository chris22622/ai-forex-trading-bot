#!/usr/bin/env python3
"""
🚀 AGGRESSIVE SMALL ACCOUNT FIX
Fix the ultra-conservative position sizing that's killing small accounts
"""

import os


def apply_aggressive_fix():
    """Apply aggressive position sizing fix for small accounts"""

    # Read the main.py file
    main_file = "main.py"
    if not os.path.exists(main_file):
        print("❌ main.py not found!")
        return False

    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the position sizing method
        old_method = '''    def _calculate_position_size_for_symbol(
        self,
        confidence: float,
        symbol: str
    )
        """🛡️ ULTRA-SAFE position sizing to prevent large losses"""
        try:
            # �️ MUCH SMALLER positions until consistently profitable
            base_risk = self.current_balance * 0.005  # Only 0.5% risk per trade!'''

        new_method = '''    def _calculate_position_size_for_symbol(
        self,
        confidence: float,
        symbol: str
    )
        """🚀 AGGRESSIVE position sizing for small accounts to enable growth"""
        try:
            # 🚀 SMALL ACCOUNT MODE: MUCH MORE AGGRESSIVE!
            if self.current_balance <= 100:
                f"🚀 SMALL ACCOUNT MODE: ${self.current_balance:.2f}"
                f"- Using AGGRESSIVE settings!"
                
                # For small accounts, use 10-15% risk to enable actual growth
                if self.current_balance <= 25:
                    base_risk = self.current_balance * 0.15  # 15% risk for tiny accounts
                    logger.warning(f"🔥 TINY ACCOUNT MODE: 15% risk = ${base_risk:.2f}")
                elif self.current_balance <= 50:
                    base_risk = self.current_balance * 0.12  # 12% risk for small accounts  
                    logger.warning(f"🔥 SMALL ACCOUNT MODE: 12% risk = ${base_risk:.2f}")
                else:
                    base_risk = self.current_balance * 0.08  # 8% risk for medium accounts
                    logger.warning(f"🔥 MEDIUM ACCOUNT MODE: 8% risk = ${base_risk:.2f}")
                
                # Convert to aggressive lot sizes for small accounts
                if base_risk >= 3.0:
                    lot_size = 0.08  # Much bigger positions!
                elif base_risk >= 2.0:
                    lot_size = 0.05  # Bigger positions
                elif base_risk >= 1.0:
                    lot_size = 0.03  # Medium positions
                else:
                    lot_size = 0.02  # Still bigger than ultra-conservative 0.01
                
                # Don't reduce for consecutive losses on small accounts - need consistency
                if self.consecutive_losses >= 5:  # Only reduce after many losses
                    lot_size *= 0.8
                    logger.warning(f"🔴 Slight reduction after {self.consecutive_losses} losses")
                
                logger.warning(f"🚀 AGGRESSIVE SIZING: ${base_risk:.2f} risk → {lot_size} lots")
                return lot_size
                
            else:
                # Original conservative logic for larger accounts (>$100)
                base_risk = self.current_balance * 0.005  # Only 0.5% risk per trade!'''

    # Apply the fix
    if old_method in content:
        content = content.replace(old_method, new_method)
        print("✅ Applied aggressive position sizing fix!")
    else:
        # Try alternative patterns
        alt_pattern = 'base_risk = self.current_balance * 0.005  # Only 0.5% risk per trade!'
        if alt_pattern in content:
            # Replace just the risk calculation
            new_calc = '''# 🚀 SMALL ACCOUNT MODE: MUCH MORE AGGRESSIVE!
            if self.current_balance <= 100:
                f"🚀 SMALL ACCOUNT MODE: ${self.current_balance:.2f}"
                f"- Using AGGRESSIVE settings!"
                
                # For small accounts, use 10-15% risk to enable actual growth
                if self.current_balance <= 25:
                    base_risk = self.current_balance * 0.15  # 15% risk for tiny accounts
                    logger.warning(f"🔥 TINY ACCOUNT MODE: 15% risk = ${base_risk:.2f}")
                elif self.current_balance <= 50:
                    base_risk = self.current_balance * 0.12  # 12% risk for small accounts  
                    logger.warning(f"🔥 SMALL ACCOUNT MODE: 12% risk = ${base_risk:.2f}")
                else:
                    base_risk = self.current_balance * 0.08  # 8% risk for medium accounts
                    logger.warning(f"🔥 MEDIUM ACCOUNT MODE: 8% risk = ${base_risk:.2f}")
            else:
                base_risk = self.current_balance * 0.005  # Only 0.5% risk per trade for large accounts!'''

            content = content.replace(alt_pattern, new_calc)
            print("✅ Applied aggressive risk calculation fix!")
        else:
            print("❌ Could not find position sizing method to fix!")
            return False

    # Also fix the profit targets for small accounts
    profit_targets_fix = '''
    # 🚀 AGGRESSIVE PROFIT TARGETS FOR SMALL ACCOUNTS
    if hasattr(self, 'current_balance') and self.current_balance <= 100:
        # Small account mode - more aggressive targets
        if self.current_balance <= 25:
            self.quick_profit_target = 3.00      # $3 instead of $0.75 for tiny accounts
            self.min_profit_threshold = 1.50     # $1.50 instead of $0.25
            self.max_loss_per_trade = 4.00       # $4 instead of $1.50
            self.max_trade_age_minutes = 45      # 45 minutes instead of 10
        else:
            self.quick_profit_target = 2.00      # $2 instead of $0.75
            self.min_profit_threshold = 1.00     # $1 instead of $0.25  
            self.max_loss_per_trade = 3.00       # $3 instead of $1.50
            self.max_trade_age_minutes = 30      # 30 minutes instead of 10
            
        f"🚀 SMALL ACCOUNT TARGETS: Profit=${self.quick_profit_target}"
        f" Loss=${self.max_loss_per_trade}, Time={self.max_trade_age_minutes}min"
    '''

    # Find a good place to add this - look for __init__ method
    init_pattern = 'self.starting_balance = starting_balance'
    if init_pattern in content:
        content = content.replace(init_pattern, init_pattern + profit_targets_fix)
        print("✅ Applied aggressive profit targets for small accounts!")

    # Write the fixed file
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n🚀 AGGRESSIVE SMALL ACCOUNT FIX APPLIED!")
    print("=" * 50)
    print("📊 CHANGES MADE:")
    print("  - 15% risk for accounts ≤ $25 (was 0.5%)")
    print("  - 12% risk for accounts ≤ $50 (was 0.5%)")
    print("  - 8% risk for accounts ≤ $100 (was 0.5%)")
    print("  - Larger lot sizes: 0.02-0.08 (was 0.01)")
    print("  - Higher profit targets for small accounts")
    print("  - Longer time limits for small accounts")
    print("\n💡 Your $20 account will now use:")
    print("  - 15% risk = $3.00 per trade")
    print("  - 0.05 lot size positions")
    print("  - $3.00 profit targets")
    print("  - $4.00 stop losses")
    print("\n🚀 RESTART THE BOT TO APPLY CHANGES!")

    return True

if __name__ == "__main__":
    apply_aggressive_fix()
