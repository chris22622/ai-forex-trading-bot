"""
QUICK API TOKEN CREATOR
Opens the correct Deriv pages to create your API tokens
"""

import time
import webbrowser


def main():
    print("🔧 DERIV API TOKEN CREATOR")
    print("=" * 50)

    print("\n📋 WHAT YOU NEED TO DO:")
    print("1. I'll open 2 browser tabs")
    print("2. Create DEMO token first")
    print("3. Create LIVE token second")
    print("4. Copy both tokens to config.py")

    input("\n👆 Press ENTER when ready...")

    print("\n🎮 Opening DEMO token page...")
    webbrowser.open("https://app.deriv.com/account/api-token")
    time.sleep(2)

    print("\n📝 DEMO TOKEN INSTRUCTIONS:")
    print("1. Make sure you're on DEMO account (top-right)")
    print("2. Click 'Create new token'")
    print("3. Name: 'Demo Trading Bot'")
    print("4. Check ALL permission boxes")
    print("5. Click 'Create'")
    print("6. COPY the long token (32+ characters)")

    input("\n✅ Press ENTER after creating DEMO token...")

    print("\n💰 Opening LIVE token page...")
    webbrowser.open("https://app.deriv.com/account/api-token")

    print("\n📝 LIVE TOKEN INSTRUCTIONS:")
    print("1. Switch to REAL account (top-right)")
    print("2. Click 'Create new token'")
    print("3. Name: 'Live Trading Bot'")
    print("4. Check ALL permission boxes")
    print("5. Click 'Create'")
    print("6. COPY the long token (32+ characters)")

    input("\n✅ Press ENTER after creating LIVE token...")

    print("\n📋 NOW UPDATE CONFIG.PY:")
    print("1. Open config.py in VS Code")
    print("2. Replace DERIV_DEMO_API_TOKEN with your demo token")
    print("3. Replace DERIV_LIVE_API_TOKEN with your live token")
    print("4. Save the file")
    print("5. Run: python FIX_API_TOKENS.py (to verify)")
    print("6. Run: python main.py (your bot will work!)")

    print("\n🎉 TOKENS CREATED! Update config.py and run your bot!")

if __name__ == "__main__":
    main()
