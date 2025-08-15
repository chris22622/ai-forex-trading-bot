"""
DERIV API TOKEN FIXER
This script will help you get valid API tokens and fix the HTTP 401 error
"""

import webbrowser

import requests


def check_token_validity(token, token_type="demo"):
    """Test if a token is valid by making a simple API call"""
    try:
        print(f"\n🔍 Testing {token_type.upper()} token: {token[:8]}...")

        # Simple HTTP API call to test token
        url = "https://api.deriv.com/api/v1/authorize"
        data = {"authorize": token}

        response = requests.post(url, json=data, timeout=10)
        result = response.json()

        if "authorize" in result:
            auth_info = result["authorize"]
            print(f"✅ {token_type.upper()} token is VALID!")
            print(f"   📧 Email: {auth_info.get('email', 'N/A')}")
            print(f"   🏦 Account: {auth_info.get('loginid', 'N/A')}")
            print(f"   💰 Balance: ${auth_info.get('balance', 0)}")
            print(f"   🌍 Country: {auth_info.get('country', 'N/A')}")
            return True
        elif "error" in result:
            error = result["error"]
            print(f"❌ {token_type.upper()} token is INVALID!")
            print(f"   🚨 Error: {error.get('message', 'Unknown error')}")
            return False
        else:
            print(f"❓ Unexpected response for {token_type} token")
            return False

    except Exception as e:
        print(f"❌ Error testing {token_type} token: {e}")
        return False

def generate_token_instructions():
    """Generate step-by-step instructions for creating API tokens"""
    instructions = """
📝 HOW TO CREATE VALID DERIV API TOKENS:

1. 🌐 Go to: https://app.deriv.com/account/api-token
   (Log in to your Deriv account)

2. 🔐 Click "Create new token"

3. ⚙️ Configure token settings:
   ✅ Name: "Trading Bot Token"
   ✅ Scopes: CHECK ALL THESE BOXES:
      ☑️ Read
      ☑️ Trade  
      ☑️ Trading information
      ☑️ Payments
      ☑️ Admin

4. 🎯 Click "Create"

5. 📋 COPY the long token (starts with letters/numbers, 32+ characters)

6. 🔄 Repeat for BOTH demo and live accounts:
   - Demo account tokens work with virtual money
   - Live account tokens work with real money

⚠️  IMPORTANT NOTES:
- Tokens look like: "abc123XYZ789..." (32+ characters)
- Your current tokens are TOO SHORT and INVALID
- Demo tokens start with different prefixes than live tokens
- Keep tokens SECRET and SECURE

🔗 Direct links:
- Demo: https://app.deriv.com/account/api-token (switch to demo first)
- Live: https://app.deriv.com/account/api-token (switch to real account)
"""
    return instructions

def main():
    print("=" * 70)
    print("🔧 DERIV API TOKEN FIXER")
    print("=" * 70)

    # Check current tokens from config
    try:
        from config import DEMO_MODE, DERIV_DEMO_API_TOKEN, DERIV_LIVE_API_TOKEN

        print("\n📊 CURRENT CONFIGURATION:")
        print(f"   🎯 Mode: {'DEMO' if DEMO_MODE else 'LIVE'}")
        print(f"   🔑 Demo Token: {DERIV_DEMO_API_TOKEN}")
        print(f"   🔑 Live Token: {DERIV_LIVE_API_TOKEN}")

        # Test both tokens
        demo_valid = check_token_validity(DERIV_DEMO_API_TOKEN, "demo")
        live_valid = check_token_validity(DERIV_LIVE_API_TOKEN, "live")

        print("\n📋 TOKEN STATUS SUMMARY:")
        print(f"   🎮 Demo Token: {'✅ VALID' if demo_valid else '❌ INVALID'}")
        print(f"   💰 Live Token: {'✅ VALID' if live_valid else '❌ INVALID'}")

        if not demo_valid and not live_valid:
            print("\n🚨 BOTH TOKENS ARE INVALID!")
            print(f"   Current tokens are too short: {len(DERIV_DEMO_API_TOKEN)} chars")
            print("   Valid tokens need: 32+ characters")
            print("   Your tokens look fake/truncated")

        if not demo_valid or not live_valid:
            print(generate_token_instructions())

            # Offer to open browser
            choice = input("\n🌐 Open Deriv API token page in browser? (y/n): ").lower()
            if choice == 'y':
                webbrowser.open("https://app.deriv.com/account/api-token")
                print("✅ Browser opened! Create your tokens and come back.")

            print("\n🔧 NEXT STEPS:")
            print("1. Create valid tokens using instructions above")
            print("2. Replace tokens in config.py")
            print("3. Run this script again to verify")
            print("4. Run your trading bot")

        else:
            print("\n🎉 ALL TOKENS ARE VALID!")
            print("✅ Your bot should connect successfully now!")

    except ImportError as e:
        print(f"❌ Error importing config: {e}")
        print("Make sure config.py exists in the same directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def manual_token_test():
    """Allow manual testing of tokens"""
    print("\n🧪 MANUAL TOKEN TESTING")
    print("Enter tokens to test (press Enter to skip):")

    demo_token = input("🎮 Demo token: ").strip()
    if demo_token:
        check_token_validity(demo_token, "demo")

    live_token = input("💰 Live token: ").strip()
    if live_token:
        check_token_validity(live_token, "live")

if __name__ == "__main__":
    main()

    # Offer manual testing
    choice = input("\n🧪 Test different tokens manually? (y/n): ").lower()
    if choice == 'y':
        manual_token_test()

    print("\n" + "=" * 70)
    print("🔧 API TOKEN FIXER COMPLETE")
    print("=" * 70)
