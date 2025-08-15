#!/usr/bin/env python3
"""
Basic WebSocket Connection Test
Tests if we can connect to Deriv WebSocket without authentication
"""

import asyncio
import json

import websockets


async def test_basic_connection():
    """Test basic WebSocket connection without auth"""
    print("🔄 Testing basic WebSocket connection...")

    try:
        uri = "wss://ws.derivws.com/websockets/v3"
        async with websockets.connect(uri) as ws:
            print("✅ WebSocket connection established successfully!")

            # Try to get server info without authentication
            server_time_request = {
                "time": 1,
                "req_id": 1
            }

            await ws.send(json.dumps(server_time_request))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(response)

            if "time" in data:
                print(f"✅ Server time received: {data['time']}")
                print("✅ Connection is working - problem is with authentication")
                return True
            else:
                print(f"❓ Unexpected response: {data}")
                return False

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def test_token_format():
    """Test token format and basic validation"""
    from config import DERIV_DEMO_API_TOKEN, DERIV_LIVE_API_TOKEN

    print("\n🔍 Checking token format...")

    tokens = [
        ("DEMO", DERIV_DEMO_API_TOKEN),
        ("LIVE", DERIV_LIVE_API_TOKEN)
    ]

    for token_type, token in tokens:
        if not token:
            print(f"❌ {token_type} token is empty!")
            continue

        print(f"\n{token_type} token: {token}")
        print(f"  Length: {len(token)} characters")
        print(f"  First 8: {token[:8]}")
        print(f"  Last 3: {token[-3:]}")

        # Basic validation
        if len(token) < 10:
            print("  ⚠️ Token seems too short (expected 15+ chars)")
        elif len(token) > 50:
            print("  ⚠️ Token seems too long")
        else:
            print("  ✅ Token length looks reasonable")

async def main():
    print("🧪 DERIV CONNECTION DEBUG TEST")
    print("=" * 40)

    # Test basic connection
    basic_ok = await test_basic_connection()

    # Test token format
    await test_token_format()

    print("\n📊 ANALYSIS")
    print("=" * 40)

    if basic_ok:
        print("✅ Network connection to Deriv is working")
        print("🔴 Problem is with API token authentication")
        print("\n🔧 NEXT STEPS:")
        print("1. Double-check your tokens at: https://app.deriv.com/account/api-token")
        print("2. Make sure tokens have these scopes:")
        print("   ✅ Read")
        print("   ✅ Trade")
        print("3. Try creating completely new tokens")
        print("4. Make sure you're copying the FULL token (no spaces)")
    else:
        print("❌ Basic network connection failed")
        print("🔧 Check your internet connection and try VPN")

if __name__ == "__main__":
    asyncio.run(main())
