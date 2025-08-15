#!/usr/bin/env python3
"""
API Token Validator for Deriv
Tests the API token and checks permissions
"""

import asyncio
import json

import websockets

from config import DERIV_API_TOKEN, DERIV_WS_URL


async def test_api_token():
    """Test the API token and check permissions"""
    print("🔍 Testing Deriv API Token...")
    print(f"Token: {DERIV_API_TOKEN}")
    print(f"WebSocket URL: {DERIV_WS_URL}")
    print("-" * 50)

    try:
        # Connect to WebSocket without authentication headers first
        print("📡 Connecting to Deriv WebSocket...")

        # Try different connection approaches
        websocket = None

        # Approach 1: Standard connection with proper headers
        try:
            websocket = await websockets.connect(
                DERIV_WS_URL,
                ping_interval=30,
                ping_timeout=10,
                additional_headers={
                    "Origin": "https://app.deriv.com",
                    "User-Agent": "TradingBot/1.0"
                }
            )
            print("✅ WebSocket connected successfully with headers")
        except Exception as e:
            print(f"❌ Connection with headers failed: {e}")

            # Approach 2: Simple connection
            try:
                websocket = await websockets.connect(
                    DERIV_WS_URL,
                    ping_interval=30,
                    ping_timeout=10
                )
                print("✅ WebSocket connected successfully (simple)")
            except Exception as e2:
                print(f"❌ Simple connection failed: {e2}")
                print(f"❌ Error details: {type(e2).__name__}: {e2}")
                return

        if not websocket:
            print("❌ Could not establish WebSocket connection")
            return

        # Test authorization
        print("� Testing authorization...")
        auth_request = {
            "authorize": DERIV_API_TOKEN,
            "req_id": 1
        }

        await websocket.send(json.dumps(auth_request))

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            data = json.loads(response)
            print(f"📨 Received response: {json.dumps(data, indent=2)}")

            if "authorize" in data:
                auth_info = data["authorize"]
                print("✅ Authorization successful!")
                print(f"� Email: {auth_info.get('email', 'N/A')}")
                print(f"💰 Currency: {auth_info.get('currency', 'N/A')}")
                print(f"🏦 Balance: {auth_info.get('balance', 'N/A')}")
                print(f"🆔 Login ID: {auth_info.get('loginid', 'N/A')}")
                print(f"🎯 Country: {auth_info.get('country', 'N/A')}")

                # Check API token details
                if 'scopes' in auth_info:
                    print(f"🔑 API Scopes: {auth_info['scopes']}")

                # Test market data access
                print("\n📈 Testing market data access...")
                tick_request = {
                    "ticks": "R_75",
                    "subscribe": 1,
                    "req_id": 3
                }

                await websocket.send(json.dumps(tick_request))
                tick_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                tick_data = json.loads(tick_response)

                if "tick" in tick_data:
                    tick = tick_data["tick"]
                    print("✅ Market data access: OK")
                    print(f"📊 Symbol: {tick.get('symbol', 'N/A')}")
                    print(f"💲 Price: {tick.get('quote', 'N/A')}")
                    print(f"🕒 Time: {tick.get('epoch', 'N/A')}")
                elif "error" in tick_data:
                    print(f"❌ Market data error: {tick_data['error'].get('message', 'Unknown')}")
                else:
                    print(f"📨 Tick response: {json.dumps(tick_data, indent=2)}")

            elif "error" in data:
                error = data["error"]
                print("❌ Authorization failed!")
                print(f"Error Code: {error.get('code', 'N/A')}")
                print(f"Error Message: {error.get('message', 'N/A')}")
                print(f"Error Details: {error.get('details', 'N/A')}")

                # Common error fixes
                if error.get('code') == 'InvalidToken':
                    print("\n🔧 Possible fixes:")
                    print("1. Check if the API token is correct")
                    print("2. Verify the token hasn't expired")
                    print("3. Make sure trading permissions are enabled")
                    print("4. Check if the account is verified")
                elif 'read' in error.get('message', '').lower():
                    print("\n🔧 Token may have limited permissions")
                    print("1. Create new token with 'Trading' permissions")
                    print("2. Enable 'Admin' scope if needed")
            else:
                print(f"❓ Unexpected response: {json.dumps(data, indent=2)}")

        except asyncio.TimeoutError:
            print("⏰ Request timed out - server not responding")

        await websocket.close()

    except websockets.exceptions.InvalidStatus as e:
        print(f"❌ WebSocket connection failed: {e}")
        if "401" in str(e):
            print("\n🔧 HTTP 401 indicates authentication issue:")
            print("1. API token may be invalid or expired")
            print("2. Token may not have required permissions")
            print("3. Check token was created for correct account type")
            print("4. Verify account is verified and active")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print("📋 Full error details:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_api_token())
