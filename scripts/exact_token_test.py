"""
EXACT TOKEN VALIDATOR
Tests your exact tokens with proper authentication flow
"""

import asyncio
import json

import requests
import websockets


async def test_token_websocket(token, token_type="demo"):
    """Test token via WebSocket with proper authentication"""
    print(f"\n🔍 Testing {token_type.upper()} token: '{token}'")
    print(f"   📏 Length: {len(token)} characters")

    # Try each WebSocket URL
    urls = [
        "wss://ws.derivws.com/websockets/v3",
        "wss://ws.binaryws.com/websockets/v3",
        "wss://frontend.derivws.com/websockets/v3"
    ]

    for url in urls:
        try:
            print(f"   🌐 Connecting to: {url}")

            # Connect to WebSocket
            websocket = await websockets.connect(url, timeout=15)
            print("   ✅ Connected successfully")

            # Send authorization with your exact token
            auth_request = {
                "authorize": token,
                "req_id": 1
            }

            print("   📤 Sending auth request...")
            await websocket.send(json.dumps(auth_request))

            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=15)
            data = json.loads(response)

            print(f"   📥 Raw response: {response}")

            if "authorize" in data:
                auth_info = data["authorize"]
                print("   ✅ TOKEN VALID! Authorization successful!")
                print(f"      📧 Email: {auth_info.get('email', 'N/A')}")
                print(f"      🏦 Account: {auth_info.get('loginid', 'N/A')}")
                print(f"      💰 Balance: ${auth_info.get('balance', 0)}")
                print(f"      🌍 Country: {auth_info.get('country', 'N/A')}")
                print(f"      🎯 Currency: {auth_info.get('currency', 'N/A')}")

                await websocket.close()
                return True, url

            elif "error" in data:
                error = data["error"]
                print("   ❌ Authorization failed!")
                print(f"      🚨 Code: {error.get('code', 'Unknown')}")
                print(f"      📝 Message: {error.get('message', 'Unknown')}")

                await websocket.close()
                return False, f"Error: {error.get('message', 'Unknown')}"
            else:
                print(f"   ❓ Unexpected response: {data}")
                await websocket.close()
                continue

        except websockets.exceptions.InvalidStatusCode as e:
            print(f"   ❌ HTTP {e.status_code} error (before token test)")
            if e.status_code == 401:
                print("      🚨 This is a NETWORK issue, not token issue!")
        except Exception as e:
            print(f"   ❌ Connection error: {e}")

    return False, "All URLs failed"

def test_token_http(token, token_type="demo"):
    """Test token via HTTP API"""
    print(f"\n🌐 Testing {token_type.upper()} token via HTTP API")

    try:
        url = "https://api.deriv.com/api/v1/authorize"
        payload = {"authorize": token}

        print(f"   📤 Sending to: {url}")
        print(f"   📦 Payload: {payload}")

        response = requests.post(url, json=payload, timeout=20)
        print(f"   📊 Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"   📥 Response: {json.dumps(result, indent=2)}")

            if "authorize" in result:
                print("   ✅ HTTP API validation successful!")
                return True, result
            elif "error" in result:
                error = result["error"]
                print(f"   ❌ API returned error: {error}")
                return False, error
        else:
            print(f"   ❌ HTTP error {response.status_code}")
            print(f"   📝 Response: {response.text}")
            return False, f"HTTP {response.status_code}"

    except Exception as e:
        print(f"   ❌ HTTP test error: {e}")
        return False, str(e)

async def main():
    """Main test function"""
    print("=" * 70)
    print("🔧 EXACT TOKEN VALIDATOR")
    print("=" * 70)

    # Get tokens from config
    try:
        from config import DEMO_MODE, DERIV_DEMO_API_TOKEN, DERIV_LIVE_API_TOKEN

        print("\n📊 YOUR CURRENT TOKENS:")
        print(f"   🎮 Demo Token: '{DERIV_DEMO_API_TOKEN}' ({len(DERIV_DEMO_API_TOKEN)} chars)")
        print(f"   💰 Live Token: '{DERIV_LIVE_API_TOKEN}' ({len(DERIV_LIVE_API_TOKEN)} chars)")
        print(f"   🎯 Current Mode: {'DEMO' if DEMO_MODE else 'LIVE'}")

        # Test both tokens via HTTP first (easier to debug)
        print("\n" + "="*50)
        print("🌐 HTTP API TESTS")
        print("="*50)

        demo_http_valid, demo_http_result = test_token_http(DERIV_DEMO_API_TOKEN, "demo")
        live_http_valid, live_http_result = test_token_http(DERIV_LIVE_API_TOKEN, "live")

        # Test both tokens via WebSocket
        print("\n" + "="*50)
        print("🔌 WEBSOCKET TESTS")
        print("="*50)

        demo_ws_valid, demo_ws_result = await test_token_websocket(DERIV_DEMO_API_TOKEN, "demo")
        live_ws_valid, live_ws_result = await test_token_websocket(DERIV_LIVE_API_TOKEN, "live")

        # Final summary
        print("\n" + "="*70)
        print("📋 FINAL TOKEN VALIDATION RESULTS")
        print("="*70)

        print(f"🎮 DEMO TOKEN: '{DERIV_DEMO_API_TOKEN}'")
        print(f"   HTTP API: {'✅ VALID' if demo_http_valid else '❌ INVALID'}")
        print(f"   WebSocket: {'✅ VALID' if demo_ws_valid else '❌ INVALID'}")
        if not demo_http_valid:
            print(f"   Error: {demo_http_result}")

        print(f"\n💰 LIVE TOKEN: '{DERIV_LIVE_API_TOKEN}'")
        print(f"   HTTP API: {'✅ VALID' if live_http_valid else '❌ INVALID'}")
        print(f"   WebSocket: {'✅ VALID' if live_ws_valid else '❌ INVALID'}")
        if not live_http_valid:
            print(f"   Error: {live_http_result}")

        # Diagnosis and recommendations
        print("\n🎯 DIAGNOSIS:")

        if demo_http_valid or live_http_valid:
            print("✅ At least one token is valid!")

            if DEMO_MODE and demo_ws_valid:
                print("✅ Your bot should work in DEMO mode!")
            elif not DEMO_MODE and live_ws_valid:
                print("✅ Your bot should work in LIVE mode!")
            elif demo_ws_valid:
                print("⚠️ Switch to DEMO mode (set DEMO_MODE = True)")
            elif live_ws_valid:
                print("⚠️ Switch to LIVE mode (set DEMO_MODE = False)")
            else:
                print("🚨 Tokens work via HTTP but not WebSocket")
                print("   This suggests a network/firewall issue")

        else:
            print("❌ Both tokens are invalid!")
            print("🔧 Your tokens might be:")
            print("   1. Expired or revoked")
            print("   2. Missing required permissions")
            print("   3. From wrong account type")

        print("\n💡 NEXT STEPS:")
        if demo_http_valid or live_http_valid:
            print("1. ✅ Your tokens are valid")
            print("2. 🔧 The HTTP 401 errors are network-related")
            print("3. 🌐 Try connecting via VPN")
            print("4. 📱 Or use mobile hotspot")
        else:
            print("1. 🔧 Go to: https://app.deriv.com/account/api-token")
            print("2. 🔄 Create new tokens with ALL permissions")
            print("3. 📋 Copy the FULL tokens")
            print("4. 📝 Update config.py")

    except ImportError:
        print("❌ Cannot import config.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
