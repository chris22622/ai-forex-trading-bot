"""
Simple Connection Test - Bypassing Complex WebSocket Issues
"""
import asyncio
import json
from datetime import datetime

import websockets

# Your tokens
DEMO_TOKEN = "s8Sww40XKyKMp0x"
LIVE_TOKEN = "g9kxuLTUWxrHGNe"

class SimpleConnectionTest:
    def __init__(self):
        self.urls = [
            "wss://ws.derivws.com/websockets/v3",
            "wss://ws.binaryws.com/websockets/v3"
        ]

    async def test_basic_connection(self, url):
        """Test basic connection without timeout parameter"""
        try:
            print(f"🔌 Testing: {url}")

            # Connect without timeout parameter to avoid the error
            async with websockets.connect(url) as websocket:
                print(f"✅ Connected to {url}")

                # Test ping
                await websocket.ping()
                print("✅ Ping successful")

                return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    async def test_token_auth(self, url, token, token_type):
        """Test token authentication"""
        try:
            print(f"\n🔐 Testing {token_type} token: {token}")
            print(f"🌐 URL: {url}")

            async with websockets.connect(url) as websocket:
                # Send authorization
                auth_request = {
                    "authorize": token,
                    "req_id": 1
                }

                await websocket.send(json.dumps(auth_request))
                print("📤 Sent authorization request")

                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)

                print(f"📥 Response: {json.dumps(data, indent=2)}")

                if "authorize" in data:
                    auth_info = data["authorize"]
                    print(f"✅ {token_type} Authorization SUCCESS!")
                    print(f"   Account: {auth_info.get('loginid', 'N/A')}")
                    print(f"   Balance: ${auth_info.get('balance', 0):.2f}")
                    print(f"   Currency: {auth_info.get('currency', 'N/A')}")
                    return True

                elif "error" in data:
                    error = data["error"]
                    print(f"❌ {token_type} Authorization FAILED:")
                    print(f"   Code: {error.get('code', 'Unknown')}")
                    print(f"   Message: {error.get('message', 'Unknown error')}")
                    return False

        except asyncio.TimeoutError:
            print(f"⏰ {token_type} authorization timeout")
            return False
        except Exception as e:
            print(f"❌ {token_type} authorization error: {e}")
            return False

    async def run_all_tests(self):
        """Run all connection tests"""
        print("🔧 SIMPLE CONNECTION TEST")
        print("=" * 50)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        working_urls = []

        # Test basic connections
        print("📡 TESTING BASIC CONNECTIONS")
        print("-" * 30)
        for url in self.urls:
            if await self.test_basic_connection(url):
                working_urls.append(url)

        if not working_urls:
            print("\n❌ NO WORKING CONNECTIONS FOUND")
            print("🔧 Troubleshooting:")
            print("   1. Check your internet connection")
            print("   2. Try different network (mobile hotspot)")
            print("   3. Check if firewall is blocking WebSocket connections")
            print("   4. Contact your ISP about WebSocket restrictions")
            return

        print(f"\n✅ Found {len(working_urls)} working connection(s)")

        # Test token authentication
        print("\n🔐 TESTING TOKEN AUTHENTICATION")
        print("-" * 35)

        success_count = 0
        for url in working_urls:
            print(f"\n🌐 Testing with: {url}")

            # Test demo token
            if await self.test_token_auth(url, DEMO_TOKEN, "DEMO"):
                success_count += 1
                break  # Stop if we find working combination

            # Test live token
            if await self.test_token_auth(url, LIVE_TOKEN, "LIVE"):
                success_count += 1
                break

        if success_count > 0:
            print("\n🎉 SUCCESS! Found working connection + token combination")
            print("✅ Your bot should work now!")
        else:
            print("\n❌ NO WORKING TOKEN COMBINATIONS")
            print("🔧 Token troubleshooting:")
            print("   1. Go to: https://app.deriv.com/account/api-token")
            print("   2. Create new tokens with ALL permissions")
            print("   3. Make sure account is verified")
            print("   4. Check token expiry dates")

async def main():
    tester = SimpleConnectionTest()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
