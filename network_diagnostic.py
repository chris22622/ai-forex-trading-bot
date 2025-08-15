#!/usr/bin/env python3
"""
Network Diagnostic Tool for Deriv WebSocket
Tests different connection methods and URLs
"""

import asyncio
import json
import socket

import websockets


async def test_dns_resolution():
    """Test DNS resolution for Deriv domains"""
    print("🌐 TESTING DNS RESOLUTION")
    print("=" * 50)

    domains = [
        "ws.derivws.com",
        "ws.binaryws.com",
        "app.deriv.com",
        "deriv.com"
    ]

    working_domains = []

    for domain in domains:
        try:
            result = socket.gethostbyname(domain)
            print(f"✅ {domain} → {result}")
            working_domains.append(domain)
        except Exception as e:
            print(f"❌ {domain} → {e}")

    return working_domains

async def test_https_connection():
    """Test HTTPS connection to Deriv"""
    print("\n🔒 TESTING HTTPS CONNECTION")
    print("=" * 50)

    import aiohttp

    urls = [
        "https://app.deriv.com",
        "https://deriv.com"
    ]

    for url in urls:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    print(f"✅ {url} → Status: {response.status}")
        except Exception as e:
            print(f"❌ {url} → {e}")

async def test_websocket_with_proper_format():
    """Test WebSocket with the exact format Deriv expects"""
    print("\n🔌 TESTING WEBSOCKET WITH PROPER FORMAT")
    print("=" * 50)

    # Different URL formats to try
    url_formats = [
        "wss://ws.derivws.com/websockets/v3",
        "wss://ws.binaryws.com/websockets/v3",
        "wss://ws.derivws.com/websocket/v3",  # Note: websocket vs websockets
        "wss://ws.binaryws.com/websocket/v3",
    ]

    for url in url_formats:
        print(f"\n📡 Testing: {url}")

        try:
            # Test with minimal connection
            websocket = await asyncio.wait_for(
                websockets.connect(
                    url,
                    additional_headers={
                        "Origin": "https://app.deriv.com"
                    }
                ),
                timeout=10.0
            )

            print("✅ Connection successful!")

            # Try to send a time request (no auth needed)
            time_request = {
                "time": 1,
                "req_id": 1
            }

            await websocket.send(json.dumps(time_request))

            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"✅ Server response: {data}")

                if "time" in data:
                    print("✅ WebSocket communication working!")
                    await websocket.close()
                    return url

            except asyncio.TimeoutError:
                print("⚠️ Connection made but no response received")

            await websocket.close()

        except websockets.exceptions.InvalidStatus as e:
            if "401" in str(e):
                print("⚠️ HTTP 401 - Server rejecting connection")
                print("   This might be due to missing Origin header or wrong URL")
            else:
                print(f"❌ Connection failed: {e}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")

    return None

async def test_with_curl_simulation():
    """Simulate what curl would do"""
    print("\n🔧 TESTING WITH ALTERNATIVE METHOD")
    print("=" * 50)

    try:
        import aiohttp

        # Test the WebSocket upgrade manually
        url = "wss://ws.derivws.com/websockets/v3"

        # Convert WSS to HTTPS for testing
        http_url = url.replace("wss://", "https://")

        headers = {
            "User-Agent": "DerivBot/1.0",
            "Origin": "https://app.deriv.com",
            "Connection": "Upgrade",
            "Upgrade": "websocket",
            "Sec-WebSocket-Version": "13",
            "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(http_url, headers=headers, timeout=10) as response:
                    print(f"✅ HTTP response: {response.status}")
                    print(f"✅ Headers: {dict(response.headers)}")
            except Exception as e:
                print(f"❌ HTTP test failed: {e}")

    except ImportError:
        print("⚠️ aiohttp not available for advanced testing")

async def comprehensive_network_test():
    """Run comprehensive network diagnostics"""
    print("🔍 COMPREHENSIVE NETWORK DIAGNOSTIC")
    print("="*60)

    # Test 1: DNS Resolution
    working_domains = await test_dns_resolution()

    if not working_domains:
        print("\n❌ CRITICAL: DNS resolution failed!")
        print("🔧 Possible fixes:")
        print("   - Check internet connection")
        print("   - Try different DNS servers (8.8.8.8, 1.1.1.1)")
        print("   - Use VPN if in restricted country")
        return False

    # Test 2: HTTPS Connection
    await test_https_connection()

    # Test 3: WebSocket with proper format
    working_ws = await test_websocket_with_proper_format()

    if working_ws:
        print(f"\n🎉 SUCCESS! Working WebSocket URL: {working_ws}")
        return working_ws

    # Test 4: Alternative method
    await test_with_curl_simulation()

    return None

async def create_fixed_config(working_url=None):
    """Create a fixed configuration"""
    print("\n📝 CREATING FIXED CONFIGURATION")
    print("=" * 50)

    if working_url:
        print(f"✅ Using working URL: {working_url}")

        # Update config with working URL
        try:
            with open('config.py', 'r') as f:
                config_content = f.read()

            # Replace the WebSocket URL
            new_content = config_content.replace(
                'DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"',
                f'DERIV_WS_URL = "{working_url}"  # ✅ TESTED WORKING URL'
            )

            with open('config_network_fixed.py', 'w') as f:
                f.write(new_content)

            print("✅ Fixed config saved to: config_network_fixed.py")
            print("\n📋 TO APPLY:")
            print("1. Backup your config.py")
            print("2. Replace config.py with config_network_fixed.py")
            print("3. Get new API tokens from app.deriv.com")
            print("4. Run the bot!")

        except Exception as e:
            print(f"❌ Error creating fixed config: {e}")
    else:
        print("❌ No working WebSocket URL found")
        print("\n🔧 NETWORK TROUBLESHOOTING:")
        print("1. Check if you're behind a corporate firewall")
        print("2. Try using a VPN (ProtonVPN, NordVPN, etc.)")
        print("3. Use mobile hotspot for testing")
        print("4. Check if your ISP blocks WebSocket connections")
        print("5. Try from a different network/location")

async def main():
    """Main diagnostic function"""
    working_url = await comprehensive_network_test()
    await create_fixed_config(working_url)

    if working_url:
        print("\n🎉 NETWORK ISSUE RESOLVED!")
        print("Next step: Get valid API tokens from app.deriv.com")
    else:
        print("\n❌ NETWORK ISSUE PERSISTS")
        print("Try the troubleshooting steps above")

if __name__ == "__main__":
    asyncio.run(main())
