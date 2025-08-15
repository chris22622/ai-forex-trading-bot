#!/usr/bin/env python3
"""
Advanced WebSocket Connection Test
Tests various connection methods to diagnose the HTTP 401 issue
"""

import asyncio
import websockets
import json

async def test_connection_with_headers():
    """Test connection with various headers"""
    print("🔄 Testing connection with custom headers...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Origin": "https://app.deriv.com",
        "Sec-WebSocket-Protocol": "binary"
    }
    
    try:
        uri = "wss://ws.derivws.com/websockets/v3"
        async with websockets.connect(uri, extra_headers=headers) as ws:
            print("✅ Connected with headers!")
            
            # Test server time
            await ws.send(json.dumps({"time": 1, "req_id": 1}))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(response)
            print(f"Response: {data}")
            return True
            
    except Exception as e:
        print(f"❌ Failed with headers: {e}")
        return False

async def test_alternative_urls():
    """Test alternative WebSocket URLs"""
    urls = [
        "wss://ws.derivws.com/websockets/v3",
        "wss://ws.binaryws.com/websockets/v3", 
        "wss://frontend.derivws.com/websockets/v3",
        "wss://ws.deriv.com/websockets/v3"
    ]
    
    for i, url in enumerate(urls, 1):
        print(f"\n🔄 Test {i}/4: {url}")
        try:
            async with websockets.connect(url) as ws:
                print("   ✅ Connected successfully!")
                
                # Test basic request
                await ws.send(json.dumps({"time": 1, "req_id": 1}))
                response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                data = json.loads(response)
                
                if "time" in data:
                    print(f"   ✅ Got server time: {data['time']}")
                    return url
                else:
                    print(f"   ❓ Unexpected response: {data}")
                    
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    return None

async def test_auth_with_token(working_url, token):
    """Test authentication with a working URL"""
    print(f"\n🔐 Testing auth with token on {working_url}")
    
    try:
        async with websockets.connect(working_url) as ws:
            # Send auth request
            auth_request = {
                "authorize": token,
                "req_id": 1
            }
            
            await ws.send(json.dumps(auth_request))
            response = await asyncio.wait_for(ws.recv(), timeout=10.0)
            data = json.loads(response)
            
            print(f"Auth response: {json.dumps(data, indent=2)}")
            
            if "authorize" in data:
                print("✅ Authentication successful!")
                return True
            elif "error" in data:
                error = data["error"]
                print(f"❌ Auth failed: {error.get('code')} - {error.get('message')}")
                
                # Specific error handling
                if error.get('code') == 'InvalidToken':
                    print("🔧 Token is invalid - create a new one at app.deriv.com")
                elif 'permission' in error.get('message', '').lower():
                    print("🔧 Token lacks required permissions - enable 'read' and 'trade'")
                else:
                    print("🔧 Unknown auth error - try creating new tokens")
                    
                return False
            else:
                print(f"❓ Unexpected auth response")
                return False
                
    except Exception as e:
        print(f"❌ Auth test failed: {e}")
        return False

async def main():
    print("🧪 ADVANCED DERIV CONNECTION TEST")
    print("=" * 50)
    
    # Test with headers first
    headers_work = await test_connection_with_headers()
    
    if not headers_work:
        print("\n🔄 Trying alternative URLs...")
        working_url = await test_alternative_urls()
        
        if working_url:
            print(f"\n✅ Found working URL: {working_url}")
            
            # Test authentication
            from config import DERIV_DEMO_API_TOKEN, DERIV_LIVE_API_TOKEN
            
            # Test demo token
            if DERIV_DEMO_API_TOKEN:
                print(f"\n🔐 Testing DEMO token...")
                demo_works = await test_auth_with_token(working_url, DERIV_DEMO_API_TOKEN)
            
            # Test live token
            if DERIV_LIVE_API_TOKEN:
                print(f"\n🔐 Testing LIVE token...")
                live_works = await test_auth_with_token(working_url, DERIV_LIVE_API_TOKEN)
        else:
            print("\n❌ No working WebSocket URLs found!")
            print("🔧 This suggests a network connectivity issue")
            print("💡 Try connecting to a VPN and run the test again")
    
    print("\n🎯 SUMMARY")
    print("=" * 50)
    print("If no URLs work, the issue is network connectivity")
    print("If URLs work but auth fails, the issue is with API tokens")
    print("Check app.deriv.com -> API Token to verify your tokens")

if __name__ == "__main__":
    asyncio.run(main())
