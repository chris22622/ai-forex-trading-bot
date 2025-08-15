#!/usr/bin/env python3
"""
Quick WebSocket Connection Test - No Authentication Required
Tests if WebSocket connection works before trying API tokens
"""

import asyncio
import websockets
import json
import time

async def test_basic_connection():
    """Test basic WebSocket connection without authentication"""
    print("🔌 TESTING BASIC WEBSOCKET CONNECTION")
    print("=" * 50)
    
    urls = [
        "wss://ws.derivws.com/websockets/v3",
        "wss://ws.binaryws.com/websockets/v3"
    ]
    
    for url in urls:
        print(f"\n📡 Testing: {url}")
        
        try:
            # Simple connection test
            websocket = await asyncio.wait_for(
                websockets.connect(url),
                timeout=10.0
            )
            
            print("✅ WebSocket connection successful!")
            
            # Try to get server time (no auth needed)
            time_request = {
                "time": 1,
                "req_id": 1
            }
            
            await websocket.send(json.dumps(time_request))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            
            if "time" in data:
                server_time = data["time"]
                print(f"✅ Server time: {server_time}")
                print(f"✅ Server communication: WORKING")
            else:
                print(f"📨 Server response: {data}")
            
            await websocket.close()
            
            print(f"✅ {url}: CONNECTION WORKS!")
            return url
            
        except Exception as e:
            print(f"❌ {url}: {e}")
    
    print("\n❌ No working WebSocket URLs found")
    return None

async def test_with_demo_token():
    """Test with a known working demo token"""
    print("\n🔑 TESTING WITH DEMO TOKEN")
    print("=" * 50)
    
    # Try to get a working URL first
    working_url = await test_basic_connection()
    
    if not working_url:
        print("❌ Cannot proceed - no working WebSocket connection")
        return False
    
    print(f"\n🔌 Using working URL: {working_url}")
    
    # Test with current tokens
    from config import DERIV_DEMO_API_TOKEN, DERIV_LIVE_API_TOKEN
    
    tokens_to_test = [
        ("DEMO", DERIV_DEMO_API_TOKEN),
        ("LIVE", DERIV_LIVE_API_TOKEN)
    ]
    
    for token_type, token in tokens_to_test:
        if not token or token.strip() == "":
            print(f"⚠️ {token_type} token is empty")
            continue
            
        print(f"\n🔑 Testing {token_type} token: {token}")
        
        try:
            websocket = await websockets.connect(working_url)
            
            auth_request = {
                "authorize": token,
                "req_id": int(time.time())
            }
            
            await websocket.send(json.dumps(auth_request))
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            data = json.loads(response)
            
            if "authorize" in data:
                auth_info = data["authorize"]
                print(f"✅ {token_type} TOKEN WORKS!")
                print(f"   Account: {auth_info.get('loginid', 'N/A')}")
                print(f"   Balance: {auth_info.get('balance', 'N/A')} {auth_info.get('currency', 'N/A')}")
                print(f"   Email: {auth_info.get('email', 'N/A')}")
                await websocket.close()
                return True
                
            elif "error" in data:
                error = data["error"]
                print(f"❌ {token_type} TOKEN FAILED")
                print(f"   Error: {error.get('code', 'Unknown')} - {error.get('message', 'Unknown')}")
                
                # Provide specific guidance
                if error.get('code') == 'InvalidToken':
                    print(f"   🔧 Token '{token}' is invalid or expired")
                    print(f"   💡 Create new token at: https://app.deriv.com/account/api-token")
                
            await websocket.close()
            
        except Exception as e:
            print(f"❌ {token_type} test failed: {e}")
    
    return False

async def generate_fix_instructions():
    """Generate specific fix instructions"""
    print("\n" + "="*60)
    print("🛠️ API TOKEN FIX INSTRUCTIONS")
    print("="*60)
    
    print("\n🎯 STEP-BY-STEP FIX:")
    print("1. Go to: https://app.deriv.com")
    print("2. Log into your account")
    print("3. Go to Settings → API Token")
    print("4. DELETE all existing tokens")
    print("5. CREATE NEW TOKEN with these permissions:")
    print("   ✅ Read")
    print("   ✅ Trade (REQUIRED)")
    print("   ✅ Trading Information")
    print("   ✅ Payments")
    print("   ✅ Admin")
    
    print("\n6. COPY the new token EXACTLY")
    print("7. UPDATE config.py:")
    print("   DERIV_DEMO_API_TOKEN = \"YOUR_NEW_TOKEN_HERE\"")
    
    print("\n8. RUN this test again:")
    print("   python quick_fix_test.py")
    
    print("\n💡 COMMON ISSUES:")
    print("- Token copied with extra spaces")
    print("- Missing 'Trade' permission")
    print("- Account not verified")
    print("- Using wrong account type (demo vs live)")
    
    print(f"\n🚀 Once fixed, your bot will work perfectly!")

async def main():
    """Main test function"""
    print("🚀 QUICK WEBSOCKET & API TEST")
    print("This will tell us exactly what's wrong")
    print("=" * 60)
    
    # Test basic connection first
    working = await test_with_demo_token()
    
    if not working:
        await generate_fix_instructions()
    else:
        print("\n🎉 SUCCESS! Your tokens work!")
        print("✅ The bot should work now")
        print("🚀 Try running: python main.py")

if __name__ == "__main__":
    asyncio.run(main())
