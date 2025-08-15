#!/usr/bin/env python3
"""
Quick Token Test Script
Tests the new API tokens to verify they work before running the main bot
"""

import asyncio
import websockets
import json
from config import DERIV_DEMO_API_TOKEN, DERIV_LIVE_API_TOKEN, DEMO_MODE

async def test_token(token, token_type):
    """Test a single API token"""
    print(f"\n🔐 Testing {token_type} token: {token[:8]}...")
    
    try:
        uri = "wss://ws.derivws.com/websockets/v3"
        async with websockets.connect(uri) as ws:
            # Send authorization request
            auth_request = {
                "authorize": token,
                "req_id": 1
            }
            
            await ws.send(json.dumps(auth_request))
            response = await asyncio.wait_for(ws.recv(), timeout=10.0)
            data = json.loads(response)
            
            if "authorize" in data:
                auth_info = data["authorize"]
                print(f"✅ {token_type} token AUTHORIZED successfully!")
                print(f"   📧 Account: {auth_info.get('email', 'N/A')}")
                print(f"   🆔 Login ID: {auth_info.get('loginid', 'N/A')}")
                print(f"   💰 Balance: ${auth_info.get('balance', 0):.2f}")
                print(f"   🏢 Company: {auth_info.get('company', 'N/A')}")
                return True
                
            elif "error" in data:
                error = data["error"]
                print(f"❌ {token_type} token FAILED: {error.get('code', 'Unknown')} - {error.get('message', 'Unknown error')}")
                return False
            else:
                print(f"❓ {token_type} token - Unexpected response: {data}")
                return False
                
    except asyncio.TimeoutError:
        print(f"⏰ {token_type} token test TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {token_type} token test ERROR: {e}")
        return False

async def main():
    """Test both tokens"""
    print("🧪 NEW API TOKEN VERIFICATION TEST")
    print("=" * 50)
    print(f"📅 Testing at: {asyncio.get_event_loop().time()}")
    print(f"🎯 Demo Mode: {'YES' if DEMO_MODE else 'NO'}")
    
    # Test tokens
    results = []
    
    # Test demo token
    if DERIV_DEMO_API_TOKEN:
        demo_result = await test_token(DERIV_DEMO_API_TOKEN, "DEMO")
        results.append(("DEMO", demo_result))
    else:
        print("⚠️ No DEMO token configured")
        results.append(("DEMO", False))
    
    # Test live token
    if DERIV_LIVE_API_TOKEN:
        live_result = await test_token(DERIV_LIVE_API_TOKEN, "LIVE")
        results.append(("LIVE", live_result))
    else:
        print("⚠️ No LIVE token configured")
        results.append(("LIVE", False))
    
    # Summary
    print("\n📊 TOKEN TEST RESULTS")
    print("=" * 50)
    
    working_tokens = 0
    for token_type, result in results:
        status = "✅ WORKING" if result else "❌ FAILED"
        print(f"   {token_type}: {status}")
        if result:
            working_tokens += 1
    
    print(f"\n🎯 Working tokens: {working_tokens}/2")
    
    if working_tokens > 0:
        current_token_type = "DEMO" if DEMO_MODE else "LIVE"
        current_token_working = any(r[1] for r in results if r[0] == current_token_type)
        
        if current_token_working:
            print(f"✅ READY TO TRADE: {current_token_type} mode is configured and working!")
            print("🚀 You can now run: python main.py")
        else:
            print(f"⚠️ WARNING: {current_token_type} mode is configured but token failed!")
            other_mode = "LIVE" if DEMO_MODE else "DEMO"
            other_working = any(r[1] for r in results if r[0] == other_mode)
            if other_working:
                print(f"💡 Consider switching to {other_mode} mode in config.py")
    else:
        print("🚨 CRITICAL: No working tokens! Please check your API tokens at app.deriv.com")
        print("🔧 Make sure tokens have 'read' and 'trade' permissions enabled")

if __name__ == "__main__":
    asyncio.run(main())
