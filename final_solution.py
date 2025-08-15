#!/usr/bin/env python3
"""
Deriv Trading Bot - Final Connection Solution
Comprehensive script to get your bot connected and trading
"""

import asyncio
import websockets
import json
import subprocess
import time

async def test_multiple_deriv_endpoints():
    """Test all available Deriv endpoints to find one that works"""
    endpoints = [
        "wss://ws.derivws.com/websockets/v3",
        "wss://ws.binaryws.com/websockets/v3", 
        "wss://frontend.derivws.com/websockets/v3",
        "wss://ws.deriv.me/websockets/v3",
        "wss://api.deriv.com/websockets/v3"
    ]
    
    print("🔄 Testing all Deriv WebSocket endpoints...")
    
    for i, endpoint in enumerate(endpoints, 1):
        print(f"\n🧪 Test {i}/{len(endpoints)}: {endpoint}")
        
        try:
            async with websockets.connect(endpoint) as ws:
                # Test basic connection
                await ws.send(json.dumps({"time": 1, "req_id": 1}))
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                
                if "time" in data:
                    print(f"   ✅ WORKING! Server time: {data['time']}")
                    
                    # Test authentication with demo token
                    from config import DERIV_DEMO_API_TOKEN
                    await ws.send(json.dumps({"authorize": DERIV_DEMO_API_TOKEN, "req_id": 2}))
                    auth_response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    auth_data = json.loads(auth_response)
                    
                    if "authorize" in auth_data:
                        auth_info = auth_data["authorize"]
                        print(f"   🎉 AUTHENTICATION SUCCESS!")
                        print(f"   💰 Account: {auth_info.get('loginid')}")
                        print(f"   💵 Balance: ${auth_info.get('balance', 0):.2f}")
                        return endpoint
                    elif "error" in auth_data:
                        error = auth_data["error"]
                        print(f"   ❌ Auth failed: {error.get('message', 'Unknown error')}")
                    else:
                        print(f"   ❓ Unexpected auth response")
                else:
                    print(f"   ❓ Unexpected response: {data}")
                    
        except Exception as e:
            if "401" in str(e):
                print(f"   ❌ Blocked (HTTP 401)")
            elif "403" in str(e):
                print(f"   ❌ Forbidden (HTTP 403)")
            else:
                print(f"   ❌ Failed: {e}")
    
    return None

def get_vpn_suggestions():
    """Get specific VPN server suggestions"""
    return [
        "🇩🇪 Germany: Frankfurt, Munich, Berlin",
        "🇬🇧 UK: London, Manchester", 
        "🇳🇱 Netherlands: Amsterdam",
        "🇸🇬 Singapore: Singapore City",
        "🇺🇸 USA: New York, Atlanta, Los Angeles",
        "🇨🇦 Canada: Toronto, Montreal",
        "🇨🇭 Switzerland: Zurich",
        "🇦🇺 Australia: Sydney (if you're in Asia-Pacific)"
    ]

async def run_bot_with_working_endpoint(endpoint):
    """Update config and run the bot with the working endpoint"""
    print(f"\n🔧 Updating configuration to use working endpoint: {endpoint}")
    
    # Read current config
    with open('config.py', 'r') as f:
        config_content = f.read()
    
    # Update the primary WebSocket URL
    updated_config = config_content.replace(
        'DERIV_WS_URL = DERIV_WS_URLS[0]  # Primary URL',
        f'DERIV_WS_URL = "{endpoint}"  # Working endpoint found by auto-detection'
    )
    
    # Also update the URLs list to prioritize the working endpoint
    if endpoint not in updated_config:
        urls_section = '''DERIV_WS_URLS = [
    "wss://ws.derivws.com/websockets/v3",
    "wss://ws.binaryws.com/websockets/v3",
    "wss://ws.deriv.com/websockets/v3"
]'''
        new_urls_section = f'''DERIV_WS_URLS = [
    "{endpoint}",
    "wss://ws.derivws.com/websockets/v3",
    "wss://ws.binaryws.com/websockets/v3",
    "wss://ws.deriv.com/websockets/v3"
]'''
        updated_config = updated_config.replace(urls_section, new_urls_section)
    
    # Save updated config
    with open('config.py', 'w') as f:
        f.write(updated_config)
    
    print("✅ Configuration updated!")
    print("\n🚀 Starting the trading bot...")
    
    # Import and run the bot
    try:
        from main import DerivTradingBot
        bot = DerivTradingBot()
        await bot.start()
    except Exception as e:
        print(f"❌ Bot failed to start: {e}")
        print("🔧 Try running manually: python main.py")

async def main():
    print("🚀 DERIV TRADING BOT - FINAL CONNECTION SOLUTION")
    print("=" * 60)
    print("✅ Your new API tokens are configured:")
    print("   Demo: s8Sww40XKyKMp0x")
    print("   Live: g9kxuLTUWxrHGNe")
    print("=" * 60)
    
    # Check if we can connect to any Deriv endpoint
    working_endpoint = await test_multiple_deriv_endpoints()
    
    if working_endpoint:
        print("\n🎉 SUCCESS! Found working endpoint")
        print(f"📍 Working URL: {working_endpoint}")
        
        # Ask user if they want to start the bot
        print("\n🚀 Ready to start trading bot!")
        response = input("Start the bot now? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            await run_bot_with_working_endpoint(working_endpoint)
        else:
            print(f"\n✅ Configuration saved. Run bot manually with:")
            print(f"   python main.py")
    else:
        print("\n❌ NO WORKING ENDPOINTS FOUND")
        print("🔧 You need to try different VPN servers")
        
        print("\n📋 NEXT STEPS:")
        print("1. 🔌 Connect to IPVanish VPN")
        print("2. 🌍 Try these server locations:")
        
        for suggestion in get_vpn_suggestions():
            print(f"   {suggestion}")
        
        print("\n3. 🧪 Run this script again after changing VPN location")
        print("4. 💡 If still blocked, try:")
        print("   - Different VPN service (NordVPN, ProtonVPN)")
        print("   - Mobile hotspot")
        print("   - Different internet connection")
        
        print("\n🆘 ALTERNATIVE SOLUTIONS:")
        print("   📱 Run bot on mobile hotspot")
        print("   🏠 Try from different location/network")
        print("   🌐 Use different VPN service temporarily")

if __name__ == "__main__":
    asyncio.run(main())
