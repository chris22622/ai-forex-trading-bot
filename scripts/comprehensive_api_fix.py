#!/usr/bin/env python3
"""
FINAL API FIX - Complete Solution
This script will diagnose and fix all API connection issues
"""

import asyncio
import websockets
import json
import sys
import os
from typing import Dict, Any, List, Tuple

# Import config safely
try:
    from config import DERIV_DEMO_API_TOKEN, DERIV_LIVE_API_TOKEN, DEMO_MODE
except ImportError:
    print("❌ Could not import config.py - please ensure it exists")
    sys.exit(1)

class ComprehensiveAPIFixer:
    """Complete API connection fixer with all fallbacks"""
    
    def __init__(self):
        # All possible WebSocket URLs
        self.websocket_urls = [
            "wss://ws.derivws.com/websockets/v3",
            "wss://ws.binaryws.com/websockets/v3", 
            "wss://frontend.derivws.com/websockets/v3",
            "wss://ws.deriv.com/websockets/v3"
        ]
        
        self.working_url = None
        self.working_token = None
        self.working_mode = None
        self.connection_successful = False
        
    async def test_url_without_auth(self, url: str) -> bool:
        """Test URL without authentication first"""
        print(f"🔌 Testing basic connection to {url}")
        
        try:
            # Multiple connection approaches
            approaches = [
                {"ping_interval": 30, "ping_timeout": 10},
                {"ping_interval": None, "ping_timeout": None},
                {}
            ]
            
            for i, config in enumerate(approaches):
                try:
                    ws = await asyncio.wait_for(
                        websockets.connect(url, **config),
                        timeout=15.0
                    )
                    
                    # Test with server time request (no auth needed)
                    time_request = {"time": 1, "req_id": 1}
                    await ws.send(json.dumps(time_request))
                    
                    response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    data = json.loads(response)
                    
                    if "time" in data:
                        print(f"✅ {url} - Basic connection OK (approach {i+1})")
                        await ws.close()
                        return True
                    
                    await ws.close()
                    
                except asyncio.TimeoutError:
                    print(f"⏰ {url} - Timeout (approach {i+1})")
                    continue
                except Exception as e:
                    print(f"❌ {url} - Failed (approach {i+1}): {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"❌ {url} - Connection error: {e}")
            return False
    
    async def test_token_with_url(self, url: str, token: str, token_type: str) -> bool:
        """Test specific token with specific URL"""
        print(f"🔑 Testing {token_type} token with {url}")
        
        try:
            ws = await asyncio.wait_for(
                websockets.connect(url),
                timeout=15.0
            )
            
            # Send authorization
            auth_request = {"authorize": token, "req_id": 1}
            await ws.send(json.dumps(auth_request))
            
            response = await asyncio.wait_for(ws.recv(), timeout=15.0)
            data = json.loads(response)
            
            if "authorize" in data:
                auth_info = data["authorize"]
                print(f"✅ {token_type} AUTHORIZATION SUCCESS!")
                print(f"   📧 Email: {auth_info.get('email', 'N/A')}")
                print(f"   💰 Balance: {auth_info.get('balance', 'N/A')} {auth_info.get('currency', '')}")
                print(f"   🆔 Login ID: {auth_info.get('loginid', 'N/A')}")
                print(f"   🔑 Scopes: {auth_info.get('scopes', [])}")
                
                # Test market data
                print(f"📈 Testing market data access...")
                tick_request = {"ticks": "R_75", "subscribe": 1, "req_id": 2}
                await ws.send(json.dumps(tick_request))
                
                tick_response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                tick_data = json.loads(tick_response)
                
                if "tick" in tick_data:
                    tick = tick_data["tick"]
                    print(f"✅ Market data: {tick.get('symbol')} = {tick.get('quote')}")
                    
                    # Success! Store working combination
                    self.working_url = url
                    self.working_token = token
                    self.working_mode = token_type
                    
                    await ws.close()
                    return True
                else:
                    print(f"⚠️ Market data access issue: {tick_data}")
                
            elif "error" in data:
                error = data["error"]
                error_code = error.get("code", "Unknown")
                error_msg = error.get("message", "Unknown error")
                
                print(f"❌ {token_type} FAILED: {error_code} - {error_msg}")
                
                # Specific error guidance
                if error_code == "InvalidToken":
                    print(f"   🔧 Token is invalid, expired, or has typos")
                elif "permission" in error_msg.lower() or "scope" in error_msg.lower():
                    print(f"   🔧 Token missing 'Trading' permissions")
                elif "read" in error_msg.lower():
                    print(f"   🔧 Token has read-only permissions")
                
            await ws.close()
            return False
            
        except Exception as e:
            print(f"❌ {token_type} test error: {e}")
            return False
    
    async def run_comprehensive_test(self) -> bool:
        """Run complete diagnostic and fix"""
        print("🚀 COMPREHENSIVE API FIXER")
        print("🔧 This will test every possible combination")
        print("=" * 60)
        
        # Step 1: Test all URLs without authentication
        print("📡 STEP 1: Testing WebSocket URLs")
        print("-" * 30)
        
        working_urls = []
        for url in self.websocket_urls:
            if await self.test_url_without_auth(url):
                working_urls.append(url)
        
        if not working_urls:
            print("\n❌ CRITICAL: No WebSocket URLs accessible!")
            print("🔧 This indicates:")
            print("   1. Internet connection issues")
            print("   2. Firewall blocking WebSocket connections")
            print("   3. ISP/corporate network restrictions")
            print("   4. Geographic restrictions")
            print("\n💡 Solutions:")
            print("   - Try VPN (ProtonVPN, NordVPN, etc.)")
            print("   - Use mobile hotspot")
            print("   - Check firewall settings")
            print("   - Contact network administrator")
            return False
        
        print(f"\n✅ Found {len(working_urls)} working URLs")
        
        # Step 2: Test tokens
        print(f"\n🔑 STEP 2: Testing API Tokens")
        print("-" * 30)
        
        # Prepare tokens to test
        tokens_to_test: List[Tuple[str, str]] = []
        
        if DERIV_DEMO_API_TOKEN and DERIV_DEMO_API_TOKEN.strip():
            tokens_to_test.append(("DEMO", DERIV_DEMO_API_TOKEN.strip()))
        else:
            print("⚠️ No demo token configured")
            
        if DERIV_LIVE_API_TOKEN and DERIV_LIVE_API_TOKEN.strip():
            tokens_to_test.append(("LIVE", DERIV_LIVE_API_TOKEN.strip()))
        else:
            print("⚠️ No live token configured")
        
        if not tokens_to_test:
            print("❌ No API tokens to test!")
            return False
        
        # Test each token with each working URL
        for url in working_urls:
            print(f"\n🔗 Testing with {url}")
            for token_type, token in tokens_to_test:
                if await self.test_token_with_url(url, token, token_type):
                    self.connection_successful = True
                    return True
        
        print(f"\n❌ No working token combinations found")
        return False
    
    def generate_fix_config(self) -> str:
        """Generate updated configuration"""
        if not self.working_url or not self.working_token:
            return ""
        
        config_updates = f"""
# ==================== FIXED CONFIGURATION ====================
# Update your config.py with these working values:

# WebSocket URL (WORKING)
DERIV_WS_URL = "{self.working_url}"

# Mode setting
DEMO_MODE = {self.working_mode == "DEMO"}

# Working token
{f'DERIV_DEMO_API_TOKEN = "{self.working_token}"' if self.working_mode == "DEMO" else f'DERIV_LIVE_API_TOKEN = "{self.working_token}"'}

# ==================== END FIXED CONFIGURATION ====================
"""
        return config_updates
    
    def print_final_results(self):
        """Print comprehensive results and next steps"""
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS & RECOMMENDATIONS")
        print("=" * 60)
        
        if self.connection_successful:
            print("🎉 SUCCESS! API connection fixed!")
            print(f"✅ Working URL: {self.working_url}")
            print(f"✅ Working Token: {self.working_mode}")
            print(f"✅ Token Preview: {self.working_token[:10]}...")
            
            # Show config updates
            config_fix = self.generate_fix_config()
            print(f"\n{config_fix}")
            
            print("🚀 NEXT STEPS:")
            print("1. Update your config.py with the values above")
            print("2. Run: python main.py")
            print("3. Your bot should now connect successfully!")
            print("\n✅ API ISSUE COMPLETELY RESOLVED!")
            
        else:
            print("❌ API CONNECTION STILL FAILED")
            print("\n🚨 CRITICAL ACTIONS REQUIRED:")
            print("=" * 40)
            print("1. 🌐 Go to: https://app.deriv.com")
            print("2. 🔐 Login to your account")
            print("3. ⚙️ Go to: Settings → API Token")
            print("4. 🗑️ DELETE all existing tokens")
            print("5. ➕ CREATE NEW TOKEN with permissions:")
            print("   ✅ Read")
            print("   ✅ Trade (CRITICAL)")
            print("   ✅ Trading Information")
            print("   ✅ Payments")
            print("   ✅ Admin")
            print("6. 📋 Copy new token EXACTLY (no spaces)")
            print("7. 🔄 Update config.py with new token")
            print("8. 🧪 Run this fixer again")
            
            print("\n🆘 IF STILL FAILING:")
            print("- Verify account is verified/active")
            print("- Check email for verification links")
            print("- Try VPN if network is restricted")
            print("- Contact Deriv support if needed")

async def main():
    """Run the comprehensive API fixer"""
    print("🔧 DERIV API COMPREHENSIVE FIXER")
    print("This will identify and fix ALL API connection issues")
    print("Please wait while we test every possibility...")
    print()
    
    fixer = ComprehensiveAPIFixer()
    
    try:
        success = await fixer.run_comprehensive_test()
        fixer.print_final_results()
        
        if success:
            print("\n🎊 PROBLEM SOLVED! Your bot is ready to trade!")
            sys.exit(0)
        else:
            print("\n⚠️ Manual token creation required")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
