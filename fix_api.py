#!/usr/bin/env python3
"""
API Connection Fixer for Deriv Trading Bot
Comprehensive API token testing and troubleshooting
"""

import asyncio
import websockets
import json
import traceback
from typing import Dict, Any, Optional
from config import DERIV_LIVE_API_TOKEN, DERIV_DEMO_API_TOKEN, DERIV_WS_URL

class DerivAPIFixer:
    """Comprehensive API connection troubleshooter"""
    
    def __init__(self):
        self.tokens = {
            "demo": DERIV_DEMO_API_TOKEN,
            "live": DERIV_LIVE_API_TOKEN
        }
        self.working_token = None
        self.working_mode = None
    
    async def test_connection_methods(self) -> Optional[Any]:
        """Test different WebSocket connection methods"""
        print("🔌 Testing WebSocket connection methods...")
        
        # Method 1: Standard connection
        try:
            print("📡 Method 1: Standard connection...")
            websocket = await websockets.connect(
                DERIV_WS_URL,
                ping_interval=30,
                ping_timeout=10
            )
            print("✅ Standard connection successful")
            return websocket
        except Exception as e:
            print(f"❌ Standard connection failed: {e}")
        
        # Method 2: With custom headers
        try:
            print("📡 Method 2: Connection with custom headers...")
            websocket = await websockets.connect(
                DERIV_WS_URL,
                ping_interval=30,
                ping_timeout=10,
                additional_headers={
                    "Origin": "https://app.deriv.com",
                    "User-Agent": "DerivBot/1.0"
                }
            )
            print("✅ Custom headers connection successful")
            return websocket
        except Exception as e:
            print(f"❌ Custom headers connection failed: {e}")
        
        # Method 3: Minimal connection
        try:
            print("📡 Method 3: Minimal connection...")
            websocket = await websockets.connect(DERIV_WS_URL)
            print("✅ Minimal connection successful")
            return websocket
        except Exception as e:
            print(f"❌ Minimal connection failed: {e}")
        
        print("❌ All connection methods failed!")
        return None
    
    async def test_token(self, token: str, mode: str) -> Dict[str, Any]:
        """Test a specific API token"""
        print(f"\n🔑 Testing {mode.upper()} token: {token}")
        print("-" * 50)
        
        result = {
            "mode": mode,
            "token": token,
            "connection": False,
            "authorization": False,
            "market_data": False,
            "error": None,
            "details": {}
        }
        
        try:
            # Test connection
            websocket = await self.test_connection_methods()
            if not websocket:
                result["error"] = "Could not establish WebSocket connection"
                return result
            
            result["connection"] = True
            
            # Test authorization
            print("🔐 Testing authorization...")
            auth_request: Dict[str, Any] = {
                "authorize": token,
                "req_id": 1
            }
            
            await websocket.send(json.dumps(auth_request))
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data: Dict[str, Any] = json.loads(response)
                
                if "authorize" in data:
                    result["authorization"] = True
                    auth_info = data["authorize"]
                    
                    result["details"] = {
                        "email": auth_info.get("email", "N/A"),
                        "currency": auth_info.get("currency", "N/A"),
                        "balance": auth_info.get("balance", "N/A"),
                        "loginid": auth_info.get("loginid", "N/A"),
                        "country": auth_info.get("country", "N/A"),
                        "scopes": auth_info.get("scopes", [])
                    }
                    
                    print("✅ Authorization successful!")
                    print(f"📧 Email: {result['details']['email']}")
                    print(f"💰 Balance: {result['details']['balance']} {result['details']['currency']}")
                    print(f"🆔 Login ID: {result['details']['loginid']}")
                    print(f"🔑 Scopes: {result['details']['scopes']}")
                    
                    # Test market data access
                    print("📈 Testing market data access...")
                    tick_request: Dict[str, Any] = {
                        "ticks": "R_75",
                        "subscribe": 1,
                        "req_id": 2
                    }
                    
                    await websocket.send(json.dumps(tick_request))
                    tick_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    tick_data: Dict[str, Any] = json.loads(tick_response)
                    
                    if "tick" in tick_data:
                        result["market_data"] = True
                        tick = tick_data["tick"]
                        print(f"✅ Market data access successful!")
                        print(f"📊 Symbol: {tick.get('symbol')}")
                        print(f"💲 Price: {tick.get('quote')}")
                    elif "error" in tick_data:
                        print(f"❌ Market data error: {tick_data['error'].get('message')}")
                    
                elif "error" in data:
                    error = data["error"]
                    result["error"] = f"{error.get('code')}: {error.get('message')}"
                    print(f"❌ Authorization failed: {result['error']}")
                    
                    # Provide specific error guidance
                    self.provide_error_guidance(error)
                
            except asyncio.TimeoutError:
                result["error"] = "Authorization request timed out"
                print("⏰ Authorization request timed out")
            
            await websocket.close()
            
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ Connection error: {e}")
            traceback.print_exc()
        
        return result
    
    def provide_error_guidance(self, error: Dict[str, Any]) -> None:
        """Provide specific guidance based on error type"""
        error_code = error.get("code", "")
        error_message = error.get("message", "")
        
        print(f"\n🔧 TROUBLESHOOTING GUIDANCE:")
        print("=" * 50)
        
        if error_code == "InvalidToken":
            print("❌ INVALID TOKEN ERROR")
            print("Possible causes:")
            print("1. Token is incorrect or has typos")
            print("2. Token has expired")
            print("3. Token was revoked")
            print("4. Wrong account type (demo token for live account)")
            print("\n💡 Solutions:")
            print("1. Go to app.deriv.com → API Token")
            print("2. Delete old tokens and create a new one")
            print("3. Make sure to select 'Trading' permissions")
            print("4. Copy the token exactly (no extra spaces)")
            
        elif "read" in error_message.lower():
            print("❌ INSUFFICIENT PERMISSIONS ERROR")
            print("The token doesn't have trading permissions")
            print("\n💡 Solutions:")
            print("1. Create a new API token")
            print("2. Select 'Trading' permissions when creating")
            print("3. Enable 'Admin' if you need account management")
            
        elif "expired" in error_message.lower():
            print("❌ TOKEN EXPIRED ERROR")
            print("\n💡 Solutions:")
            print("1. Create a new API token")
            print("2. API tokens can expire after extended periods")
            
        else:
            print(f"❌ UNKNOWN ERROR: {error_code}")
            print(f"Message: {error_message}")
            print("\n💡 General solutions:")
            print("1. Check account verification status")
            print("2. Ensure account has sufficient balance")
            print("3. Verify country restrictions")
            print("4. Contact Deriv support if issue persists")
    
    async def test_all_tokens(self) -> Dict[str, Dict[str, Any]]:
        """Test all available tokens"""
        print("🚀 DERIV API CONNECTION FIXER")
        print("=" * 60)
        print(f"WebSocket URL: {DERIV_WS_URL}")
        print("=" * 60)
        
        results = {}
        
        for mode, token in self.tokens.items():
            if token and token.strip():
                result = await self.test_token(token, mode)
                results[mode] = result
                
                # Track working token
                if result["authorization"] and result["market_data"]:
                    self.working_token = token
                    self.working_mode = mode
                    print(f"🎉 WORKING TOKEN FOUND: {mode.upper()}")
            else:
                print(f"\n⚠️ {mode.upper()} token is empty or missing")
                results[mode] = {
                    "mode": mode,
                    "error": "Token not configured",
                    "connection": False,
                    "authorization": False,
                    "market_data": False
                }
        
        return results
    
    def generate_fix_recommendations(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Generate fix recommendations based on test results"""
        print("\n" + "=" * 60)
        print("🔧 FIX RECOMMENDATIONS")
        print("=" * 60)
        
        working_tokens = [mode for mode, result in results.items() 
                         if result.get("authorization") and result.get("market_data")]
        
        if working_tokens:
            print(f"✅ WORKING TOKENS FOUND: {', '.join(working_tokens).upper()}")
            print(f"\n🎯 RECOMMENDED ACTION:")
            print(f"1. Use the {working_tokens[0].upper()} token for trading")
            print(f"2. Update config.py to use the working token")
            
            if len(working_tokens) > 1:
                print(f"3. You have multiple working tokens - choose based on your needs:")
                print(f"   - DEMO: For testing and practice")
                print(f"   - LIVE: For real trading with real money")
            
        else:
            print("❌ NO WORKING TOKENS FOUND")
            print("\n🚨 CRITICAL ACTIONS NEEDED:")
            print("1. Go to app.deriv.com")
            print("2. Log into your account")
            print("3. Navigate to 'API Token' section")
            print("4. Delete existing tokens")
            print("5. Create NEW tokens with these settings:")
            print("   ✅ Trading: YES")
            print("   ✅ Payments: NO (unless needed)")
            print("   ✅ Admin: YES (recommended)")
            print("6. Copy the new tokens carefully")
            print("7. Update config.py with new tokens")
            print("\n⚠️ IMPORTANT CHECKS:")
            print("- Account must be verified")
            print("- Account must have sufficient balance")
            print("- No country restrictions")
            print("- Account not suspended/restricted")
    
    async def create_fixed_config(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Create a fixed configuration file"""
        print("\n📝 Creating fixed configuration...")
        
        if self.working_token and self.working_mode:
            # Read current config
            with open('config.py', 'r') as f:
                config_content = f.read()
            
            # Update to use working token
            if self.working_mode == "demo":
                new_content = config_content.replace(
                    'DEMO_MODE = True',
                    'DEMO_MODE = True  # ✅ USING WORKING DEMO TOKEN'
                ).replace(
                    'DEMO_MODE = False',
                    'DEMO_MODE = True  # ✅ SWITCHED TO WORKING DEMO TOKEN'
                )
            else:
                new_content = config_content.replace(
                    'DEMO_MODE = True',
                    'DEMO_MODE = False  # ✅ USING WORKING LIVE TOKEN'
                ).replace(
                    'DEMO_MODE = False',
                    'DEMO_MODE = False  # ✅ USING WORKING LIVE TOKEN'
                )
            
            # Save fixed config
            with open('config_fixed.py', 'w') as f:
                f.write(new_content)
            
            print(f"✅ Fixed configuration saved to config_fixed.py")
            print(f"🎯 Mode set to: {self.working_mode.upper()}")
            print("\n📋 TO APPLY THE FIX:")
            print("1. Review config_fixed.py")
            print("2. Backup your current config.py")
            print("3. Replace config.py with config_fixed.py")
            print("4. Run the bot again")

async def main():
    """Main function"""
    fixer = DerivAPIFixer()
    
    try:
        # Test all tokens
        results = await fixer.test_all_tokens()
        
        # Generate recommendations
        fixer.generate_fix_recommendations(results)
        
        # Create fixed config if possible
        await fixer.create_fixed_config(results)
        
        print("\n" + "=" * 60)
        print("🏁 API TESTING COMPLETE")
        print("=" * 60)
        
        # Summary
        for mode, result in results.items():
            status = "✅ WORKING" if (result.get("authorization") and result.get("market_data")) else "❌ FAILED"
            print(f"{mode.upper()} Token: {status}")
            if result.get("error"):
                print(f"  Error: {result['error']}")
        
    except Exception as e:
        print(f"\n❌ Critical error in API fixer: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
