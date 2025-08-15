#!/usr/bin/env python3
"""
Enhanced WebSocket Connection Manager for Deriv
Handles multiple URLs, connection retries, and network issues
"""

import asyncio
import websockets
import json
import time
import traceback
from typing import Dict, Any, Optional, List
from config import DERIV_WS_URLS, DERIV_API_TOKEN

class DerivConnectionManager:
    """Enhanced connection manager with multiple fallbacks"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.websocket: Optional[Any] = None
        self.connected = False
        self.authorized = False
        self.current_url = ""
        self.connection_attempts = 0
        self.max_attempts = 5
        
    async def test_websocket_url(self, url: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Test a single WebSocket URL"""
        print(f"🔍 Testing: {url}")
        
        try:
            # Try connection with different strategies
            connection_configs = [
                # Strategy 1: Standard connection
                {
                    "ping_interval": 30,
                    "ping_timeout": 10,
                    "close_timeout": 10
                },
                # Strategy 2: Basic connection
                {
                    "ping_interval": None,
                    "ping_timeout": None
                },
                # Strategy 3: With headers
                {
                    "ping_interval": 20,
                    "ping_timeout": 5,
                    "additional_headers": {
                        "Origin": "https://app.deriv.com",
                        "User-Agent": "DerivTradingBot/1.0"
                    }
                }
            ]
            
            for i, config in enumerate(connection_configs):
                try:
                    print(f"  📡 Strategy {i+1}: {list(config.keys())}")
                    
                    websocket = await asyncio.wait_for(
                        websockets.connect(url, **config),
                        timeout=timeout
                    )
                    
                    print(f"  ✅ Connected with strategy {i+1}")
                    
                    # Test authorization
                    auth_result = await self.test_authorization(websocket)
                    await websocket.close()
                    
                    return {
                        "url": url,
                        "strategy": i+1,
                        "config": config,
                        "connection": True,
                        "authorization": auth_result["success"],
                        "error": auth_result.get("error"),
                        "details": auth_result.get("details", {})
                    }
                    
                except Exception as e:
                    print(f"  ❌ Strategy {i+1} failed: {e}")
                    continue
            
            return {
                "url": url,
                "connection": False,
                "authorization": False,
                "error": "All connection strategies failed"
            }
            
        except Exception as e:
            return {
                "url": url,
                "connection": False,
                "authorization": False,
                "error": str(e)
            }
    
    async def test_authorization(self, websocket: Any) -> Dict[str, Any]:
        """Test API token authorization"""
        try:
            auth_request = {
                "authorize": self.api_token,
                "req_id": int(time.time())
            }
            
            await websocket.send(json.dumps(auth_request))
            
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            data = json.loads(response)
            
            if "authorize" in data:
                auth_info = data["authorize"]
                return {
                    "success": True,
                    "details": {
                        "email": auth_info.get("email", "N/A"),
                        "currency": auth_info.get("currency", "N/A"),
                        "balance": auth_info.get("balance", "N/A"),
                        "loginid": auth_info.get("loginid", "N/A"),
                        "scopes": auth_info.get("scopes", [])
                    }
                }
            elif "error" in data:
                error = data["error"]
                return {
                    "success": False,
                    "error": f"{error.get('code', 'Unknown')}: {error.get('message', 'Unknown error')}"
                }
            else:
                return {
                    "success": False,
                    "error": "Unexpected response format"
                }
                
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Authorization timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def find_working_connection(self) -> Optional[Dict[str, Any]]:
        """Find a working WebSocket URL and configuration"""
        print("🔍 TESTING ALL WEBSOCKET CONNECTIONS")
        print("=" * 50)
        
        results = []
        
        for url in DERIV_WS_URLS:
            result = await self.test_websocket_url(url)
            results.append(result)
            
            if result.get("connection") and result.get("authorization"):
                print(f"✅ WORKING CONNECTION FOUND!")
                print(f"URL: {url}")
                print(f"Strategy: {result.get('strategy', 'N/A')}")
                if result.get("details"):
                    details = result["details"]
                    print(f"Account: {details.get('loginid', 'N/A')}")
                    print(f"Balance: {details.get('balance', 'N/A')} {details.get('currency', 'N/A')}")
                    print(f"Email: {details.get('email', 'N/A')}")
                return result
            
            print(f"❌ {url}: {result.get('error', 'Unknown error')}")
            
            # Brief pause between attempts
            await asyncio.sleep(1)
        
        print("\n❌ NO WORKING CONNECTIONS FOUND!")
        print("=" * 50)
        return None
    
    async def connect_with_config(self, url: str, config: Dict[str, Any]) -> bool:
        """Connect using a specific URL and configuration"""
        try:
            print(f"🔌 Connecting to: {url}")
            print(f"📋 Config: {config}")
            
            self.websocket = await websockets.connect(url, **config)
            self.connected = True
            self.current_url = url
            
            # Authorize the connection
            auth_result = await self.test_authorization(self.websocket)
            
            if auth_result["success"]:
                self.authorized = True
                print("✅ Connection and authorization successful!")
                
                if auth_result.get("details"):
                    details = auth_result["details"]
                    print(f"💰 Account: {details.get('loginid', 'N/A')}")
                    print(f"💵 Balance: {details.get('balance', 'N/A')} {details.get('currency', 'N/A')}")
                
                return True
            else:
                print(f"❌ Authorization failed: {auth_result.get('error')}")
                await self.websocket.close()
                self.connected = False
                return False
                
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
    async def auto_connect(self) -> bool:
        """Automatically find and connect to the best working URL"""
        working_config = await self.find_working_connection()
        
        if not working_config:
            return False
        
        # Extract connection details
        url = working_config["url"]
        config = working_config.get("config", {})
        
        # Connect using the working configuration
        return await self.connect_with_config(url, config)
    
    async def send(self, message: str) -> None:
        """Send message through WebSocket"""
        if self.websocket and self.connected:
            await self.websocket.send(message)
        else:
            raise ConnectionError("WebSocket not connected")
    
    async def recv(self) -> str:
        """Receive message from WebSocket"""
        if self.websocket and self.connected:
            return await self.websocket.recv()
        else:
            raise ConnectionError("WebSocket not connected")
    
    async def close(self) -> None:
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
        self.connected = False
        self.authorized = False

async def test_connection_manager():
    """Test the connection manager"""
    print("🚀 DERIV CONNECTION MANAGER TEST")
    print("=" * 60)
    
    manager = DerivConnectionManager(DERIV_API_TOKEN)
    
    try:
        success = await manager.auto_connect()
        
        if success:
            print("\n🎉 SUCCESS! Bot can connect to Deriv")
            print("✅ WebSocket connection: WORKING")
            print("✅ API token authorization: WORKING")
            print("✅ Ready for live trading!")
            
            # Test a simple tick subscription
            print("\n📊 Testing market data subscription...")
            tick_request = {
                "ticks": "R_75",
                "subscribe": 1,
                "req_id": 999
            }
            
            await manager.send(json.dumps(tick_request))
            response = await asyncio.wait_for(manager.recv(), timeout=5.0)
            data = json.loads(response)
            
            if "tick" in data:
                tick = data["tick"]
                print(f"✅ Market data: {tick.get('symbol')} = {tick.get('quote')}")
            elif "error" in data:
                print(f"⚠️ Market data error: {data['error'].get('message')}")
            
            await manager.close()
            return True
            
        else:
            print("\n❌ CONNECTION FAILED")
            print("🔧 TROUBLESHOOTING STEPS:")
            print("1. Check your internet connection")
            print("2. Try using a VPN or mobile hotspot")
            print("3. Check if your firewall blocks WebSocket connections")
            print("4. Verify your API token is valid at app.deriv.com")
            print("5. Make sure system clock is synchronized")
            
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_connection_manager())
