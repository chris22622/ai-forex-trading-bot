"""
Advanced Deriv Connection Test with Headers and Different Approaches
"""
import asyncio
import websockets
import json
import requests
from datetime import datetime

# Your tokens
DEMO_TOKEN = "s8Sww40XKyKMp0x"
LIVE_TOKEN = "g9kxuLTUWxrHGNe"

class AdvancedDerivTest:
    def __init__(self):
        self.urls = [
            "wss://ws.derivws.com/websockets/v3",
            "wss://ws.binaryws.com/websockets/v3",
            "wss://ws.derivws.com/websockets/v1",
            "wss://ws.binaryws.com/websockets/v1"
        ]
    
    async def test_with_headers(self, url):
        """Test connection with proper headers"""
        try:
            print(f"🔌 Testing with headers: {url}")
            
            # Add headers that Deriv might expect
            headers = {
                "Origin": "https://app.deriv.com",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Sec-WebSocket-Version": "13",
                "Cache-Control": "no-cache"
            }
            
            async with websockets.connect(url, extra_headers=headers) as websocket:
                print(f"✅ Connected with headers to {url}")
                
                # Test authorization immediately
                auth_request = {
                    "authorize": DEMO_TOKEN,
                    "req_id": 1
                }
                
                await websocket.send(json.dumps(auth_request))
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)
                
                print(f"📥 Auth response: {json.dumps(data, indent=2)}")
                
                if "authorize" in data:
                    print(f"✅ SUCCESS! Connection + Auth working!")
                    return True
                elif "error" in data:
                    error = data["error"]
                    print(f"❌ Auth failed: {error.get('message', 'Unknown')}")
                    return False
                    
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    def test_http_api(self):
        """Test HTTP API endpoint"""
        try:
            print(f"\n🌐 TESTING HTTP API")
            print("-" * 25)
            
            # Try different HTTP endpoints
            endpoints = [
                "https://api.deriv.com/v1/authorize",
                "https://app.deriv.com/api/v1/authorize",
                "https://binary.com/api/v1/authorize"
            ]
            
            for endpoint in endpoints:
                try:
                    print(f"🌐 Testing: {endpoint}")
                    
                    headers = {
                        "Content-Type": "application/json",
                        "Origin": "https://app.deriv.com",
                        "User-Agent": "TradingBot/1.0"
                    }
                    
                    data = {"authorize": DEMO_TOKEN}
                    
                    response = requests.post(endpoint, json=data, headers=headers, timeout=10)
                    print(f"   Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"   ✅ SUCCESS: {result}")
                        return True
                    else:
                        print(f"   ❌ HTTP {response.status_code}: {response.text[:200]}")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
            return False
            
        except Exception as e:
            print(f"❌ HTTP API test failed: {e}")
            return False
    
    def check_deriv_status(self):
        """Check if Deriv API is operational"""
        try:
            print(f"\n🏥 CHECKING DERIV STATUS")
            print("-" * 25)
            
            # Check main site
            response = requests.get("https://deriv.com", timeout=10)
            print(f"Deriv.com: {response.status_code}")
            
            # Check app
            response = requests.get("https://app.deriv.com", timeout=10)
            print(f"App.deriv.com: {response.status_code}")
            
            # Check API documentation
            response = requests.get("https://api.deriv.com", timeout=10)
            print(f"API.deriv.com: {response.status_code}")
            
            return True
            
        except Exception as e:
            print(f"❌ Status check failed: {e}")
            return False
    
    async def test_minimal_connection(self):
        """Test absolute minimal connection"""
        try:
            print(f"\n🔧 MINIMAL CONNECTION TEST")
            print("-" * 28)
            
            url = "wss://ws.derivws.com/websockets/v3"
            
            # Most basic connection possible
            websocket = await websockets.connect(url)
            print(f"✅ Raw connection successful!")
            
            # Just send ping
            await websocket.ping()
            print(f"✅ Ping successful!")
            
            await websocket.close()
            return True
            
        except Exception as e:
            print(f"❌ Minimal connection failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run comprehensive tests"""
        print("🔧 ADVANCED DERIV CONNECTION TEST")
        print("=" * 40)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check Deriv status
        self.check_deriv_status()
        
        # Test HTTP API
        self.test_http_api()
        
        # Test minimal WebSocket
        await self.test_minimal_connection()
        
        # Test with headers
        print(f"\n🔐 TESTING WITH PROPER HEADERS")
        print("-" * 32)
        
        success_count = 0
        for url in self.urls:
            if await self.test_with_headers(url):
                success_count += 1
                break
        
        if success_count > 0:
            print(f"\n🎉 SUCCESS! Found working method")
        else:
            print(f"\n❌ ALL METHODS FAILED")
            print("\n🔧 POSSIBLE CAUSES:")
            print("1. Deriv changed API requirements (new auth method)")
            print("2. Your account region is restricted")
            print("3. Tokens need to be regenerated")
            print("4. API is temporarily down")
            print("5. Your account needs verification")
            
            print(f"\n💡 IMMEDIATE SOLUTIONS:")
            print("1. Try paper trading mode: Set PAPER_TRADING = True")
            print("2. Create completely new API tokens")
            print("3. Check Deriv community forums")
            print("4. Contact Deriv support")

async def main():
    tester = AdvancedDerivTest()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
