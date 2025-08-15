"""
Automated Connection Manager for Deriv Trading Bot
This script helps automate the connection process and provides clear guidance
"""

import asyncio
import json
import platform
import subprocess
from typing import Any, Dict

import requests
import websockets

from config import DERIV_API_TOKEN, DERIV_WS_URLS
from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

class AutoConnectionManager:
    """Automated connection management and troubleshooting"""

    def __init__(self):
        self.system = platform.system()
        self.recommended_vpn_servers = {
            "ProtonVPN": ["Germany", "Netherlands", "UK", "Singapore"],
            "NordVPN": ["Germany", "Netherlands", "United Kingdom", "Singapore"],
            "ExpressVPN": ["Germany", "Netherlands", "UK", "Singapore"],
            "Surfshark": ["Germany", "Netherlands", "UK", "Singapore"]
        }

    def check_internet_connectivity(self) -> bool:
        """Check basic internet connectivity"""
        try:
            logger.info("🌐 Checking internet connectivity...")
            response = requests.get("https://www.google.com", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Internet connection: OK")
                return True
            else:
                logger.error("❌ Internet connection: Failed")
                return False
        except Exception as e:
            logger.error(f"❌ Internet connection error: {e}")
            return False

    def check_vpn_status(self) -> Dict[str, Any]:
        """Check if VPN is active and get IP location"""
        try:
            logger.info("🔍 Checking VPN status...")

            # Get public IP and location
            response = requests.get("https://ipapi.co/json/", timeout=10)
            data = response.json()

            country = data.get('country_name', 'Unknown')
            city = data.get('city', 'Unknown')
            ip = data.get('ip', 'Unknown')

            # Check if country is in supported list
            supported_countries = ['Germany', 'Netherlands', 'United Kingdom', 'Singapore', 'United States']
            is_supported = any(sup_country.lower() in country.lower() for sup_country in supported_countries)

            vpn_info = {
                'ip': ip,
                'country': country,
                'city': city,
                'is_supported': is_supported,
                'needs_vpn': not is_supported
            }

            if is_supported:
                logger.info(f"✅ Location: {city}, {country} (Supported)")
            else:
                logger.warning(f"⚠️ Location: {city}, {country} (NOT Supported - VPN Required)")

            return vpn_info

        except Exception as e:
            logger.error(f"❌ VPN status check failed: {e}")
            return {'needs_vpn': True, 'error': str(e)}

    async def test_deriv_connectivity(self) -> Dict[str, Any]:
        """Test connectivity to Deriv WebSocket servers"""
        logger.info("🔗 Testing Deriv server connectivity...")

        results = {}
        working_urls = []

        for i, url in enumerate(DERIV_WS_URLS, 1):
            try:
                logger.info(f"🔄 Testing {i}/{len(DERIV_WS_URLS)}: {url}")

                async with websockets.connect(url, timeout=10) as websocket:
                    # Send ping
                    ping_msg = {"ping": 1}
                    await websocket.send(json.dumps(ping_msg))

                    # Wait for pong
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    data = json.loads(response)

                    if "pong" in data:
                        logger.info(f"✅ {url}: Connection successful")
                        results[url] = "SUCCESS"
                        working_urls.append(url)
                    else:
                        logger.warning(f"⚠️ {url}: Unexpected response")
                        results[url] = "UNEXPECTED_RESPONSE"

            except websockets.exceptions.InvalidStatusCode as e:
                if e.status_code == 401:
                    logger.warning(f"🔒 {url}: HTTP 401 (VPN Required)")
                    results[url] = "VPN_REQUIRED"
                else:
                    logger.error(f"❌ {url}: HTTP {e.status_code}")
                    results[url] = f"HTTP_{e.status_code}"
            except Exception as e:
                logger.error(f"❌ {url}: {str(e)[:50]}")
                results[url] = "FAILED"

        return {
            'results': results,
            'working_urls': working_urls,
            'success_count': len(working_urls),
            'total_count': len(DERIV_WS_URLS)
        }

    async def test_api_authorization(self, working_url: str) -> bool:
        """Test API token authorization"""
        if not DERIV_API_TOKEN:
            logger.error("❌ No API token configured")
            return False

        try:
            logger.info("🔐 Testing API token authorization...")

            async with websockets.connect(working_url, timeout=10) as websocket:
                auth_msg = {"authorize": DERIV_API_TOKEN}
                await websocket.send(json.dumps(auth_msg))

                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(response)

                if "authorize" in data:
                    account = data["authorize"]
                    logger.info("✅ API Token: Valid")
                    logger.info(f"💰 Account: {account.get('loginid', 'Unknown')}")
                    logger.info(f"💵 Balance: ${float(account.get('balance', 0)):.2f}")
                    return True
                elif "error" in data:
                    error = data["error"]
                    logger.error(f"❌ API Token Error: {error.get('message', 'Unknown')}")
                    return False

        except Exception as e:
            logger.error(f"❌ API authorization test failed: {e}")
            return False

    def get_vpn_setup_instructions(self) -> str:
        """Get detailed VPN setup instructions"""
        instructions = """
🚀 VPN SETUP INSTRUCTIONS FOR DERIV TRADING

RECOMMENDED VPN PROVIDERS (Choose ONE):

1. 🥇 PROTONVPN (FREE OPTION AVAILABLE)
   • Download: https://protonvpn.com/download
   • Free servers in Netherlands available
   • Connect to: Netherlands, Germany, or US

2. 🥈 NORDVPN (Most Popular)
   • Download: https://nordvpn.com/download
   • 30-day money-back guarantee
   • Connect to: Germany, Netherlands, UK

3. 🥉 EXPRESSVPN (Fastest)
   • Download: https://www.expressvpn.com/download
   • Premium service with 30-day guarantee
   • Connect to: Germany, Netherlands, Singapore

SETUP STEPS:
1. Install your chosen VPN
2. Create account and login
3. Connect to supported country:
   ✅ Germany (Frankfurt)
   ✅ Netherlands (Amsterdam)
   ✅ United Kingdom (London)
   ✅ Singapore
   ✅ United States (New York/Atlanta)

4. Verify connection by running: python auto_connection_manager.py
5. Once VPN is connected, run your bot: python main.py

❌ AVOID THESE LOCATIONS:
• Caribbean countries
• Jamaica
• Most African servers
• Some Asian countries (except Singapore)
"""
        return instructions

    def open_vpn_download_pages(self):
        """Open VPN provider websites (Windows only)"""
        if self.system == "Windows":
            vpn_urls = [
                "https://protonvpn.com/download",
                "https://nordvpn.com/download",
                "https://www.expressvpn.com/download"
            ]

            for url in vpn_urls:
                try:
                    subprocess.run(["start", url], shell=True, check=True)
                    logger.info(f"🌐 Opened: {url}")
                except Exception as e:
                    logger.error(f"Failed to open {url}: {e}")

    async def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete diagnostic and provide action plan"""
        logger.info("🔍 STARTING FULL DIAGNOSTIC")
        logger.info("=" * 60)

        # Step 1: Check internet
        internet_ok = self.check_internet_connectivity()
        if not internet_ok:
            return {
                'status': 'FAILED',
                'step': 'INTERNET',
                'action': 'Fix your internet connection first'
            }

        # Step 2: Check VPN status
        vpn_info = self.check_vpn_status()

        # Step 3: Test Deriv connectivity
        deriv_test = await self.test_deriv_connectivity()

        # Step 4: Test API if connection works
        api_ok = False
        if deriv_test['working_urls']:
            api_ok = await self.test_api_authorization(deriv_test['working_urls'][0])

        # Generate action plan
        if deriv_test['success_count'] > 0 and api_ok:
            return {
                'status': 'SUCCESS',
                'message': 'All systems ready! Your bot should work now.',
                'vpn_info': vpn_info,
                'deriv_test': deriv_test
            }
        elif deriv_test['success_count'] > 0 and not api_ok:
            return {
                'status': 'API_ISSUE',
                'message': 'Connection works but API token has issues',
                'action': 'Check your API token at https://app.deriv.com/account/api-token',
                'vpn_info': vpn_info,
                'deriv_test': deriv_test
            }
        elif vpn_info.get('needs_vpn', True):
            return {
                'status': 'VPN_REQUIRED',
                'message': 'VPN connection required',
                'action': 'Connect to VPN in supported country',
                'vpn_info': vpn_info,
                'deriv_test': deriv_test,
                'instructions': self.get_vpn_setup_instructions()
            }
        else:
            return {
                'status': 'NETWORK_ISSUE',
                'message': 'Network connectivity problems',
                'action': 'Check firewall, try different network, or contact ISP',
                'vpn_info': vpn_info,
                'deriv_test': deriv_test
            }

async def main():
    """Main diagnostic function"""
    print("🔧 DERIV BOT AUTO CONNECTION MANAGER")
    print("=" * 60)

    manager = AutoConnectionManager()

    try:
        # Run full diagnostic
        result = await manager.run_full_diagnostic()

        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC RESULTS")
        print("=" * 60)

        status = result['status']

        if status == 'SUCCESS':
            print("🎉 SUCCESS! Your bot is ready to run!")
            print("✅ Internet: Connected")
            print("✅ VPN: Properly configured")
            print("✅ Deriv: Accessible")
            print("✅ API Token: Valid")
            print("\n🚀 You can now run: python main.py")

        elif status == 'VPN_REQUIRED':
            print("🔒 VPN CONNECTION REQUIRED")
            print("❌ Deriv servers are blocked from your location")
            print("\n📋 IMMEDIATE ACTIONS NEEDED:")
            print("1. Install a VPN (ProtonVPN has free option)")
            print("2. Connect to: Germany, Netherlands, UK, or Singapore")
            print("3. Run this diagnostic again")
            print("4. Once connected, run your bot")

            if result.get('instructions'):
                print("\n" + result['instructions'])

        elif status == 'API_ISSUE':
            print("🔑 API TOKEN ISSUE")
            print("✅ Network: Connected to Deriv")
            print("❌ API Token: Invalid or expired")
            print("\n📋 FIX YOUR API TOKEN:")
            print("1. Go to: https://app.deriv.com/account/api-token")
            print("2. Delete old tokens")
            print("3. Create new token with 'Trading' permissions")
            print("4. Update config.py with new token")

        elif status == 'NETWORK_ISSUE':
            print("🌐 NETWORK CONNECTIVITY ISSUE")
            print("❌ Unable to reach Deriv servers")
            print("\n📋 TROUBLESHOOTING STEPS:")
            print("1. Check Windows Firewall settings")
            print("2. Try different internet connection")
            print("3. Restart your router/modem")
            print("4. Contact your ISP if problems persist")

        else:
            print("❌ INTERNET CONNECTION FAILED")
            print("📋 BASIC TROUBLESHOOTING:")
            print("1. Check your internet connection")
            print("2. Restart your router/modem")
            print("3. Try using mobile hotspot")

        # Show detailed results
        if 'vpn_info' in result:
            vpn = result['vpn_info']
            print(f"\n🌍 Your Location: {vpn.get('city', 'Unknown')}, {vpn.get('country', 'Unknown')}")
            print(f"🔢 Your IP: {vpn.get('ip', 'Unknown')}")

        if 'deriv_test' in result:
            deriv = result['deriv_test']
            print(f"\n📊 Deriv Servers: {deriv['success_count']}/{deriv['total_count']} accessible")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        print("❌ Diagnostic failed. Try running with administrator privileges.")

if __name__ == "__main__":
    asyncio.run(main())
