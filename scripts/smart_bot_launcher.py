"""
Smart Bot Launcher - Automated Connection and Fallback System
This launcher will help you get connected automatically with multiple fallback strategies
"""

import asyncio
import json
import subprocess
import sys
from typing import Any, Dict

import requests
import websockets

from config import DERIV_API_TOKEN, DERIV_DEMO_API_TOKEN, DERIV_WS_URLS
from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

class SmartBotLauncher:
    """Smart launcher with automatic connection strategies"""

    def __init__(self):
        self.connection_strategies = [
            "direct_connection",
            "dns_change",
            "alternative_ports",
            "proxy_detection",
            "vpn_guidance"
        ]
        self.working_connection = None

    def get_location_info(self) -> Dict[str, Any]:
        """Get current location and IP info"""
        try:
            # Try multiple IP detection services
            services = [
                "https://ipapi.co/json/",
                "https://api.ipify.org?format=json",
                "https://httpbin.org/ip"
            ]

            for service in services:
                try:
                    response = requests.get(service, timeout=5)
                    if "ipapi.co" in service:
                        data = response.json()
                        return {
                            'ip': data.get('ip', 'Unknown'),
                            'country': data.get('country_name', 'Unknown'),
                            'city': data.get('city', 'Unknown'),
                            'isp': data.get('org', 'Unknown'),
                            'blocked_regions': self.check_if_blocked_region(data.get('country_name', ''))
                        }
                except:
                    continue

            return {'ip': 'Unknown', 'country': 'Unknown', 'needs_vpn': True}

        except Exception as e:
            logger.error(f"Location detection failed: {e}")
            return {'ip': 'Unknown', 'country': 'Unknown', 'needs_vpn': True}

    def check_if_blocked_region(self, country: str) -> bool:
        """Check if country is in blocked regions"""
        blocked_keywords = [
            'jamaica', 'caribbean', 'cuba', 'iran', 'north korea',
            'syria', 'myanmar', 'afghanistan', 'belarus', 'russia'
        ]
        return any(keyword in country.lower() for keyword in blocked_keywords)

    async def try_direct_connection(self) -> bool:
        """Strategy 1: Try direct connection to all URLs"""
        logger.info("🔄 Strategy 1: Testing direct connections...")

        for url in DERIV_WS_URLS:
            try:
                async with websockets.connect(url, timeout=10) as ws:
                    ping_msg = {"ping": 1}
                    await ws.send(json.dumps(ping_msg))
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(response)

                    if "pong" in data:
                        logger.info(f"✅ Direct connection successful: {url}")
                        self.working_connection = url
                        return True

            except Exception as e:
                logger.debug(f"Direct connection failed for {url}: {str(e)[:50]}")
                continue

        logger.warning("❌ Strategy 1: All direct connections failed")
        return False

    def try_dns_change(self) -> bool:
        """Strategy 2: Suggest DNS change"""
        logger.info("🔄 Strategy 2: DNS optimization...")

        dns_servers = [
            ("Google DNS", "8.8.8.8, 8.8.4.4"),
            ("Cloudflare DNS", "1.1.1.1, 1.0.0.1"),
            ("OpenDNS", "208.67.222.222, 208.67.220.220")
        ]

        logger.info("💡 DNS Change Recommendation:")
        logger.info("Your ISP's DNS might be blocking Deriv. Try changing to:")
        for name, servers in dns_servers:
            logger.info(f"   • {name}: {servers}")

        logger.info("\n🔧 Windows DNS Change Steps:")
        logger.info("1. Press Win+R, type 'ncpa.cpl', press Enter")
        logger.info("2. Right-click your network connection")
        logger.info("3. Properties → Internet Protocol Version 4 (TCP/IPv4)")
        logger.info("4. Use the following DNS server addresses:")
        logger.info("   Preferred: 8.8.8.8")
        logger.info("   Alternate: 8.8.4.4")
        logger.info("5. Click OK and restart your connection")

        return False  # User needs to manually change DNS

    async def try_alternative_ports(self) -> bool:
        """Strategy 3: Try alternative connection methods"""
        logger.info("🔄 Strategy 3: Testing alternative connection methods...")

        # Try HTTP fallback (if Deriv supports it)
        alternative_urls = [
            "wss://ws.derivws.com:443/websockets/v3",
            "wss://ws.binaryws.com:443/websockets/v3",
            "wss://frontend.derivws.com:443/websockets/v3"
        ]

        for url in alternative_urls:
            try:
                async with websockets.connect(url, timeout=8) as ws:
                    ping_msg = {"ping": 1}
                    await ws.send(json.dumps(ping_msg))
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(response)

                    if "pong" in data:
                        logger.info(f"✅ Alternative connection successful: {url}")
                        self.working_connection = url
                        return True

            except Exception as e:
                logger.debug(f"Alternative connection failed for {url}: {str(e)[:50]}")
                continue

        logger.warning("❌ Strategy 3: Alternative connections failed")
        return False

    def detect_proxy_or_firewall(self) -> Dict[str, Any]:
        """Strategy 4: Detect proxy/firewall issues"""
        logger.info("🔄 Strategy 4: Detecting network restrictions...")

        issues = []

        # Check for proxy
        try:
            import urllib.request
            proxy = urllib.request.getproxies()
            if proxy:
                issues.append(f"Proxy detected: {proxy}")
                logger.warning(f"⚠️ Proxy detected: {proxy}")
        except:
            pass

        # Check Windows Firewall (basic detection)
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["netsh", "advfirewall", "show", "allprofiles", "state"],
                    capture_output=True, text=True, timeout=10
                )
                if "ON" in result.stdout:
                    issues.append("Windows Firewall is active")
                    logger.warning("⚠️ Windows Firewall is active - may block connections")
            except:
                pass

        return {
            'issues': issues,
            'recommendations': [
                "Temporarily disable Windows Firewall",
                "Check antivirus software settings",
                "Try using mobile hotspot",
                "Contact your ISP about WebSocket blocking"
            ]
        }

    def provide_vpn_guidance(self, location_info: Dict[str, Any]) -> str:
        """Strategy 5: Provide specific VPN guidance"""
        logger.info("🔄 Strategy 5: VPN setup guidance...")

        country = location_info.get('country', 'Unknown')

        if country == 'United Kingdom':
            guidance = """
🇬🇧 UK-SPECIFIC VPN GUIDANCE:

Your location (UK) SHOULD work with Deriv, but your ISP appears to be blocking connections.

RECOMMENDED IMMEDIATE ACTIONS:

1. 📱 TRY MOBILE HOTSPOT FIRST (Often works immediately)
   • Enable hotspot on your phone
   • Connect computer to phone's hotspot  
   • Test bot again

2. 🌐 FREE VPN OPTIONS FOR UK:
   • ProtonVPN (Free tier available)
   • Windscribe (10GB free monthly)
   • Connect to: Netherlands or Germany servers

3. 🔧 ISP-SPECIFIC FIXES:
   • BT/EE customers: Often block WebSockets
   • Sky: Check parental controls
   • Virgin Media: Try DNS change first
   • Three/O2: Mobile hotspot usually works

4. ⚡ QUICK DNS FIX FOR UK ISPs:
   • Change DNS to 8.8.8.8 and 8.8.4.4
   • Restart network connection
   • Test again
"""
        else:
            guidance = f"""
🌍 VPN GUIDANCE FOR {country.upper()}:

IMMEDIATE ACTIONS:
1. Install VPN (ProtonVPN has free option)
2. Connect to supported region:
   ✅ Germany, Netherlands, UK, Singapore
3. Test connection again
4. Run your bot

RECOMMENDED VPN SERVERS:
• Germany (Frankfurt) - Fast, reliable
• Netherlands (Amsterdam) - Low latency  
• Singapore - Good for global access
"""

        return guidance

    async def attempt_connection_with_auth(self, url: str) -> bool:
        """Test full connection with authentication"""
        if not DERIV_API_TOKEN and not DERIV_DEMO_API_TOKEN:
            logger.error("❌ No API tokens configured")
            return False

        token = DERIV_DEMO_API_TOKEN or DERIV_API_TOKEN

        try:
            logger.info(f"🔐 Testing full connection with auth to {url}...")

            async with websockets.connect(url, timeout=15) as ws:
                # First, ping
                ping_msg = {"ping": 1}
                await ws.send(json.dumps(ping_msg))
                response = await asyncio.wait_for(ws.recv(), timeout=5)

                # Then, authorize
                auth_msg = {"authorize": token}
                await ws.send(json.dumps(auth_msg))
                auth_response = await asyncio.wait_for(ws.recv(), timeout=10)
                auth_data = json.loads(auth_response)

                if "authorize" in auth_data:
                    account = auth_data["authorize"]
                    logger.info("🎉 FULL CONNECTION SUCCESSFUL!")
                    logger.info(f"💰 Account: {account.get('loginid', 'Unknown')}")
                    logger.info(f"💵 Balance: ${float(account.get('balance', 0)):.2f}")
                    return True

        except Exception as e:
            logger.error(f"Full connection test failed: {e}")
            return False

        return False

    async def run_smart_connection_process(self) -> Dict[str, Any]:
        """Run the complete smart connection process"""
        logger.info("🚀 SMART BOT LAUNCHER - STARTING CONNECTION PROCESS")
        logger.info("=" * 70)

        # Get location info
        location_info = self.get_location_info()
        f"🌍 Your location: {location_info.get('city', 'Unknown')}"
        f" {location_info.get('country', 'Unknown')}"
        logger.info(f"🔢 Your IP: {location_info.get('ip', 'Unknown')}")
        logger.info(f"🏢 ISP: {location_info.get('isp', 'Unknown')}")

        # Strategy 1: Direct connection
        if await self.try_direct_connection():
            if await self.attempt_connection_with_auth(self.working_connection):
                return {
                    'status': 'SUCCESS',
                    'method': 'direct_connection',
                    'url': self.working_connection,
                    'message': 'Bot is ready to run!'
                }

        # Strategy 2: DNS change guidance
        self.try_dns_change()

        # Strategy 3: Alternative ports
        if await self.try_alternative_ports():
            if await self.attempt_connection_with_auth(self.working_connection):
                return {
                    'status': 'SUCCESS',
                    'method': 'alternative_ports',
                    'url': self.working_connection,
                    'message': 'Bot connected via alternative method!'
                }

        # Strategy 4: Detect issues
        network_issues = self.detect_proxy_or_firewall()

        # Strategy 5: VPN guidance
        vpn_guidance = self.provide_vpn_guidance(location_info)

        return {
            'status': 'VPN_REQUIRED',
            'location_info': location_info,
            'network_issues': network_issues,
            'vpn_guidance': vpn_guidance,
            'message': 'VPN or network changes required'
        }

async def main():
    """Main launcher function"""
    print("🚀 SMART BOT LAUNCHER")
    print("=" * 50)
    print("⚡ Automatically finding the best way to connect...")
    print("=" * 50)

    launcher = SmartBotLauncher()

    try:
        result = await launcher.run_smart_connection_process()

        print("\n" + "=" * 70)
        print("📊 CONNECTION ANALYSIS COMPLETE")
        print("=" * 70)

        if result['status'] == 'SUCCESS':
            print("🎉 SUCCESS! YOUR BOT CAN NOW RUN!")
            print(f"✅ Connection method: {result['method']}")
            print(f"✅ Working URL: {result['url']}")
            print("\n🚀 NEXT STEP: Run your bot with:")
            print("   python main.py")

        else:
            print("🔧 CONNECTION SETUP REQUIRED")
            print("\n📱 QUICKEST SOLUTION: Try mobile hotspot first!")
            print("   1. Enable hotspot on your phone")
            print("   2. Connect computer to phone's Wi-Fi")
            print("   3. Run this launcher again")

            if 'vpn_guidance' in result:
                print("\n" + result['vpn_guidance'])

            if 'network_issues' in result and result['network_issues']['issues']:
                print("\n🔍 DETECTED ISSUES:")
                for issue in result['network_issues']['issues']:
                    print(f"   • {issue}")

                print("\n💡 RECOMMENDATIONS:")
                for rec in result['network_issues']['recommendations']:
                    print(f"   • {rec}")

        print("\n" + "=" * 70)

    except Exception as e:
        logger.error(f"Launcher failed: {e}")
        print(f"❌ Launcher error: {e}")
        print("\n🆘 MANUAL STEPS:")
        print("1. Try mobile hotspot")
        print("2. Install ProtonVPN (free)")
        print("3. Connect to Netherlands or Germany")
        print("4. Run: python main.py")

if __name__ == "__main__":
    asyncio.run(main())
