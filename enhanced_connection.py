"""
Enhanced WebSocket Connection Manager for Deriv API
Includes VPN/proxy workarounds and multiple connection strategies
"""

import asyncio
import json
from typing import Optional, Tuple

import websockets

from safe_logger import get_safe_logger

# Use safe logger for Windows compatibility
logger = get_safe_logger(__name__)

class EnhancedConnectionManager:
    """Advanced connection manager with multiple fallback strategies"""

    def __init__(self):
        # Multiple WebSocket URLs to try (in order of preference)
        self.websocket_urls = [
            "wss://ws.derivws.com/websockets/v3",      # Primary (recommended)
            "wss://ws.binaryws.com/websockets/v3",     # Legacy fallback
            "wss://frontend.derivws.com/websockets/v3" # Alternative
        ]

        self.websocket = None
        self.connected = False
        self.current_url = None

    async def test_basic_connectivity(self) -> Tuple[bool, str]:
        """Test basic WebSocket connectivity without authentication"""
        logger.info("🧪 Testing basic WebSocket connectivity...")

        connectivity_results = []

        for url in self.websocket_urls:
            try:
                logger.info(f"   Testing {url}...")

                # Try basic connection with short timeout
                websocket = await asyncio.wait_for(
                    websockets.connect(url),
                    timeout=8.0
                )

                # Send ping and wait for pong
                await websocket.ping()

                # Close connection
                await websocket.close()

                logger.info(f"   ✅ SUCCESS - {url}")
                connectivity_results.append(f"✅ {url}")

                # Return immediately on first success
                return True, f"Network connectivity OK - {url} accessible"

            except Exception as e:
                error_msg = str(e)[:60]
                logger.warning(f"   ❌ FAILED - {url}: {error_msg}...")
                connectivity_results.append(f"❌ {url}: {error_msg}")

        # All URLs failed
        failure_summary = "\n".join(connectivity_results)
        return False, f"All WebSocket URLs blocked:\n{failure_summary}"

    async def connect_with_fallbacks(self) -> bool:
        """Connect using multiple fallback strategies"""
        logger.info("🔄 Starting enhanced connection with fallbacks...")

        # Strategy 1: Test basic connectivity first
        is_connected, message = await self.test_basic_connectivity()
        if not is_connected:
            logger.error("🚨 NETWORK CONNECTIVITY ISSUE!")
            logger.error("📡 No WebSocket URLs are accessible from your network.")
            logger.error("\n🔧 IMMEDIATE SOLUTIONS:")
            logger.error("   1. 🌐 Connect IPVanish to: Germany, Netherlands, UK, or Singapore")
            logger.error("   2. 📱 Try mobile hotspot (4G/5G)")
            logger.error("   3. 🔥 Check Windows Firewall settings")
            logger.error("   4. 🏢 Contact IT admin if on corporate network")
            logger.error("\n💡 RECOMMENDED IPVanish SERVERS:")
            logger.error("   • 🇩🇪 Germany (Frankfurt)")
            logger.error("   • 🇳🇱 Netherlands (Amsterdam)")
            logger.error("   • 🇬🇧 UK (London)")
            logger.error("   • 🇸🇬 Singapore")
            logger.error("   • 🇺🇸 USA (New York/Atlanta)")
            logger.error("\n⚠️  AVOID: Caribbean, Jamaica, Africa servers")
            return False

        # Strategy 2: Try enhanced connection methods
        logger.info("✅ Basic connectivity OK, attempting enhanced connection...")

        connection_strategies = [
            self._connect_standard,
            self._connect_with_custom_headers,
            self._connect_with_relaxed_timeouts,
            self._connect_minimal
        ]

        for i, strategy in enumerate(connection_strategies, 1):
            logger.info(f"🔄 Trying connection strategy {i}/{len(connection_strategies)}...")

            if await strategy():
                logger.info(f"✅ Connected successfully using strategy {i}")
                return True

            # Brief delay between strategies
            await asyncio.sleep(2)

        logger.error("❌ All connection strategies failed")
        return False

    async def _connect_standard(self) -> bool:
        """Standard connection method"""
        for url in self.websocket_urls:
            try:
                logger.info(f"   Standard connection to {url}...")

                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        url,
                        ping_interval=30,
                        ping_timeout=10,
                        close_timeout=10,
                        max_size=2**20,  # 1MB
                        max_queue=32
                    ),
                    timeout=15.0
                )

                # Test with ping
                await self.websocket.ping()

                self.connected = True
                self.current_url = url
                logger.info("   ✅ Standard connection successful")
                return True

            except Exception as e:
                logger.warning(f"   ❌ Standard connection failed: {e}")
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except:
                        pass
                    self.websocket = None

        return False

    async def _connect_with_custom_headers(self) -> bool:
        """Connection with custom headers (helps with some proxies/firewalls)"""
        for url in self.websocket_urls:
            try:
                logger.info(f"   Custom headers connection to {url}...")

                # Custom headers that sometimes help with corporate firewalls
                extra_headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Origin": "https://app.deriv.com",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache"
                }

                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        url,
                        extra_headers=extra_headers,
                        ping_interval=30,
                        ping_timeout=10
                    ),
                    timeout=15.0
                )

                await self.websocket.ping()

                self.connected = True
                self.current_url = url
                logger.info("   ✅ Custom headers connection successful")
                return True

            except Exception as e:
                logger.warning(f"   ❌ Custom headers connection failed: {e}")
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except:
                        pass
                    self.websocket = None

        return False

    async def _connect_with_relaxed_timeouts(self) -> bool:
        """Connection with very relaxed timeouts (for slow networks)"""
        for url in self.websocket_urls:
            try:
                logger.info(f"   Relaxed timeouts connection to {url}...")

                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        url,
                        ping_interval=60,  # Very relaxed
                        ping_timeout=30,
                        close_timeout=30,
                        open_timeout=20
                    ),
                    timeout=25.0
                )

                # Skip ping test for this method (might be too slow)

                self.connected = True
                self.current_url = url
                logger.info("   ✅ Relaxed timeouts connection successful")
                return True

            except Exception as e:
                logger.warning(f"   ❌ Relaxed timeouts connection failed: {e}")
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except:
                        pass
                    self.websocket = None

        return False

    async def _connect_minimal(self) -> bool:
        """Minimal connection (no extra parameters)"""
        for url in self.websocket_urls:
            try:
                logger.info(f"   Minimal connection to {url}...")

                self.websocket = await asyncio.wait_for(
                    websockets.connect(url),
                    timeout=10.0
                )

                self.connected = True
                self.current_url = url
                logger.info("   ✅ Minimal connection successful")
                return True

            except Exception as e:
                logger.warning(f"   ❌ Minimal connection failed: {e}")
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except:
                        pass
                    self.websocket = None

        return False

    async def test_connection_with_ping(self) -> bool:
        """Test connection with ping message"""
        if not self.websocket or not self.connected:
            return False

        try:
            # Send ping request
            ping_request = {"ping": 1}
            await self.websocket.send(json.dumps(ping_request))

            # Wait for response
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=5.0
            )

            data = json.loads(response)
            if "ping" in data:
                logger.info("✅ Ping test successful")
                return True
            else:
                logger.warning(f"❓ Unexpected ping response: {data}")
                return False

        except Exception as e:
            logger.error(f"❌ Ping test failed: {e}")
            return False

    async def send_message(self, message: dict) -> Optional[dict]:
        """Send message and get response"""
        if not self.websocket or not self.connected:
            logger.error("❌ No active WebSocket connection")
            return None

        try:
            await self.websocket.send(json.dumps(message))

            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=10.0
            )

            return json.loads(response)

        except Exception as e:
            logger.error(f"❌ Message send failed: {e}")
            return None

    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("🔌 WebSocket connection closed")
            except:
                pass

        self.websocket = None
        self.connected = False
        self.current_url = None

# Quick connectivity test function
async def quick_network_test():
    """Standalone function to quickly test network connectivity"""
    print("🌐 Quick Network Connectivity Test")
    print("=" * 50)

    manager = EnhancedConnectionManager()
    is_connected, message = await manager.test_basic_connectivity()

    print(f"\n📊 Result: {'✅ PASS' if is_connected else '❌ FAIL'}")
    print(f"📝 Details: {message}")

    if not is_connected:
        print("\n🔧 SOLUTION: Use IPVanish VPN")
        print("🌐 Recommended servers:")
        print("   • 🇩🇪 Germany (Frankfurt)")
        print("   • 🇳🇱 Netherlands (Amsterdam)")
        print("   • 🇬🇧 UK (London)")
        print("   • 🇸🇬 Singapore")

    return is_connected

if __name__ == "__main__":
    # Run quick test
    asyncio.run(quick_network_test())
