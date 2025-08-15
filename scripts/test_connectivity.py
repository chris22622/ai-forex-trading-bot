"""
Quick WebSocket Connectivity Test for Deriv API
Use this to test if your VPN/network allows WebSocket connections

Run this BEFORE starting your trading bot to verify connectivity.
"""

import asyncio
import json
import time

import websockets


async def test_websocket_connectivity():
    """Test WebSocket connectivity to Deriv servers"""

    print("🧪 DERIV WEBSOCKET CONNECTIVITY TEST")
    print("=" * 50)
    print("📅 Testing at:", time.strftime('%Y-%m-%d %H:%M:%S'))
    print()

    # URLs to test (in order of preference)
    test_urls = [
        "wss://ws.derivws.com/websockets/v3",      # Primary
        "wss://ws.binaryws.com/websockets/v3",     # Legacy
        "wss://frontend.derivws.com/websockets/v3" # Alternative
    ]

    successful_connections = 0
    detailed_results = []

    for i, uri in enumerate(test_urls, 1):
        print(f"🔄 Test {i}/3: {uri}")

        try:
            # Test connection with timeout
            start_time = time.time()

            async with websockets.connect(uri, open_timeout=10) as ws:
                connect_time = time.time() - start_time

                # Send ping and measure response time
                ping_start = time.time()
                await ws.send(json.dumps({"ping": 1}))

                response = await asyncio.wait_for(ws.recv(), timeout=5)
                ping_time = time.time() - ping_start

                data = json.loads(response)

                if "ping" in data:
                    print("   ✅ SUCCESS")
                    print(f"   📡 Connect time: {connect_time:.2f}s")
                    print(f"   🏓 Ping time: {ping_time:.3f}s")

                    successful_connections += 1
                    detailed_results.append({
                        'url': uri,
                        'status': 'SUCCESS',
                        'connect_time': connect_time,
                        'ping_time': ping_time
                    })
                else:
                    print(f"   ❓ UNEXPECTED RESPONSE: {data}")
                    detailed_results.append({
                        'url': uri,
                        'status': 'UNEXPECTED_RESPONSE',
                        'response': data
                    })

        except asyncio.TimeoutError:
            print("   ❌ TIMEOUT - Connection took too long")
            detailed_results.append({
                'url': uri,
                'status': 'TIMEOUT'
            })

        except ConnectionRefusedError:
            print("   ❌ CONNECTION REFUSED - Server rejected connection")
            detailed_results.append({
                'url': uri,
                'status': 'CONNECTION_REFUSED'
            })

        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ ERROR: {error_msg}")
            detailed_results.append({
                'url': uri,
                'status': 'ERROR',
                'error': error_msg
            })

        print()  # Empty line between tests

    # Summary
    print("📊 CONNECTIVITY TEST RESULTS")
    print("=" * 50)
    print(f"✅ Successful connections: {successful_connections}/3")

    if successful_connections == 0:
        print("\n🚨 CRITICAL: No WebSocket servers accessible!")
        print("\n🔧 IMMEDIATE ACTIONS NEEDED:")
        print("   1. 🌐 Connect to IPVanish VPN")
        print("   2. 🌍 Choose server in: Germany, UK, Netherlands, or Singapore")
        print("   3. 🔄 Run this test again")
        print("   4. ❌ Avoid: Caribbean, Jamaica, Africa servers")
        print("\n💡 IPVanish Server Recommendations:")
        print("   • 🇩🇪 Germany - Frankfurt (fast, reliable)")
        print("   • 🇬🇧 UK - London (good for trading)")
        print("   • 🇳🇱 Netherlands - Amsterdam (low latency)")
        print("   • 🇸🇬 Singapore (best for Asia)")
        print("   • 🇺🇸 USA - New York or Atlanta")

    elif successful_connections < 3:
        print(f"\n⚠️  PARTIAL ACCESS: {successful_connections}/3 servers working")
        print("📡 Some servers blocked, but trading should work")
        print("🎯 Your bot will use the working connections")

    else:
        print("\n🎉 EXCELLENT: All servers accessible!")
        print("✅ Your network allows WebSocket connections")
        print("🚀 Trading bot should work perfectly")

        # Show best server
        if detailed_results:
            fastest = min(detailed_results, key=lambda x: x.get('connect_time', 999))
            print(f"🏆 Fastest server: {fastest['url']}")
            print(f"⚡ Connect time: {fastest['connect_time']:.2f}s")

    print("\n" + "=" * 50)
    return successful_connections > 0

async def main():
    """Main test function"""
    success = await test_websocket_connectivity()

    if success:
        print("🎯 NEXT STEP: Run your trading bot - connections should work!")
    else:
        print("🛑 FIX NETWORK FIRST: Connect VPN, then rerun this test")

    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit_code = 0 if result else 1
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        exit(1)
