#!/usr/bin/env python3
"""
🚀 DERIV TRADING BOT STARTUP SCRIPT
Smart launcher with automatic problem detection
"""

import asyncio
import sys
import os
import traceback

def print_banner():
    """Print startup banner"""
    print("=" * 70)
    print("🚀 DERIV TRADING BOT - ENHANCED STARTUP")
    print("🤖 AI-Powered Trading with Smart Connection Handling")
    print("=" * 70)

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check config file
    if not os.path.exists('config.py'):
        print("❌ config.py not found!")
        return False
    
    # Check API tokens
    try:
        from config import DERIV_DEMO_API_TOKEN, DERIV_LIVE_API_TOKEN, DEMO_MODE
        
        if DEMO_MODE:
            if not DERIV_DEMO_API_TOKEN or len(DERIV_DEMO_API_TOKEN.strip()) < 10:
                print("❌ Demo API token missing or invalid!")
                print("🔧 Please set DERIV_DEMO_API_TOKEN in config.py")
                return False
        else:
            if not DERIV_LIVE_API_TOKEN or len(DERIV_LIVE_API_TOKEN.strip()) < 10:
                print("❌ Live API token missing or invalid!")
                print("🔧 Please set DERIV_LIVE_API_TOKEN in config.py")
                return False
                
        print("✅ Configuration check passed")
        return True
        
    except ImportError as e:
        print(f"❌ Config import error: {e}")
        return False

async def test_connection():
    """Quick connection test with enhanced diagnostics"""
    print("🔌 Testing API connection...")
    
    try:
        # First, test basic WebSocket connectivity without auth
        import websockets
        import asyncio
        
        print("🔍 Step 1: Testing basic WebSocket connectivity...")
        
        urls_to_test = [
            "wss://ws.derivws.com/websockets/v3",
            "wss://ws.binaryws.com/websockets/v3",
            "wss://frontend.derivws.com/websockets/v3"
        ]
        
        connectivity_ok = False
        for url in urls_to_test:
            try:
                print(f"   Testing {url}...")
                websocket = await asyncio.wait_for(
                    websockets.connect(url),
                    timeout=10.0
                )
                await websocket.close()
                print(f"   ✅ {url} - Basic connection successful")
                connectivity_ok = True
                break
            except Exception as e:
                print(f"   ❌ {url} - {str(e)[:50]}...")
                continue
        
        if not connectivity_ok:
            print("\n🚨 NETWORK CONNECTIVITY ISSUE DETECTED!")
            print("📡 No WebSocket URLs are accessible from your network.")
            print("\n🔧 IMMEDIATE SOLUTIONS:")
            print("   1. 🌐 Use VPN (ProtonVPN, NordVPN, ExpressVPN)")
            print("   2. 📱 Try mobile hotspot")
            print("   3. 🔥 Check firewall settings")
            print("   4. 🏢 Contact IT admin if on corporate network")
            print("\n💡 RECOMMENDED VPN SERVERS:")
            print("   • UK (London)")
            print("   • Singapore") 
            print("   • Germany (Frankfurt)")
            print("   • Netherlands (Amsterdam)")
            print("\n⚠️  AFTER fixing network access, you'll need NEW API tokens!")
            return False
        
        print("\n🔍 Step 2: Testing API token authentication...")
        
        # Import the comprehensive tester
        from comprehensive_api_fix import ComprehensiveAPIFixer
        
        fixer = ComprehensiveAPIFixer()
        
        # Quick test (just check one URL and one token)
        from config import DERIV_DEMO_API_TOKEN, DERIV_LIVE_API_TOKEN, DEMO_MODE
        
        token = DERIV_DEMO_API_TOKEN if DEMO_MODE else DERIV_LIVE_API_TOKEN
        token_type = "DEMO" if DEMO_MODE else "LIVE"
        
        # Test primary URL with token
        if await fixer.test_token_with_url(urls_to_test[0], token, token_type):
            print("✅ API connection and authentication successful!")
            return True
        else:
            print(f"\n🚨 TOKEN AUTHENTICATION FAILED!")
            print(f"❌ Your {token_type} token is invalid or expired")
            print(f"🔑 Current token: {token[:8]}...")
            print("\n🔧 TOKEN SOLUTIONS:")
            print("   1. 🌐 Go to: https://app.deriv.com/account/api-token")
            print("   2. 🗑️  Delete old tokens")
            print("   3. ➕ Create NEW token with these permissions:")
            print("      • ✅ Read")
            print("      • ✅ Trade") 
            print("      • ✅ Trading Information")
            print("      • ✅ Payments")
            print("      • ✅ Admin")
            print("   4. 📝 Update config.py with new token")
            print("   5. 🧪 Test again with option 4")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Check internet connection")
        print("   2. Try VPN if blocked")
        print("   3. Run: python comprehensive_api_fix.py")
        return False

def show_menu():
    """Show startup menu"""
    print("\n🎯 STARTUP OPTIONS:")
    print("1. 🚀 Start Trading Bot (Full Mode)")
    print("2. 🎮 Run Demo Trading (No API needed)")
    print("3. 🔧 Fix API Connection Issues") 
    print("4. 🧪 Test API Connection & Auth")
    print("5. 🌐 Quick Network Test (No Auth)")
    print("6. 🔬 VPN Connectivity Test")
    print("7. 📖 Show Solution Guide")
    print("8. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-8): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                return int(choice)
            else:
                print("❌ Please enter a number between 1-8")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

async def quick_network_test():
    """Quick network connectivity test"""
    print("🌐 Quick Network Test...")
    print("🔍 Testing basic WebSocket connectivity (no auth)...")
    
    import websockets
    
    urls_to_test = [
        "wss://ws.derivws.com/websockets/v3",
        "wss://ws.binaryws.com/websockets/v3", 
        "wss://frontend.derivws.com/websockets/v3"
    ]
    
    success_count = 0
    for i, url in enumerate(urls_to_test, 1):
        try:
            print(f"   {i}/3 Testing {url}...")
            websocket = await asyncio.wait_for(
                websockets.connect(url),
                timeout=8.0
            )
            await websocket.close()
            print(f"   ✅ SUCCESS - {url}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ FAILED - {str(e)[:60]}...")
    
    print(f"\n📊 Results: {success_count}/3 URLs accessible")
    
    if success_count == 0:
        print("\n🚨 NETWORK BLOCKED - No Deriv servers accessible!")
        print("🔧 SOLUTIONS:")
        print("   1. 🌐 Connect to IPVanish VPN")
        print("   2. 📱 Try mobile hotspot")
        print("   3. 🔥 Check Windows Firewall")
        print("   4. 🏢 Contact IT if on corporate network")
    elif success_count < 3:
        print(f"\n⚠️  PARTIAL ACCESS - {success_count}/3 servers accessible")
        print("📡 Some servers blocked, but trading may still work")
    else:
        print("\n✅ EXCELLENT - All servers accessible!")
        print("🔗 Network connectivity is good")
        print("🔑 If trading fails, it's likely a token issue")

async def test_vpn_connectivity():
    """Run comprehensive VPN connectivity test"""
    print("🔬 VPN Connectivity Test...")
    print("🎯 This test checks if WebSocket connections work through your VPN")
    print()
    
    try:
        from test_connectivity import test_websocket_connectivity
        success = await test_websocket_connectivity()
        
        if success:
            print("\n🎉 VPN Test Result: SUCCESS")
            print("✅ Your VPN/network allows Deriv WebSocket connections")
            print("🚀 You can now start the trading bot!")
        else:
            print("\n🚨 VPN Test Result: FAILED")
            print("❌ WebSocket connections are blocked")
            print("\n🔧 IPVanish VPN Setup Instructions:")
            print("   1. 🌐 Open IPVanish application")
            print("   2. 🔌 Disconnect from current server (if connected)")
            print("   3. 🌍 Connect to recommended server:")
            print("      • 🇩🇪 Germany - Frankfurt")
            print("      • 🇬🇧 UK - London")  
            print("      • 🇳🇱 Netherlands - Amsterdam")
            print("      • 🇸🇬 Singapore")
            print("   4. ✅ Wait for connection to establish")
            print("   5. 🔄 Run this test again")
            print("\n⚠️  AVOID these server locations:")
            print("   ❌ Caribbean countries")
            print("   ❌ Jamaica")
            print("   ❌ African servers")
            print("   ❌ Some US servers (try NY/Atlanta only)")
        
        return success
        
    except Exception as e:
        print(f"❌ VPN test failed: {e}")
        return False
    """Run comprehensive VPN connectivity test"""
    print("🔬 VPN Connectivity Test...")
    print("🎯 This test checks if WebSocket connections work through your VPN")
    print()
    
    try:
        from test_connectivity import test_websocket_connectivity
        success = await test_websocket_connectivity()
        
        if success:
            print("\n🎉 VPN Test Result: SUCCESS")
            print("✅ Your VPN/network allows Deriv WebSocket connections")
            print("🚀 You can now start the trading bot!")
        else:
            print("\n🚨 VPN Test Result: FAILED")
            print("❌ WebSocket connections are blocked")
            print("\n🔧 IPVanish VPN Setup Instructions:")
            print("   1. 🌐 Open IPVanish application")
            print("   2. 🔌 Disconnect from current server (if connected)")
            print("   3. 🌍 Connect to recommended server:")
            print("      • 🇩🇪 Germany - Frankfurt")
            print("      • 🇬🇧 UK - London")  
            print("      • 🇳🇱 Netherlands - Amsterdam")
            print("      • 🇸🇬 Singapore")
            print("   4. ✅ Wait for connection to establish")
            print("   5. � Run this test again")
            print("\n⚠️  AVOID these server locations:")
            print("   ❌ Caribbean countries")
            print("   ❌ Jamaica")
            print("   ❌ African servers")
            print("   ❌ Some US servers (try NY/Atlanta only)")
        
        return success
        
    except Exception as e:
        print(f"❌ VPN test failed: {e}")
        return False

async def start_trading_bot():
    """Start the main trading bot"""
    print("🚀 Starting trading bot...")
    
    try:
        # Import and run main bot
        from main import main as bot_main
        await bot_main()
    except Exception as e:
        print(f"❌ Bot startup failed: {e}")
        traceback.print_exc()

async def run_demo_trading():
    """Run demo trading"""
    print("🎮 Starting demo trading...")
    
    try:
        from demo_trading import main as demo_main
        await demo_main()
    except Exception as e:
        print(f"❌ Demo startup failed: {e}")
        traceback.print_exc()

async def fix_api_connection():
    """Run comprehensive API fixer"""
    print("🔧 Running comprehensive API fixer...")
    
    try:
        from comprehensive_api_fix import main as fix_main
        await fix_main()
    except Exception as e:
        print(f"❌ API fixer failed: {e}")
        traceback.print_exc()

async def main():
    """Main startup function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n🔧 Please fix configuration issues and try again")
        print("📖 See FINAL_SOLUTION.md for detailed instructions")
        return
    
    # Show menu and handle choice
    while True:
        choice = show_menu()
        
        if choice == 1:
            # Test connection first
            print("\n🔍 Pre-flight check...")
            if await test_connection():
                await start_trading_bot()
            else:
                print("\n❌ Connection failed. Please fix issues first.")
                print("\n🎯 NEXT STEPS:")
                print("   • If NETWORK issue: Use VPN → Try option 4 again")
                print("   • If TOKEN issue: Create new tokens → Try option 4 again") 
                print("   • For detailed help: Try option 3 (Comprehensive Fix)")
                print("   • For step-by-step guide: Read FINAL_SOLUTION.md")
                
                # Ask what they want to do
                print("\n❓ What would you like to do?")
                print("   A. 🔧 Run comprehensive API fixer")
                print("   B. 🧪 Test connection again") 
                print("   C. 📖 Show solution guide")
                print("   D. 🔙 Return to main menu")
                
                sub_choice = input("Choose (A/B/C/D): ").strip().upper()
                if sub_choice == 'A':
                    await fix_api_connection()
                elif sub_choice == 'B':
                    await test_connection()
                elif sub_choice == 'C':
                    print("\n📖 Reading FINAL_SOLUTION.md...")
                    try:
                        with open('FINAL_SOLUTION.md', 'r', encoding='utf-8') as f:
                            print(f.read())
                    except FileNotFoundError:
                        print("❌ FINAL_SOLUTION.md not found. Run option 3 to create it.")
                # D or any other choice returns to menu
                continue
                
        elif choice == 2:
            await run_demo_trading()
            
        elif choice == 3:
            await fix_api_connection()
            
        elif choice == 4:
            await test_connection()
            
        elif choice == 5:
            await quick_network_test()
            
        elif choice == 6:
            await test_vpn_connectivity()
            
        elif choice == 7:
            print("\n📖 Solution Guide:")
            try:
                with open('FINAL_SOLUTION.md', 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Show first 2000 characters to avoid overwhelming
                    if len(content) > 2000:
                        print(content[:2000])
                        print("\n... (truncated, see FINAL_SOLUTION.md for full guide)")
                    else:
                        print(content)
            except FileNotFoundError:
                print("❌ FINAL_SOLUTION.md not found.")
                print("💡 Run option 3 (Fix API Connection) to create it.")
            
        elif choice == 8:
            print("👋 Goodbye!")
            break
        
        # Ask if user wants to continue
        print("\n" + "=" * 50)
        continue_choice = input("Do you want to return to menu? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Startup interrupted by user")
    except Exception as e:
        print(f"\n💥 Startup error: {e}")
        traceback.print_exc()
