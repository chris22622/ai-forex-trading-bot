#!/usr/bin/env python3
"""
VPN Connection Check and Bot Launcher
Checks VPN status and provides guidance before starting the bot
"""

import asyncio
import subprocess
import json
import sys

def check_ip_location():
    """Check current IP location to verify VPN status"""
    print("🌍 Checking current IP location...")
    
    try:
        # Use curl to check IP info (Windows PowerShell compatible)
        result = subprocess.run([
            "powershell", "-Command", 
            "Invoke-RestMethod -Uri 'http://ipinfo.io/json'"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            try:
                ip_info = json.loads(result.stdout)
                print(f"📍 Current Location: {ip_info.get('city', 'Unknown')}, {ip_info.get('country', 'Unknown')}")
                print(f"🌐 IP Address: {ip_info.get('ip', 'Unknown')}")
                print(f"🏢 ISP: {ip_info.get('org', 'Unknown')}")
                
                # Check if we're using a VPN (common VPN indicators)
                location = ip_info.get('country', '').lower()
                org = ip_info.get('org', '').lower()
                
                vpn_indicators = ['vpn', 'proxy', 'datacenter', 'hosting', 'cloud']
                is_likely_vpn = any(indicator in org for indicator in vpn_indicators)
                
                return {
                    'country': ip_info.get('country', 'Unknown'),
                    'city': ip_info.get('city', 'Unknown'),
                    'ip': ip_info.get('ip', 'Unknown'),
                    'org': ip_info.get('org', 'Unknown'),
                    'is_likely_vpn': is_likely_vpn
                }
            except json.JSONDecodeError:
                print(f"❌ Failed to parse IP info: {result.stdout}")
                return None
        else:
            print(f"❌ Failed to get IP info: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error checking IP: {e}")
        return None

async def quick_deriv_test():
    """Quick test to see if Deriv is accessible"""
    print("\n🧪 Quick Deriv accessibility test...")
    
    try:
        import websockets
        uri = "wss://ws.derivws.com/websockets/v3"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"time": 1, "req_id": 1}))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(response)
            
            if "time" in data:
                print("✅ Deriv WebSocket is accessible!")
                return True
            else:
                print(f"❓ Unexpected response: {data}")
                return False
                
    except Exception as e:
        if "401" in str(e):
            print("❌ Deriv blocked (HTTP 401) - VPN required")
        else:
            print(f"❌ Connection failed: {e}")
        return False

def print_vpn_recommendations():
    """Print VPN setup recommendations"""
    print("\n🔧 VPN SETUP RECOMMENDATIONS")
    print("=" * 50)
    print("🌍 RECOMMENDED COUNTRIES:")
    print("   ✅ Germany (Frankfurt, Berlin)")
    print("   ✅ United Kingdom (London)")
    print("   ✅ Netherlands (Amsterdam)")
    print("   ✅ Singapore")
    print("   ✅ United States (New York, Atlanta)")
    print("\n❌ AVOID THESE LOCATIONS:")
    print("   ❌ Caribbean islands")
    print("   ❌ Jamaica") 
    print("   ❌ African countries")
    print("   ❌ Some Asian countries")
    print("\n📱 RECOMMENDED VPN SERVICES:")
    print("   🥇 IPVanish (you have this)")
    print("   🥈 NordVPN")
    print("   🥉 ProtonVPN")
    print("   🏆 ExpressVPN")

def main():
    print("🚀 DERIV TRADING BOT - VPN CHECK & LAUNCHER")
    print("=" * 60)
    
    # Check current IP location
    ip_info = check_ip_location()
    
    if ip_info:
        print(f"\n📊 CONNECTION ANALYSIS")
        print("=" * 30)
        
        if ip_info['is_likely_vpn']:
            print("✅ Likely using VPN/Proxy")
        else:
            print("⚠️ Likely using direct connection")
            
        # Country-specific recommendations
        country = ip_info['country'].upper()
        if country in ['DE', 'GERMANY', 'GB', 'UK', 'NL', 'NETHERLANDS', 'SG', 'SINGAPORE', 'US', 'USA']:
            print(f"✅ {country} is a good location for Deriv trading")
        else:
            print(f"⚠️ {country} might be blocked by Deriv")
    
    # Test Deriv accessibility
    print("\n" + "=" * 60)
    deriv_accessible = asyncio.run(quick_deriv_test())
    
    if deriv_accessible:
        print("\n🎉 READY TO TRADE!")
        print("=" * 20)
        print("✅ Deriv is accessible from your current connection")
        print("✅ Your new API tokens are ready")
        print("\n🚀 LAUNCH THE BOT:")
        print("   python main.py")
        print("\n💡 OR run with enhanced startup:")
        print("   python start_bot.py")
        
    else:
        print("\n🚨 VPN REQUIRED!")
        print("=" * 20)
        print("❌ Deriv is not accessible from your current connection")
        print("🔧 You need to connect to a VPN first")
        
        print_vpn_recommendations()
        
        print("\n📋 STEP-BY-STEP SOLUTION:")
        print("1. 🔌 Connect to IPVanish VPN")
        print("2. 🌍 Choose Germany, UK, Netherlands, or Singapore")
        print("3. 🧪 Run this script again: python vpn_check.py")
        print("4. 🚀 Once accessible, run: python main.py")
    
    print("\n" + "=" * 60)
    print("Your new API tokens are configured and ready!")
    print("Demo: s8Sww40XKyKMp0x")
    print("Live: g9kxuLTUWxrHGNe")
    print("=" * 60)

if __name__ == "__main__":
    main()
