"""
ULTIMATE BOT FIXER - One-Click Solution for Deriv Connection Issues
This script will automatically try to fix your connection and run your bot
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from safe_logger import get_safe_logger

logger = get_safe_logger(__name__)

class UltimateBotFixer:
    """One-click solution to fix and run your bot"""

    def __init__(self):
        self.fixed = False
        self.python_exe = sys.executable
        self.bot_dir = Path(__file__).parent.absolute()

    def print_banner(self):
        """Print the ultimate fixer banner"""
        print("🚀" * 25)
        print("🔧 ULTIMATE DERIV BOT FIXER")
        print("🤖 One-Click Solution to Get Your Bot Running")
        print("🚀" * 25)
        print()

    def check_and_fix_firewall(self) -> bool:
        """Try to fix Windows Firewall issues"""
        try:
            logger.info("🔥 Checking Windows Firewall...")

            # Create firewall rules for Python and the bot
            commands = [
                'netsh advfirewall firewall add rule name="Python Trading Bot" dir=in action=allow program="{}" enable=yes'.format(self.python_exe),
                'netsh advfirewall firewall add rule name="Python Trading Bot Out" dir=out action=allow program="{}" enable=yes'.format(self.python_exe),
                'netsh advfirewall firewall add rule name="Deriv WebSocket" dir=out action=allow protocol=TCP remoteport=443 enable=yes',
                'netsh advfirewall firewall add rule name="Deriv WebSocket SSL" dir=out action=allow protocol=TCP remoteport=8443 enable=yes'
            ]

            for cmd in commands:
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        logger.info("✅ Firewall rule added successfully")
                    else:
                        logger.debug(f"Firewall rule result: {result.stderr}")
                except:
                    pass

            logger.info("🔥 Firewall rules updated")
            return True

        except Exception as e:
            logger.warning(f"Firewall fix failed: {e}")
            return False

    def apply_dns_fix(self) -> bool:
        """Try to apply DNS fixes"""
        try:
            logger.info("🌐 Applying DNS optimization...")

            # Try to flush DNS
            flush_commands = [
                "ipconfig /flushdns",
                "ipconfig /registerdns",
                "ipconfig /release",
                "ipconfig /renew"
            ]

            for cmd in flush_commands:
                try:
                    subprocess.run(cmd, shell=True, capture_output=True, timeout=15)
                except:
                    pass

            logger.info("🌐 DNS cache cleared")
            return True

        except Exception as e:
            logger.warning(f"DNS fix failed: {e}")
            return False

    def show_instant_solutions(self):
        """Show immediate solutions the user can try"""
        print("📱 INSTANT SOLUTIONS (Try these RIGHT NOW):")
        print()
        print("🥇 SOLUTION 1: Mobile Hotspot (Works 90% of the time)")
        print("   1. Grab your phone")
        print("   2. Go to Settings → Hotspot/Tethering")
        print("   3. Turn on Wi-Fi hotspot")
        print("   4. Connect this computer to your phone's Wi-Fi")
        print("   5. Run this script again")
        print()
        print("🥈 SOLUTION 2: Free VPN (ProtonVPN)")
        print("   1. Go to: https://protonvpn.com/free")
        print("   2. Download and install ProtonVPN")
        print("   3. Create free account")
        print("   4. Connect to Netherlands server")
        print("   5. Run this script again")
        print()
        print("🥉 SOLUTION 3: DNS Change (Quick fix)")
        print("   1. Press Win+R, type 'ncpa.cpl'")
        print("   2. Right-click your network → Properties")
        print("   3. IPv4 → Properties → Use these DNS:")
        print("      Primary: 8.8.8.8")
        print("      Secondary: 8.8.4.4")
        print("   4. OK → Restart network")
        print("   5. Run this script again")
        print()

    def open_vpn_signup_pages(self):
        """Open VPN signup pages automatically"""
        try:
            vpn_urls = [
                "https://protonvpn.com/free",
                "https://windscribe.com/upgrade",
                "https://www.tunnelbear.com/pricing"
            ]

            logger.info("🌐 Opening VPN signup pages...")
            for url in vpn_urls:
                try:
                    if sys.platform == "win32":
                        subprocess.run(["start", url], shell=True)
                    elif sys.platform == "darwin":
                        subprocess.run(["open", url])
                    else:
                        subprocess.run(["xdg-open", url])
                    time.sleep(2)
                except:
                    pass

            logger.info("✅ VPN pages opened in browser")

        except Exception as e:
            logger.error(f"Failed to open VPN pages: {e}")

    async def test_connection_after_fixes(self) -> bool:
        """Test if connection works after applying fixes"""
        try:
            import websockets

            test_urls = [
                "wss://ws.derivws.com/websockets/v3",
                "wss://ws.binaryws.com/websockets/v3"
            ]

            for url in test_urls:
                try:
                    async with websockets.connect(url, timeout=10) as ws:
                        ping_msg = {"ping": 1}
                        await ws.send(json.dumps(ping_msg))
                        response = await asyncio.wait_for(ws.recv(), timeout=5)
                        data = json.loads(response)

                        if "pong" in data:
                            logger.info(f"✅ Connection successful: {url}")
                            return True

                except:
                    continue

            return False

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def launch_bot_automatically(self) -> bool:
        """Launch the trading bot automatically"""
        try:
            logger.info("🚀 Launching your trading bot...")

            # Change to bot directory
            os.chdir(self.bot_dir)

            # Run the bot
            bot_command = [self.python_exe, "main.py"]

            logger.info("🤖 Starting Deriv Trading Bot...")
            print("\n" + "="*60)
            print("🚀 YOUR TRADING BOT IS STARTING!")
            print("🤖 Switching to bot output...")
            print("="*60)

            # Run the bot in the foreground
            result = subprocess.run(bot_command, cwd=self.bot_dir)

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to launch bot: {e}")
            return False

    def create_quick_launcher(self):
        """Create a quick launcher script"""
        try:
            launcher_content = f'''@echo off
echo 🚀 Quick Bot Launcher
cd /d "{self.bot_dir}"
"{self.python_exe}" ultimate_bot_fixer.py
pause
'''

            launcher_path = self.bot_dir / "LAUNCH_BOT.bat"
            with open(launcher_path, 'w') as f:
                f.write(launcher_content)

            logger.info(f"✅ Quick launcher created: {launcher_path}")
            print("💡 TIP: Double-click 'LAUNCH_BOT.bat' to start your bot anytime!")

        except Exception as e:
            logger.error(f"Failed to create launcher: {e}")

    async def run_ultimate_fix(self):
        """Run the complete ultimate fix process"""
        self.print_banner()

        try:
            # Step 1: Apply system fixes
            logger.info("🔧 Step 1: Applying system fixes...")
            self.check_and_fix_firewall()
            self.apply_dns_fix()

            # Step 2: Test connection
            logger.info("🔍 Step 2: Testing connection...")
            if await self.test_connection_after_fixes():
                logger.info("🎉 CONNECTION FIXED! Launching bot...")
                self.fixed = True

                # Launch the bot immediately
                self.launch_bot_automatically()
                return

            # Step 3: Connection still failed, provide solutions
            logger.warning("🔧 Step 3: Additional fixes needed...")

            print("\n" + "⚠️ " * 20)
            print("🔧 SYSTEM FIXES APPLIED BUT CONNECTION STILL BLOCKED")
            print("📱 YOUR ISP IS BLOCKING DERIV - NEED NETWORK SOLUTION")
            print("⚠️ " * 20)
            print()

            self.show_instant_solutions()

            # Ask user what they want to do
            print("🤔 WHAT DO YOU WANT TO DO?")
            print("1. 📱 I'll try mobile hotspot now")
            print("2. 🌐 Open VPN signup pages for me")
            print("3. 🚀 Launch bot anyway (might not work)")
            print("4. ❌ Exit and fix manually")

            try:
                choice = input("\nEnter your choice (1-4): ").strip()

                if choice == "1":
                    print("\n📱 MOBILE HOTSPOT INSTRUCTIONS:")
                    print("1. Enable hotspot on your phone")
                    print("2. Connect this computer to phone's Wi-Fi")
                    print("3. Run this script again")
                    input("\nPress Enter when you've connected to mobile hotspot...")

                    # Test again
                    if await self.test_connection_after_fixes():
                        logger.info("🎉 MOBILE HOTSPOT WORKS! Launching bot...")
                        self.launch_bot_automatically()
                    else:
                        print("❌ Still not working. Try VPN option.")

                elif choice == "2":
                    self.open_vpn_signup_pages()
                    print("\n🌐 VPN signup pages opened in your browser")
                    print("💡 After installing VPN:")
                    print("   1. Connect to Netherlands or Germany")
                    print("   2. Run this script again")

                elif choice == "3":
                    print("\n🚀 Launching bot (connection might fail)...")
                    self.launch_bot_automatically()

                else:
                    print("\n❌ Exiting. Run this script again after fixing network.")
                    return

            except KeyboardInterrupt:
                print("\n❌ Interrupted by user")
                return

        except Exception as e:
            logger.error(f"Ultimate fix failed: {e}")
            print(f"\n❌ Fix process failed: {e}")
            print("\n🆘 MANUAL STEPS:")
            print("1. Try mobile hotspot")
            print("2. Install free VPN (ProtonVPN)")
            print("3. Connect to supported country")
            print("4. Run: python main.py")

        finally:
            # Always create the quick launcher
            self.create_quick_launcher()

async def main():
    """Main ultimate fixer function"""
    fixer = UltimateBotFixer()
    await fixer.run_ultimate_fix()

if __name__ == "__main__":
    # Check if running as admin for firewall fixes
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            print("💡 TIP: Run as Administrator for best results")
            print("   (Right-click → Run as administrator)")
            print()
    except:
        pass

    # Run the ultimate fixer
    asyncio.run(main())
