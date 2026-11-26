# agents/phone_wallet_agent.py
import os, json, uuid, logging, requests, re, subprocess, hashlib
import platform
from agents.base import BaseAgent
from economic.wallet import WalletManager
from agents.stealth_manager import StealthManager

class PhoneWalletAgent(BaseAgent):
    def should_activate(self, config=None):
        """Decide if agent should activate based on config or environment."""
        # Example logic: activate if phone_seed is present and wallet manager is ready
        phone_ready = self.discover_phone() is not None
        wallet_ready = hasattr(self.wallet_manager, 'create_wallet')
        return phone_ready and wallet_ready
    def __init__(self):
        super().__init__(codename="PX-WALLET")
        self.agent_id = f"PW-{uuid.uuid4().hex[:8]}"
        try:
            from core.redis_client import get_redis_client
            r_client = get_redis_client()
            self.wallet_manager = WalletManager(redis_client=r_client)
        except Exception:
            try:
                from redis import Redis
                self.wallet_manager = WalletManager(redis_client=Redis())
            except Exception as exc:
                logging.getLogger("PhoneWalletAgent").warning(f"WalletManager init fallback: {exc}")
                self.wallet_manager = None
        self.phone_seed = None
        self.host_map = {}
        self.stealth_manager = StealthManager()

    def discover_phone(self):
        """Extract phone / SIM info per OS"""
        self.phone_seed = None
        if os.name == 'nt':
            # Windows -> WMI + AT commands
            try:
                out = subprocess.check_output(
                    ["powershell", "-c", "Get-WmiObject -Query \"SELECT * FROM Win32_POTSModem\""],
                    text=True, timeout=3)
                m = re.search(r'Phone Number\s*:\s*(\+?\d+)', out)
                if m:
                    self.phone_seed = m.group(1)[-4:]  # last 4 digits
            except Exception:
                pass
        else:
            # Linux/Android -> nmcli or sim-tool
            try:
                out = subprocess.check_output(["nmcli", "dev", "show"], text=True, timeout=3)
                m = re.search(r'GSM.*Number:\s*(\+?\d+)', out)
                if m:
                    self.phone_seed = m.group(1)[-4:]
            except Exception:
                pass
        return self.phone_seed

    def map_host(self):
        """Full host + network fingerprint"""
        import psutil, socket, uuid, platform
        self.host_map = {
            "hostname": socket.gethostname(),
            "mac": ':'.join(re.findall('..', '%012x' % uuid.getnode())),
            "ssid": self._current_wifi_ssid(),
            "os": platform.system(),
            "phone_suffix": self.discover_phone()
        }
        return self.host_map

    def _current_wifi_ssid(self):
        try:
            if platform.system() == "Windows":
                out = subprocess.check_output(["netsh", "wlan", "show", "profiles"], text=True)
                ssid = re.findall(r'All User Profile\s*:\s*(.+)', out)
                return ssid[0] if ssid else "unknown"
            else:
                out = subprocess.check_output(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"], text=True)
                ssid = [l.split(":")[1] for l in out.splitlines() if l.startswith("yes")]
                return ssid[0] if ssid else "unknown"
        except Exception:
            return "unknown"

    async def spawn_wallet(self, fund_amount=100):
        """Wallet seed = phone_suffix + MAC, fund after creation"""
        seed = (self.host_map.get("phone_suffix") or "0000") + self.host_map.get("mac", "").replace(':', '')
        networks = ["ethereum"]  # or set as appropriate for your use case
        wallets = await self.wallet_manager.create_wallet(agent_id=self.agent_id, networks=networks)
        wallet = wallets.get("ethereum")
        # Fund the wallet after creation
        try:
            await self.wallet_manager.update_balance(self.agent_id, "ethereum", native_balance=fund_amount)
            logging.info(f"[{self.agent_id}] funded wallet {getattr(wallet, 'address', None)} with ${fund_amount}")
        except Exception as e:
            logging.error(f"[{self.agent_id}] wallet funding failed: {e}")
        return wallet

    async def launder_cycle(self, amount_usd):
        """Split → micro-accounts → churn"""
        wallets = [await self.spawn_wallet() for _ in range(3)]
        for w in wallets:
            # dummy: push 1/3 of amount into each
            logging.info(f"[{self.agent_id}] laundering ${amount_usd/3:.2f} into {getattr(w, 'address', None)}")
        return wallets

    def social_bot(self, platform, action="follow", target_id=None):
        """Phone-verified account social bot"""
        payload = {
            "platform": platform,
            "action": action,
            "target_id": target_id,
            "phone_suffix": self.host_map["phone_suffix"]
        }
        # hit social API (stub)
        logging.info(f"[{self.agent_id}] {action} on {platform} for {target_id}")
        return payload

    def run(self, fund_amount=100):
        self.map_host()
        # Apply stealth before wallet ops
        stealth_applied = self.stealth_manager.apply_stealth()
        logging.info(f"[{self.agent_id}] Stealth applied: {stealth_applied}")
        # Call spawn_wallet synchronously
        import asyncio
        wallet = asyncio.run(self.spawn_wallet(fund_amount=fund_amount))
        logging.info(f"[{self.agent_id}] ready – host={self.host_map}, wallet={getattr(wallet, 'address', None)}")