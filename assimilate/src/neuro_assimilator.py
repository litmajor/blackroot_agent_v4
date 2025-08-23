from black_vault import BlackVault
import ast
import random
import hashlib
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Union
import psutil
import os
import subprocess
import tempfile
import importlib.util
import signal
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple alternatives to restrictedpython for demonstration
def safe_compile(source: str, filename: str = '<string>', mode: str = 'exec') -> Any:
    """Safe compilation of Python code with basic restrictions."""
    # Parse and validate AST
    tree = ast.parse(source)
    # Basic validation - in real implementation, add more checks
    return compile(tree, filename, mode)

safe_globals = {
    '__builtins__': {
        'print': print, 'len': len, 'range': range, 'str': str, 
        'int': int, 'float': float, 'bool': bool, 'list': list,
        'dict': dict, 'tuple': tuple, 'set': set
    }
}

@dataclass
class TrustMetrics:
    """Trust evaluation metrics for code assessment."""
    code_complexity: float = 0.0
    source_reputation: float = 0.0
    execution_history: float = 0.0
    security_score: float = 0.0

class TrustMatrix:
    """Evaluates trust scores for foreign code."""
    
    def __init__(self):
        self.trust_history: Dict[str, float] = {}
        
    def evaluate(self, code: Dict[str, Any]) -> float:
        """Evaluate trust score for given code."""
        metrics = self._analyze_code(code)
        
        # Weighted trust calculation
        trust_score = (
            metrics.code_complexity * 0.2 +
            metrics.source_reputation * 0.3 +
            metrics.execution_history * 0.3 +
            metrics.security_score * 0.2
        )
        
        # Store in history
        code_id = hashlib.sha256(str(code).encode()).hexdigest()
        self.trust_history[code_id] = trust_score
        
        return min(1.0, max(0.0, trust_score))
    
    def _analyze_code(self, code: Dict[str, Any]) -> TrustMetrics:
        """Analyze code and return trust metrics."""
        metrics = TrustMetrics()
        
        # Simple complexity analysis
        source = code.get('source', '')
        metrics.code_complexity = min(1.0, len(source) / 1000)  # Normalize by length
        
        # Default reputation for unknown sources
        metrics.source_reputation = 0.5
        
        # No execution history initially
        metrics.execution_history = 0.5
        
        # Improved security score: whitelist-based
        safe_tokens = {'def', 'return', 'f"', 'print', 'len', 'range'}
        danger_count = sum(t not in safe_tokens for t in source.split())
        metrics.security_score = max(0.0, 1.0 - danger_count * 0.01)
        
        return metrics

class SecurityManager:
    """Enhanced security manager for code execution."""
    
    def __init__(self):
        self.dangerous_functions = {
            'exec', 'eval', 'compile', '__import__', 'open', 'file',
            'subprocess', 'os.system', 'os.popen'
        }
        
    def analyze_code(self, code: str) -> bool:
        """Perform static analysis on code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.dangerous_functions:
                            logger.warning(f"Dangerous function detected: {node.func.id}")
                            return False
            return True
        except SyntaxError:
            return False
    
    def execute_safely(self, code: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute code in a restricted environment."""
        context = context or {}

        # Set resource limits (POSIX only)
        try:
            if os.name == "posix":
                resource.setrlimit(resource.RLIMIT_CPU, (5, 5))  # 5 seconds CPU time
                resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))  # 128MB memory
        except (AttributeError, OSError):
            pass  # Resource limits not available on all systems

        # Execute with timeout (POSIX only)
        if os.name == "posix":
            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timeout")
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 second timeout
        else:
            old_handler = None

        try:
            local_vars = {}
            exec(code, safe_globals, local_vars)
            return local_vars.get('result', True)
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
        finally:
            if os.name == "posix":
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

class ReliabilityMonitor:
    """Monitor system reliability and performance."""
    
    def __init__(self):
        self.failure_count = 0
        self.success_count = 0
        
    def record_success(self):
        self.success_count += 1
        
    def record_failure(self):
        self.failure_count += 1
        
    def get_reliability_score(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0

class PerformanceProfiler:
    """Profile and monitor performance metrics."""
    
    def __init__(self):
        self.execution_times: List[float] = []
        
    def record_execution_time(self, duration: float):
        self.execution_times.append(duration)
        if len(self.execution_times) > 100:  # Keep only last 100 records
            self.execution_times.pop(0)
            
    def calculate_performance_score(self) -> float:
        if not self.execution_times:
            return 1.0
        avg_time = sum(self.execution_times) / len(self.execution_times)
        return max(0.0, 1.0 - min(avg_time / 10.0, 1.0))  # Normalize to 0-1

class PluginManager:
    """Manage plugins and extensions."""
    
    def __init__(self):
        self.plugins: Dict[str, Any] = {}
        
    def register_plugin(self, name: str, plugin: Any):
        self.plugins[name] = plugin
        logger.info(f"Registered plugin: {name}")
        
    def get_plugin(self, name: str) -> Optional[Any]:
        return self.plugins.get(name)

class SoftwareHandler(ABC):
    """Abstract base class for software handlers."""
    
    @abstractmethod
    def execute(self, code_obj: Dict[str, Any], context: Dict[str, Any]) -> Any:
        pass

class PythonScriptHandler(SoftwareHandler):
    """Handler for Python scripts."""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
    
    def execute(self, code_obj: Dict[str, Any], context: Dict[str, Any]) -> Any:
        source = code_obj.get('source', '')
        if not self.security_manager.analyze_code(source):
            raise ValueError("Code failed security analysis")
        
        compiled_code = safe_compile(source)
        return self.security_manager.execute_safely(compiled_code, context)

class BinaryHandler(SoftwareHandler):
    """Handler for binary executables."""
    
    def execute(self, code_obj: Dict[str, Any], context: Dict[str, Any]) -> Any:
        # Simulate binary execution (actual implementation would be platform-specific)
        logger.info(f"Simulating execution of binary: {code_obj.get('name', 'unknown')}")
        return True

class RustScriptHandler(SoftwareHandler):
    """Handler for Rust code execution."""
    
    def execute(self, code_obj: Dict[str, Any], context: Dict[str, Any]) -> Any:
        logger.info("Simulating Rust code execution")
        # In real implementation, this would compile and execute Rust code
        return True

# Create handler instances
security_manager = SecurityManager()
SOFTWARE_HANDLERS = {
    'python_script': PythonScriptHandler(security_manager),
    'binary': BinaryHandler(),
    'shellcode': BinaryHandler(),
    'dll': BinaryHandler(),
    'elf': BinaryHandler(),
    'macho': BinaryHandler(),
    'exe': BinaryHandler(),
    'pyd': BinaryHandler(),
    'so': BinaryHandler(),
    'arm64': BinaryHandler(),
    'rust': RustScriptHandler()
}

class PayloadEngine:
    """Multi-format payload engine with security considerations."""
    
    def __init__(self):
        self.supported_formats = ['shellcode', 'dll', 'elf', 'macho', 'arm64', 'exe', 'pyd', 'so', 'rust']
        self.security_manager = SecurityManager()

    def load(self, path: str, fmt: str) -> bytes:
        """Load payload from file with format validation."""
        if fmt.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported payload format: {fmt}")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Payload file not found: {path}")
            
        try:
            with open(path, 'rb') as f:
                data = f.read()
            logger.info(f"Loaded {fmt.upper()} payload ({len(data)} bytes)")
            return data
        except Exception as e:
            logger.error(f"Failed to load payload: {e}")
            raise

import os, tempfile, subprocess, shutil, stat, logging, platform, sys
from typing import Optional
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

logger = logging.getLogger("PayloadEngine")

def _real_shellcode_emulation(self, sc: bytes) -> bool:
    """
    Emulates x86/x64 shellcode inside Unicorn Engine.
    Returns True if emulation reaches the last byte without crash.
    """
    try:
        from unicorn.unicorn import Uc
        from unicorn import UC_ARCH_X86, UC_MODE_64
        from unicorn.x86_const import UC_X86_REG_RIP
    except ImportError:
        logger.error("pip install unicorn")
        return False

    mu = Uc(UC_ARCH_X86, UC_MODE_64)
    BASE = 0x400000
    STACK = 0x500000
    mu.mem_map(BASE, 0x1000)
    mu.mem_map(STACK - 0x1000, 0x1000)
    mu.mem_write(BASE, sc.ljust(0x1000, b"\x00"))
    mu.reg_write(UC_X86_REG_RIP, BASE)

    try:
        mu.emu_start(BASE, BASE + len(sc), timeout=1000)
        logger.info("Shellcode emulated successfully")
        return True
    except Exception as e:
        logger.warning(f"Emulation failed: {e}")
        return False

def _real_dll_test(self, dll: bytes, target_proc: str = "rundll32.exe") -> bool:
    """
    Spawns a *new* process and loads the DLL there.
    Returns True if LoadLibrary succeeds (no remote injection).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dll") as f:
        f.write(dll)
        dll_path = f.name

    try:
        # rundll32.exe  dll_path,EntryPoint
        subprocess.run([target_proc, dll_path, "#1"], timeout=5, check=True)
        logger.info("DLL loaded safely in sacrificial process")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"DLL load failed: {e}")
        return False
    finally:
        os.unlink(dll_path)



def _real_firecracker_run(self, binary: bytes, fmt: str) -> bool:
    """
    Writes binary to a tmp file, copies into a pre-baked micro-VM,
    runs it, and returns the exit code.
    Requires firecracker binary and a minimal rootfs.
    """
    from pathlib import Path
    vm_dir = Path("/opt/fc_images")            # pre-built rootfs & kernel
    with tempfile.NamedTemporaryFile(dir=vm_dir / "overlay", delete=False) as f:
        f.write(binary)
        guest_path = f"/overlay/{f.name.split('/')[-1]}"

    cmd = [
        "firecracker", "--no-api",
        "--kernel", str(vm_dir / "vmlinux"),
        "--rootfs", str(vm_dir / "rootfs.ext4"),
        "--overlay", guest_path
    ]
    try:
        result = subprocess.run(cmd, timeout=10, check=True)
        logger.info(f"{fmt.upper()} executed in micro-VM")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.warning(f"Micro-VM run failed: {e}")
        return False


def _real_python_binding(self, binary: bytes, module_name: str) -> bool:
    """
    Drops the .pyd/.so into a temp dir, spawns a fresh Python
    process, and imports it.
    """
    suffix = ".pyd" if platform.system() == "Windows" else ".so"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(binary)
        mod_path = f.name

    try:
        subprocess.run(
            [sys.executable, "-c", f"import {module_name}; print('OK')"],
            cwd=os.path.dirname(mod_path),
            timeout=5,
            check=True
        )
        logger.info("Python binding imported safely")
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        os.unlink(mod_path)

def _real_rust_run(self, rust_src: bytes) -> bool:
    """
    1. Writes src to main.rs
    2. cargo run --quiet
    3. Returns True if compilation & run succeed.
    """
    with tempfile.TemporaryDirectory() as d:
        src_path = os.path.join(d, "main.rs")
        with open(src_path, "wb") as f:
            f.write(rust_src)
        try:
            subprocess.run(
                ["rustc", src_path, "-o", "main"],
                cwd=d, check=True, timeout=15
            )
            subprocess.run([os.path.join(d, "main")], check=True, timeout=5)
            logger.info("Rust code compiled & executed successfully")
            return True
        except subprocess.CalledProcessError:
            return False

def _real_qemu_arm64(self, patch: bytes) -> bool:
    """
    Runs ARM64 code under QEMU user-mode on x86_64 host.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".elf") as f:
        f.write(patch)
        elf_path = f.name
        os.chmod(elf_path, 0o755)

    try:
        subprocess.run(["qemu-aarch64", "-L", "/usr/aarch64-linux-gnu", elf_path],
                       check=True, timeout=5)
        logger.info("ARM64 ELF executed under QEMU")
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        os.unlink(elf_path)

def inject(self, payload: bytes, fmt: str, pid: Optional[int] = None,
           module_name: Optional[str] = None) -> bool:
    fmt = fmt.lower()
    mapping = {
        "shellcode": lambda: self._real_shellcode_emulation(payload),
        "dll": lambda: self._real_dll_test(payload),
        "elf": lambda: self._real_firecracker_run(payload, "elf"),
        "macho": lambda: self._real_qemu_arm64(payload),   # reuse QEMU
        "exe": lambda: self._real_firecracker_run(payload, "exe"),
        "pyd": lambda: self._real_python_binding(payload, module_name or "testmod"),
        "so": lambda: self._real_python_binding(payload, module_name or "testmod"),
        "rust": lambda: self._real_rust_run(payload),
        "arm64": lambda: self._real_qemu_arm64(payload),
    }
    handler = mapping.get(fmt)
    if not handler:
        raise ValueError(f"Unsupported format: {fmt}")
    return handler()

class NeuroAssimilatorAgent:

    def execute_blob(self, blob: bytes, context: Optional[dict] = None) -> Any:
        """
        Execute a code blob (bytes) in a safe context.
        Decodes to utf-8, compiles, and runs using execute_safely.
        """
        if context is None:
            context = {}
        try:
            code_str = blob.decode('utf-8')
        except Exception as e:
            return {"error": f"Blob decode failed: {e}"}
        try:
            compiled = compile(code_str, '<blob>', 'exec')
            return self.execute_safely(compiled, context)  # type: ignore[attr-defined]
        except Exception as e:
            return {"error": f"Blob execution failed: {e}"}
    """Advanced agent for code analysis and execution."""
    
    MEMORY_CAPACITY: int = 50
    TRUST_THRESHOLD: float = 0.6
    HIGH_CPU_THRESHOLD: float = 80.0
    PERFORMANCE_THRESHOLD: float = 0.5
    ADAPTATION_MEMORY_THRESHOLD: int = 5
    ADAPTATION_TRIGGER_COUNT: int = 2
    ADAPTATION_TIME_WINDOW: int = 3600  # seconds
    ADAPTATION_MEMORY_LIMIT: int = 1024 * 1024 * 100  # 100 MB
    ADAPTATION_CPU_LIMIT: float = 80.0  # 80%

    def __init__(
        self,
        trust_matrix: Optional[TrustMatrix] = None,
        tactical_codex: Optional[Dict[str, Any]] = None,
        traits: Optional[Dict[str, float]] = None,
        reflex_tree: Optional[List[Dict]] = None,
        vault_password: Optional[str] = None,
        vault_rotate_days: int = 7
    ):
        self.vault = BlackVault(password=vault_password or "change-me", rotate_days=vault_rotate_days)
        self.trust_matrix = trust_matrix or TrustMatrix()
        self.codex = tactical_codex or {}
        self.traits = traits or self._randomize_traits()
        self.reflex_tree = reflex_tree or self._generate_reflex_tree()
        self.memory: List[Dict[str, Any]] = []
        self.performance_log: Dict[str, List[float]] = {}
        self.security_manager = SecurityManager()
        self.reliability_monitor = ReliabilityMonitor()
        self.performance_profiler = PerformanceProfiler()
        self.plugin_manager = PluginManager()

        # Auto-clean expired artifacts on agent start
        self.vault.clean_expired()

    def observe(self, system_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Observe current system state."""
        system_state = system_state or {}
        observation: Dict[str, Any] = {
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'timestamp': time.time(),
            'performance_score': self.performance_profiler.calculate_performance_score()
        }
        observation.update(system_state)


        # Top 5 CPU-consuming processes
        try:
            procs = [(p.pid, p.name(), p.cpu_percent()) for p in psutil.process_iter(['pid', 'name', 'cpu_percent'])]
            top_procs = sorted(procs, key=lambda x: x[2], reverse=True)[:5]
            observation['top_cpu_processes'] = [{'pid': p[0], 'name': p[1], 'cpu': p[2]} for p in top_procs]
        except Exception:
            observation['top_cpu_processes'] = []

        # Disk I/O pressure
        io = psutil.disk_io_counters()
        if io is not None:
            observation['disk_read_MB']  = round(io.read_bytes  / 1024 / 1024, 2)
            observation['disk_write_MB'] = round(io.write_bytes / 1024 / 1024, 2)
        else:
            observation['disk_read_MB'] = None
            observation['disk_write_MB'] = None

        # Network chatter
        net = psutil.net_io_counters()
        if net is not None:
            observation['net_rx_MB'] = round(net.bytes_recv / 1024 / 1024, 2)
            observation['net_tx_MB'] = round(net.bytes_sent / 1024 / 1024, 2)
        else:
            observation['net_rx_MB'] = None
            observation['net_tx_MB'] = None

        # Thermal & battery (laptops / ARM boxes)
        cpu_temp = None
        sensors_temperatures = getattr(psutil, "sensors_temperatures", None)
        if sensors_temperatures is not None:
            try:
                temps = sensors_temperatures()
                if temps:
                    first_sensor = next(iter(temps.values()), None)
                    if first_sensor and len(first_sensor) > 0 and hasattr(first_sensor[0], 'current'):
                        cpu_temp = first_sensor[0].current
            except Exception:
                pass
        observation['cpu_temp_C'] = cpu_temp

        battery_pct = None
        if hasattr(psutil, "sensors_battery"):
            try:
                bat = psutil.sensors_battery()
                if bat is not None and hasattr(bat, 'percent'):
                    battery_pct = bat.percent
            except Exception:
                pass
        observation['battery_pct'] = battery_pct

        # Docker / container count
        docker_pipe = "//./pipe/dockerDesktopLinuxEngine"
        if os.path.exists(docker_pipe):
            try:
                import json
                cnt = len(json.loads(subprocess.check_output(["docker", "ps", "-q"], timeout=2)))
                observation['running_containers'] = cnt
            except Exception:
                observation['running_containers'] = None
        else:
            observation['running_containers'] = None

        # Let plugins contribute observations
        for name, plugin in self.plugin_manager.plugins.items():
            if hasattr(plugin, "observe"):
                observation.update(plugin.observe())

        self.memory.append(observation)
        if len(self.memory) > self.MEMORY_CAPACITY:
            self.memory.pop(0)

        return observation

    def decide(self, observation: Dict[str, Any]) -> str:
        """Make decision based on observation."""
        for reflex in self.reflex_tree:
            if reflex['condition'](observation):
                return reflex['action']
        return 'idle'

    def act(self, action: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the decided action."""
        context = context or {}
        action_map = {
            'execute_code': lambda: self._execute_code(context),
            'optimize': self._optimize,
            'idle': self._idle
        }
        action_func = action_map.get(action, self._idle)
        return action_func()

    def _execute_code(self, context: Dict[str, Any]) -> Any:
        """Execute code from the tactical codex."""
        results = {}
        for code_hash, code_obj in self.codex.items():
            start_time = time.time()
            try:
                handler = SOFTWARE_HANDLERS.get(code_obj['type'])
                if not handler:
                    logger.error(f"No handler for type: {code_obj['type']}")
                    continue
                    
                code_len = len(code_obj['code']) if hasattr(code_obj['code'], '__len__') else 'unknown'
                logger.info(f"About to execute {code_obj['name']} ({code_len} bytes)")
                result = handler.execute(code_obj, context)
                elapsed = time.time() - start_time
                
                # Record performance
                self.performance_profiler.record_execution_time(elapsed)
                performance = 1.0 / (1.0 + elapsed) if result else 0.0
                self.performance_log.setdefault(code_hash, []).append(performance)
                
                results[code_hash] = result
                self.reliability_monitor.record_success()
                logger.info(f"Executed {code_obj['name']}: {result}")

                # Store payload in vault if execution was successful (exit_code == 0 or result is True)
                artifact_name = code_obj.get('name', code_hash)
                payload_bytes = code_obj.get('source') if isinstance(code_obj.get('source'), bytes) else None
                if (result is True or result == 0) and payload_bytes:
                    self.vault.store(artifact_name, payload_bytes, expire_minutes=1440)
                    logger.info(f"Stored {artifact_name} in vault after successful execution.")
                
            except Exception as e:
                self.performance_log.setdefault(code_hash, []).append(0.0)
                self.reliability_monitor.record_failure()
                logger.error(f"Error executing {code_obj['name']}: {e}")
                
        return results

    def retrieve_payload(self, artifact_name: str) -> Optional[bytes]:
        """Retrieve payload from vault for ghostlayer or other use."""
        try:
            return self.vault.retrieve_for_ghostlayer(artifact_name)
        except Exception as e:
            logger.error(f"Failed to retrieve {artifact_name} from vault: {e}")
            return None

    def audit_vault(self):
        """List and audit all artifacts in the vault."""
        for name_hash in self.vault.list_artifacts():
            try:
                data = self.vault.retrieve(name_hash)
                print(name_hash, len(data))
            except Exception as e:
                print(name_hash, f"error: {e}")

    def _optimize(self) -> None:
        """Optimize system resources."""
        logger.info("Lowering priority of top CPU hogs…")
        warned_pids = getattr(self, '_warned_pids', set())
        for p in sorted(psutil.process_iter(['pid', 'name', 'cpu_percent']),
                        key=lambda x: x.info['cpu_percent'] or 0, reverse=True)[:3]:
            try:
                # Ignore System Idle Process
                if p.name().lower() == "system idle process":
                    continue
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
                logger.info(f"Lowered priority for PID {p.pid} ({p.name()}); now at {p.nice()}")
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                if p.pid not in warned_pids:
                    logger.warning(f"Could not lower priority for PID {p.pid}")
                    warned_pids.add(p.pid)
        self._warned_pids = warned_pids

    def _idle(self) -> None:
        """Idle state - minimal activity."""
        logger.debug("Agent idling...")

    def _randomize_traits(self) -> Dict[str, float]:
        """Generate random traits for the agent."""
        return {
            'efficiency': random.uniform(0.3, 0.9),
            'adaptability': random.uniform(0.5, 1.0),
            'reliability': random.uniform(0.1, 1.0)
        }

    def _generate_reflex_tree(self) -> List[Dict]:
        """Generate reflex tree for decision making."""
        return [
            {
                'condition': lambda obs: obs.get('cpu_usage', 0) > self.HIGH_CPU_THRESHOLD,
                'action': 'optimize'
            },
            {
                'condition': lambda obs: obs.get('cpu_temp_C', 0) is not None and obs.get('cpu_temp_C', 0) > 75,
                'action': 'optimize'
            },
            {
                'condition': lambda obs: obs.get('performance_score', 1) < self.PERFORMANCE_THRESHOLD,
                'action': 'optimize'
            },
            {
                'condition': lambda obs: len(self.codex) > 0,
                'action': 'execute_code'
            }
        ]

    def discover_and_assimilate(self, foreign_code: Dict[str, Any]) -> Optional[Any]:
        """Discover and assimilate foreign code if trusted."""
        trust_score = self.trust_matrix.evaluate(foreign_code)
        identity_hash = self._fingerprint(foreign_code)
        
        if trust_score >= self.TRUST_THRESHOLD and identity_hash not in self.codex:
            logger.info(f"Trust score {trust_score:.2f} for '{foreign_code['name']}'. Assimilating...")
            code_obj = self._rewrite_and_absorb(foreign_code)
            if code_obj:
                self.codex[identity_hash] = {
                    'name': foreign_code['name'],
                    'type': foreign_code.get('type', 'python_script'),
                    'code': code_obj,
                    'source': foreign_code.get('source', '')
                }
                return code_obj
        
        logger.info(f"'{foreign_code['name']}' not assimilated (Trust: {trust_score:.2f}, Known: {identity_hash in self.codex})")
        return None

    def _rewrite_and_absorb(self, foreign_code: Dict[str, Any]) -> Optional[Any]:
        """Rewrite and absorb foreign code safely."""
        try:
            code_type = foreign_code.get('type', 'python_script')
            
            if code_type == 'python_script':
                source = foreign_code.get('source', '')
                if not self.security_manager.analyze_code(source):
                    logger.warning("Code failed security analysis")
                    return None
                    
                tree = ast.parse(source)
                sanitized_ast = self._Sanitizer().visit(tree)
                adapted_ast = self._ControlInjector().visit(sanitized_ast)
                ast.fix_missing_locations(adapted_ast)
                return safe_compile(ast.unparse(adapted_ast) if hasattr(ast, 'unparse') else source)
            
            elif code_type in SOFTWARE_HANDLERS:
                return {
                    'name': foreign_code['name'], 
                    'source': foreign_code.get('source', ''),
                    'type': code_type
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to assimilate '{foreign_code['name']}': {e}")
            return None

    class _Sanitizer(ast.NodeTransformer):
        """Remove dangerous imports and calls."""
        
        def visit_Import(self, node: ast.Import) -> Optional[ast.Import]:
            # Allow only safe imports
            safe_modules = {'math', 'time', 'random', 'datetime'}
            node.names = [alias for alias in node.names if alias.name in safe_modules]
            return node if node.names else None
            
        def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.ImportFrom]:
            # Allow only safe modules
            safe_modules = {'math', 'time', 'random', 'datetime'}
            if node.module in safe_modules:
                return node
            return None

    class _ControlInjector(ast.NodeTransformer):
        """Inject control hooks into functions."""
        
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            # Inject logging hook
            hook = ast.parse("print('[CONTROL] Function executed')").body[0]
            node.body.insert(0, hook)
            return node

    def _fingerprint(self, code: Dict[str, Any]) -> str:
        """Generate fingerprint for code identification."""
        key = f"{code['name']}::{code.get('source', '')}"
        return hashlib.sha256(key.encode()).hexdigest()

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    import getpass
    vault_password = getpass.getpass("Vault password (leave blank for default): ") or None
    agent = NeuroAssimilatorAgent(vault_password=vault_password)
   
    sample_code = {
        'name': 'hello_world',
        'type': 'python_script',
        'source': '''
def greet(name):
    return f"Hello, {name}!"

result = greet("World")
print(result)
'''
    }
    
    # Attempt to assimilate the code
    result = agent.discover_and_assimilate(sample_code)
    if result:
        print("Code successfully assimilated!")
    
    # Observe system state
    observation = agent.observe()
    print(f"System observation: {observation}")
    
    # Make decision and act
    decision = agent.decide(observation)
    logger.info(f"Loop: {decision}")
    print(f"Decision: {decision}")
    
    action_result = agent.act(decision)
    print(f"Action result: {action_result}")

    # Integrate and test with ghost-utility binary (robust check)
    import os, hashlib
    path = r'E:\repos\keylogger\target\debug\ghost-utility.exe'
    assert os.path.exists(path), f"file missing: {path}"
    with open(path, 'rb') as f:
        blob = f.read()
    print(f"Loaded ghost-utility.exe: {len(blob)} bytes")
    agent.codex = {}  # wipe stale entries
    agent.discover_and_assimilate({
        'name': 'ghost-utility',
        'type': 'exe',
        'source': blob
    })

    # Print codex entry size by hash
    for k in agent.codex:
        print(f"codex key: {k} size: {len(agent.codex[k]['source'])}")
    agent.act('execute_code')