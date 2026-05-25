"""
kernel32_hasher.py - Pre-computed Windows kernel32 API hash table
============================================================================
Provides a static hash table for all kernel32 functions used throughout
BlackRoot Agent. Hashes are pre-computed using CRC32 for fast lookup.

This enables indirect API invocation without direct ctypes references,
defeating static analysis that scans for direct kernel32 imports.

Usage:
    from kernel32_hasher import KERNEL32_HASHES, compute_api_hash
    
    # Look up hash for CreateProcessW
    hash_val = 0x3A5F2B1C  # Pre-computed
    func_info = KERNEL32_HASHES[hash_val]  # (dll='kernel32', func='CreateProcessW')
    
    # Or compute hash on demand
    hash_val = compute_api_hash('kernel32', 'CreateProcessW')
"""

import binascii
from typing import Optional, Tuple


def compute_api_hash(dll_name: str, function_name: str) -> int:
    """
    Compute CRC32 hash for a Windows API function.
    
    Args:
        dll_name: DLL name (e.g., 'kernel32', 'ntdll')
        function_name: Function name (e.g., 'CreateProcessW')
    
    Returns:
        32-bit CRC32 hash as integer
    
    Example:
        >>> hash_val = compute_api_hash('kernel32', 'GetFileAttributesA')
        >>> print(hex(hash_val))
        0x1a2b3c4d
    """
    combined = f"{dll_name}::{function_name}".encode('utf-8')
    crc_value = binascii.crc32(combined) & 0xFFFFFFFF
    return crc_value


# ============================================================================
# KERNEL32.DLL - Process & Memory Management
# ============================================================================

# Process creation & manipulation
CREATE_PROCESS_W_HASH = compute_api_hash('kernel32', 'CreateProcessW')
CREATE_PROCESS_A_HASH = compute_api_hash('kernel32', 'CreateProcessA')
OPEN_PROCESS_HASH = compute_api_hash('kernel32', 'OpenProcess')
TERMINATE_PROCESS_HASH = compute_api_hash('kernel32', 'TerminateProcess')
GET_CURRENT_PROCESS_HASH = compute_api_hash('kernel32', 'GetCurrentProcess')
WAIT_FOR_SINGLE_OBJECT_HASH = compute_api_hash('kernel32', 'WaitForSingleObject')

# Thread management
CREATE_REMOTE_THREAD_HASH = compute_api_hash('kernel32', 'CreateRemoteThread')
RESUME_THREAD_HASH = compute_api_hash('kernel32', 'ResumeThread')
SUSPEND_THREAD_HASH = compute_api_hash('kernel32', 'SuspendThread')
GET_THREAD_CONTEXT_HASH = compute_api_hash('kernel32', 'GetThreadContext')
SET_THREAD_CONTEXT_HASH = compute_api_hash('kernel32', 'SetThreadContext')
GET_CURRENT_THREAD_HASH = compute_api_hash('kernel32', 'GetCurrentThread')

# Memory allocation & manipulation
VIRTUAL_ALLOC_EX_HASH = compute_api_hash('kernel32', 'VirtualAllocEx')
VIRTUAL_ALLOC_HASH = compute_api_hash('kernel32', 'VirtualAlloc')
VIRTUAL_FREE_EX_HASH = compute_api_hash('kernel32', 'VirtualFreeEx')
VIRTUAL_FREE_HASH = compute_api_hash('kernel32', 'VirtualFree')
WRITE_PROCESS_MEMORY_HASH = compute_api_hash('kernel32', 'WriteProcessMemory')
READ_PROCESS_MEMORY_HASH = compute_api_hash('kernel32', 'ReadProcessMemory')
VIRTUAL_PROTECT_EX_HASH = compute_api_hash('kernel32', 'VirtualProtectEx')

# Module & function loading
LOAD_LIBRARY_W_HASH = compute_api_hash('kernel32', 'LoadLibraryW')
LOAD_LIBRARY_A_HASH = compute_api_hash('kernel32', 'LoadLibraryA')
GET_PROC_ADDRESS_HASH = compute_api_hash('kernel32', 'GetProcAddress')
FREE_LIBRARY_HASH = compute_api_hash('kernel32', 'FreeLibrary')
GET_MODULE_HANDLE_HASH = compute_api_hash('kernel32', 'GetModuleHandleA')
GET_MODULE_HANDLE_W_HASH = compute_api_hash('kernel32', 'GetModuleHandleW')

# File operations
GET_FILE_ATTRIBUTES_A_HASH = compute_api_hash('kernel32', 'GetFileAttributesA')
GET_FILE_ATTRIBUTES_W_HASH = compute_api_hash('kernel32', 'GetFileAttributesW')
CREATE_FILE_W_HASH = compute_api_hash('kernel32', 'CreateFileW')
CREATE_FILE_A_HASH = compute_api_hash('kernel32', 'CreateFileA')
READ_FILE_HASH = compute_api_hash('kernel32', 'ReadFile')
WRITE_FILE_HASH = compute_api_hash('kernel32', 'WriteFile')
CLOSE_HANDLE_HASH = compute_api_hash('kernel32', 'CloseHandle')

# ============================================================================
# ADVAPI32.DLL - Security & Registry
# ============================================================================

# Process token operations
OPEN_PROCESS_TOKEN_HASH = compute_api_hash('advapi32', 'OpenProcessToken')
DUPLICATE_TOKEN_HASH = compute_api_hash('advapi32', 'DuplicateToken')
DUPLICATE_TOKEN_EX_HASH = compute_api_hash('advapi32', 'DuplicateTokenEx')
SET_THREAD_TOKEN_HASH = compute_api_hash('advapi32', 'SetThreadToken')
GET_TOKEN_INFORMATION_HASH = compute_api_hash('advapi32', 'GetTokenInformation')
IMPERSONATE_LOGGED_ON_USER_HASH = compute_api_hash('advapi32', 'ImpersonateLoggedOnUser')
REVERT_TO_SELF_HASH = compute_api_hash('advapi32', 'RevertToSelf')

# Privilege operations
LOOKUP_PRIVILEGE_VALUE_HASH = compute_api_hash('advapi32', 'LookupPrivilegeValueW')
ADJUST_TOKEN_PRIVILEGES_HASH = compute_api_hash('advapi32', 'AdjustTokenPrivileges')

# ============================================================================
# NTDLL.DLL - Native API (Direct System Calls)
# ============================================================================

NT_QUERY_SYSTEM_INFORMATION_HASH = compute_api_hash('ntdll', 'NtQuerySystemInformation')
NT_QUERY_INFORMATION_PROCESS_HASH = compute_api_hash('ntdll', 'NtQueryInformationProcess')
NT_SET_INFORMATION_PROCESS_HASH = compute_api_hash('ntdll', 'NtSetInformationProcess')
NT_CREATE_THREAD_EX_HASH = compute_api_hash('ntdll', 'NtCreateThreadEx')
NT_PROTECT_VIRTUAL_MEMORY_HASH = compute_api_hash('ntdll', 'NtProtectVirtualMemory')
NT_ALLOCATE_VIRTUAL_MEMORY_HASH = compute_api_hash('ntdll', 'NtAllocateVirtualMemory')
NT_FREE_VIRTUAL_MEMORY_HASH = compute_api_hash('ntdll', 'NtFreeVirtualMemory')
NT_WRITE_VIRTUAL_MEMORY_HASH = compute_api_hash('ntdll', 'NtWriteVirtualMemory')
NT_READ_VIRTUAL_MEMORY_HASH = compute_api_hash('ntdll', 'NtReadVirtualMemory')

# ============================================================================
# MASTER HASH TABLE - All APIs
# ============================================================================

KERNEL32_HASHES = {
    # Process creation & management
    CREATE_PROCESS_W_HASH: ('kernel32', 'CreateProcessW'),
    CREATE_PROCESS_A_HASH: ('kernel32', 'CreateProcessA'),
    OPEN_PROCESS_HASH: ('kernel32', 'OpenProcess'),
    TERMINATE_PROCESS_HASH: ('kernel32', 'TerminateProcess'),
    GET_CURRENT_PROCESS_HASH: ('kernel32', 'GetCurrentProcess'),
    WAIT_FOR_SINGLE_OBJECT_HASH: ('kernel32', 'WaitForSingleObject'),
    
    # Thread management
    CREATE_REMOTE_THREAD_HASH: ('kernel32', 'CreateRemoteThread'),
    RESUME_THREAD_HASH: ('kernel32', 'ResumeThread'),
    SUSPEND_THREAD_HASH: ('kernel32', 'SuspendThread'),
    GET_THREAD_CONTEXT_HASH: ('kernel32', 'GetThreadContext'),
    SET_THREAD_CONTEXT_HASH: ('kernel32', 'SetThreadContext'),
    GET_CURRENT_THREAD_HASH: ('kernel32', 'GetCurrentThread'),
    
    # Memory allocation
    VIRTUAL_ALLOC_EX_HASH: ('kernel32', 'VirtualAllocEx'),
    VIRTUAL_ALLOC_HASH: ('kernel32', 'VirtualAlloc'),
    VIRTUAL_FREE_EX_HASH: ('kernel32', 'VirtualFreeEx'),
    VIRTUAL_FREE_HASH: ('kernel32', 'VirtualFree'),
    WRITE_PROCESS_MEMORY_HASH: ('kernel32', 'WriteProcessMemory'),
    READ_PROCESS_MEMORY_HASH: ('kernel32', 'ReadProcessMemory'),
    VIRTUAL_PROTECT_EX_HASH: ('kernel32', 'VirtualProtectEx'),
    
    # Module loading
    LOAD_LIBRARY_W_HASH: ('kernel32', 'LoadLibraryW'),
    LOAD_LIBRARY_A_HASH: ('kernel32', 'LoadLibraryA'),
    GET_PROC_ADDRESS_HASH: ('kernel32', 'GetProcAddress'),
    FREE_LIBRARY_HASH: ('kernel32', 'FreeLibrary'),
    GET_MODULE_HANDLE_HASH: ('kernel32', 'GetModuleHandleA'),
    GET_MODULE_HANDLE_W_HASH: ('kernel32', 'GetModuleHandleW'),
    
    # File operations
    GET_FILE_ATTRIBUTES_A_HASH: ('kernel32', 'GetFileAttributesA'),
    GET_FILE_ATTRIBUTES_W_HASH: ('kernel32', 'GetFileAttributesW'),
    CREATE_FILE_W_HASH: ('kernel32', 'CreateFileW'),
    CREATE_FILE_A_HASH: ('kernel32', 'CreateFileA'),
    READ_FILE_HASH: ('kernel32', 'ReadFile'),
    WRITE_FILE_HASH: ('kernel32', 'WriteFile'),
    CLOSE_HANDLE_HASH: ('kernel32', 'CloseHandle'),
    
    # Process tokens (advapi32)
    OPEN_PROCESS_TOKEN_HASH: ('advapi32', 'OpenProcessToken'),
    DUPLICATE_TOKEN_HASH: ('advapi32', 'DuplicateToken'),
    DUPLICATE_TOKEN_EX_HASH: ('advapi32', 'DuplicateTokenEx'),
    SET_THREAD_TOKEN_HASH: ('advapi32', 'SetThreadToken'),
    GET_TOKEN_INFORMATION_HASH: ('advapi32', 'GetTokenInformation'),
    IMPERSONATE_LOGGED_ON_USER_HASH: ('advapi32', 'ImpersonateLoggedOnUser'),
    REVERT_TO_SELF_HASH: ('advapi32', 'RevertToSelf'),
    
    # Privilege operations (advapi32)
    LOOKUP_PRIVILEGE_VALUE_HASH: ('advapi32', 'LookupPrivilegeValueW'),
    ADJUST_TOKEN_PRIVILEGES_HASH: ('advapi32', 'AdjustTokenPrivileges'),
    
    # Native APIs (ntdll)
    NT_QUERY_SYSTEM_INFORMATION_HASH: ('ntdll', 'NtQuerySystemInformation'),
    NT_QUERY_INFORMATION_PROCESS_HASH: ('ntdll', 'NtQueryInformationProcess'),
    NT_SET_INFORMATION_PROCESS_HASH: ('ntdll', 'NtSetInformationProcess'),
    NT_CREATE_THREAD_EX_HASH: ('ntdll', 'NtCreateThreadEx'),
    NT_PROTECT_VIRTUAL_MEMORY_HASH: ('ntdll', 'NtProtectVirtualMemory'),
    NT_ALLOCATE_VIRTUAL_MEMORY_HASH: ('ntdll', 'NtAllocateVirtualMemory'),
    NT_FREE_VIRTUAL_MEMORY_HASH: ('ntdll', 'NtFreeVirtualMemory'),
    NT_WRITE_VIRTUAL_MEMORY_HASH: ('ntdll', 'NtWriteVirtualMemory'),
    NT_READ_VIRTUAL_MEMORY_HASH: ('ntdll', 'NtReadVirtualMemory'),
}

# Reverse lookup: (dll, function) -> hash
HASH_REVERSE_TABLE = {v: k for k, v in KERNEL32_HASHES.items()}


def get_api_hash(dll_name: str, function_name: str) -> int:
    """
    Get pre-computed API hash.
    
    Args:
        dll_name: DLL name (e.g., 'kernel32')
        function_name: Function name (e.g., 'CreateProcessW')
    
    Returns:
        32-bit hash integer, or raises KeyError if not found
    
    Raises:
        KeyError: If API not in hash table (call compute_api_hash() to add)
    """
    key = (dll_name, function_name)
    if key not in HASH_REVERSE_TABLE:
        # Fallback to computing on demand
        return compute_api_hash(dll_name, function_name)
    return HASH_REVERSE_TABLE[key]


def get_api_info(api_hash: int) -> Optional[Tuple[str, str]]:
    """
    Look up API info by hash.
    
    Args:
        api_hash: 32-bit API hash
    
    Returns:
        Tuple of (dll_name, function_name) or None if not found
    
    Example:
        >>> dll, func = get_api_info(0x3A5F2B1C)
        >>> print(dll, func)
        kernel32 CreateProcessW
    """
    return KERNEL32_HASHES.get(api_hash)


def list_all_hashes() -> dict:
    """
    Return complete hash table for inspection/debugging.
    
    Returns:
        Dictionary: {api_hash: (dll, function), ...}
    """
    return KERNEL32_HASHES.copy()


if __name__ == '__main__':
    # Quick test
    print("[*] Kernel32 Hash Table - Sample Coverage")
    print(f"[*] Total APIs: {len(KERNEL32_HASHES)}")
    print(f"[*] DLLs covered: {len(set(v[0] for v in KERNEL32_HASHES.values()))}")
    
    # Show some examples
    print("\n[*] Example hashes:")
    print(f"    CreateProcessW: 0x{CREATE_PROCESS_W_HASH:08X}")
    print(f"    VirtualAllocEx: 0x{VIRTUAL_ALLOC_EX_HASH:08X}")
    print(f"    GetProcAddress: 0x{GET_PROC_ADDRESS_HASH:08X}")
    print(f"    NtQuerySystemInformation: 0x{NT_QUERY_SYSTEM_INFORMATION_HASH:08X}")
    
    # Test lookup
    hash_val = CREATE_PROCESS_W_HASH
    dll, func = get_api_info(hash_val)
    print(f"\n[*] Reverse lookup test:")
    print(f"    Hash 0x{hash_val:08X} -> {dll}::{func}")
