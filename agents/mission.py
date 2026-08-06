import time, logging, json, uuid, random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, List
from agents.base import BaseAgent
# --- Use core agent anatomy types ---
from agent_core_anatomy import AgentID, AgentStatus, Mission, Event

# --- Phase 2.4 Mission Implementation Modules ---
try:
    from recon.mission_executor import get_mission_executor, MissionExecutor, PatchInfo, ScanResult
    MISSION_EXECUTOR_AVAILABLE = True
except ImportError:
    MISSION_EXECUTOR_AVAILABLE = False

try:
    from core.file_monitor import get_file_monitor, FileMonitor, MonitorResult
    FILE_MONITOR_AVAILABLE = True
except ImportError:
    FILE_MONITOR_AVAILABLE = False

try:
    from core.patch_manager import get_patch_manager, PatchManager, PatchInfo as PatchInfo2, PatchResult
    PATCH_MANAGER_AVAILABLE = True
except ImportError:
    PATCH_MANAGER_AVAILABLE = False

# --- Phase 2.5: Mission Approval Enforcement ---
try:
    from agents.enforcer import EnforcerAgent
    ENFORCER_AVAILABLE = True
except ImportError:
    ENFORCER_AVAILABLE = False

log = logging.getLogger("MIS-EXECUTOR")


class MissionAgent(BaseAgent):
    def should_activate(self, config):
        return self.codename in config.get("agent_whitelist", [])
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__("MIS-EXECUTOR")
        cfg = cfg or {}
        self.workers = max(1, cfg.get("workers", 2))
        self.retry_limit = cfg.get("retry_limit", 3)
        self.backoff_base = cfg.get("backoff_base", 1.0)  # seconds
        self.log_every = cfg.get("log_every", 60)        # seconds
        self.last_log = datetime.utcnow()
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=self.workers)
        
        # --- Phase 2.4: Initialize real mission executors ---
        self.mission_executor = get_mission_executor() if MISSION_EXECUTOR_AVAILABLE else None
        self.file_monitor = None
        if FILE_MONITOR_AVAILABLE:
            # Skip baseline during init (too slow for tests), start async if needed
            self.file_monitor = get_file_monitor(skip_baseline=True, baseline_timeout=5)
            # Optional: Start baseline in background
            # self.file_monitor.start_baseline_async()
        
        self.patch_manager = get_patch_manager() if PATCH_MANAGER_AVAILABLE else None
        
        log.info("Mission executors initialized (MissionExecutor=%s, FileMonitor=%s, PatchManager=%s)",
                 MISSION_EXECUTOR_AVAILABLE, FILE_MONITOR_AVAILABLE, PATCH_MANAGER_AVAILABLE)

    # ------------------------------------------------------------------
    def run(self):
        super().run()
        log.info("Mission processor started (workers=%s)", self.workers)
        while self.running:
            if getattr(self, "priority", None) == "low":
                log.debug("Low priority — sleeping")
                time.sleep(3)
                continue

            buffer = getattr(getattr(self, "kernel", None), "mission_buffer", None)
            if buffer is None or not hasattr(buffer, "tasks"):
                time.sleep(1)
                continue

            # Pop missions and submit to thread-pool
            while buffer.tasks:
                m = buffer.tasks.pop(0)
                # Accept both dict and Mission
                mission = m if isinstance(m, Mission) else Mission(
                    name=m.get("task", "unknown"),
                    objectives=m.get("objectives", []),
                    parameters=m,
                    status=m.get("status", "pending")
                )
                self.executor.submit(self._execute_with_retry, mission)

            time.sleep(0.2)  # small sleep to reduce CPU spin

    # ------------------------------------------------------------------
    def _execute_with_retry(self, mission: Mission):
        log.debug("Executing mission: %s", mission.name)
        
        # Phase 2.5: Check mission approval before execution
        if ENFORCER_AVAILABLE:
            enforcer = self._get_enforcer()
            if enforcer:
                approval = self._check_mission_approval(enforcer, mission)
                if not approval.get("approved", False):
                    log.warning(f"Mission {mission.name} DENIED: {approval.get('reason', 'Unknown')}")
                    event_denied = Event(
                        event_type="mission_denied",
                        payload={
                            "mission_id": mission.mission_id,
                            "name": mission.name,
                            "reason": approval.get("reason", "Constitutional violation"),
                        }
                    )
                    redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
                    if redis:
                        redis.publish("mission_denied", json.dumps(event_denied.payload))
                    return
        
        redis = getattr(getattr(self.kernel, "swarm", None), "redis", None)
        # Emit Event for mission start
        event_start = Event(event_type="mission_start", payload={"mission_id": mission.mission_id, "name": mission.name, "ts": datetime.utcnow().isoformat()})
        if redis:
            redis.publish("mission_start", json.dumps(event_start.payload))
        for attempt in range(1, self.retry_limit + 1):
            try:
                self._dispatch(mission.name, mission)
                # Emit Event for mission done
                event_done = Event(event_type="mission_done", payload={"mission_id": mission.mission_id, "name": mission.name, "status": "done"})
                if redis:
                    redis.publish("mission_done", json.dumps(event_done.payload))
                return
            except Exception as exc:
                log.warning("Mission %s failed (attempt %s/%s): %s", mission.name, attempt, self.retry_limit, exc)
                if attempt < self.retry_limit:
                    time.sleep(self.backoff_base * (2 ** (attempt - 1)) * random.uniform(0.5, 1.5))
                else:
                    log.error("Mission %s permanently failed.", mission.name)
                    event_err = Event(event_type="mission_error", payload={"mission_id": mission.mission_id, "name": mission.name, "error": str(exc)})
                    if redis:
                        redis.publish("mission_error", json.dumps(event_err.payload))

    # ------------------------------------------------------------------
    def _dispatch(self, task: str, mission: Mission):
        handler = {
            "deep_scan": self._deep_scan,
            "monitor_changes": self._monitor_changes,
            "apply_patch": self._apply_patch,
        }.get(task, self._unknown_task)
        handler(mission)

    # ------------------------------------------------------------------
    def _deep_scan(self, mission: Mission):
        """Real vulnerability scanning with nmap integration."""
        log.info("Deep scan mission %s: Executing real vulnerability scan", mission.mission_id)
        
        # Extract target from mission parameters
        target_ip = mission.parameters.get("target") or mission.parameters.get("target_ip")
        target_port = mission.parameters.get("port") or mission.parameters.get("target_port")
        aggressive = mission.parameters.get("aggressive", False)
        timeout = mission.parameters.get("timeout", 30)
        
        if not target_ip:
            log.error("Deep scan failed: No target specified")
            return {"status": "failed", "error": "No target IP provided"}
        
        try:
            if MISSION_EXECUTOR_AVAILABLE and self.mission_executor:
                # Real nmap scan
                log.debug(f"Scanning target {target_ip} with aggressive={aggressive}")
                result = self.mission_executor.scan_target(
                    target_ip=target_ip,
                    target_port=target_port,
                    aggressive=aggressive,
                    timeout=timeout
                )
                
                scan_data = result.to_dict()
                log.info(f"Scan result: {len(result.vulnerabilities)} vulnerabilities, "
                        f"{result.critical_count} critical, {result.exploit_count} exploitable")
                
                # Store scan result in mission data
                mission.data["scan_result"] = scan_data
                mission.data["status"] = "completed"
                
                # Deduct economic cost
                self._deduct_scan_cost(len(result.vulnerabilities), result.critical_count)
                
                return scan_data
            else:
                return self._unavailable_result(mission, "MissionExecutor")
        
        except Exception as e:
            log.error(f"Deep scan failed: {e}")
            mission.data["status"] = "failed"
            mission.data["error"] = str(e)
            return {"status": "failed", "error": str(e)}
    
    def _deduct_scan_cost(self, vuln_count: int, critical_count: int) -> None:
        """Deduct economic cost for deep scan."""
        # Base cost: 50 units
        # Additional: 10 units per vulnerability, 50 units per critical
        cost = 50 + (vuln_count * 10) + (critical_count * 50)
        vault = getattr(getattr(self.kernel, "economic", None), "vault", None)
        if vault:
            try:
                vault.deduct_balance(self.codename, cost, "deep_scan_cost")
                log.debug(f"Deducted {cost} units for deep scan")
            except Exception as e:
                log.warning(f"Failed to deduct cost: {e}")

    def _monitor_changes(self, mission: Mission):
        """Real file/process monitoring with change detection."""
        log.info("Monitor changes mission %s: Executing real monitoring", mission.mission_id)
        
        monitor_type = mission.parameters.get("type", "full")  # full, files, processes, connections
        
        try:
            if FILE_MONITOR_AVAILABLE and self.file_monitor:
                # Real monitoring
                log.debug(f"Running {monitor_type} monitoring scan")
                result = self.file_monitor.monitor()
                
                monitor_data = result.to_dict()
                
                # Filter based on requested type
                if monitor_type == "files":
                    monitor_data = {"file_changes": monitor_data["file_changes"]}
                elif monitor_type == "processes":
                    monitor_data = {
                        "new_processes": monitor_data["new_processes"],
                        "suspicious_processes": monitor_data["suspicious_processes"],
                    }
                elif monitor_type == "connections":
                    monitor_data = {"new_connections": monitor_data["new_connections"]}
                
                log.info(f"Monitor result: {result.total_changes} changes detected, "
                        f"{len(result.suspicious_processes)} suspicious processes")
                
                # Store monitoring result
                mission.data["monitor_result"] = monitor_data
                mission.data["status"] = "completed"
                mission.data["changes_detected"] = result.changes_detected
                
                # Deduct economic cost
                self._deduct_monitor_cost(result.total_changes)
                
                return monitor_data
            else:
                return self._unavailable_result(mission, "FileMonitor")
        
        except Exception as e:
            log.error(f"Monitor mission failed: {e}")
            mission.data["status"] = "failed"
            mission.data["error"] = str(e)
            return {"status": "failed", "error": str(e)}
    
    def _deduct_monitor_cost(self, change_count: int) -> None:
        """Deduct economic cost for monitoring."""
        # Base cost: 20 units
        # Additional: 5 units per change detected
        cost = 20 + (change_count * 5)
        vault = getattr(getattr(self.kernel, "economic", None), "vault", None)
        if vault:
            try:
                vault.deduct_balance(self.codename, cost, "monitor_changes_cost")
                log.debug(f"Deducted {cost} units for monitoring")
            except Exception as e:
                log.warning(f"Failed to deduct cost: {e}")

    def _apply_patch(self, mission: Mission):
        """Real safe patching with backup/rollback capability."""
        log.info("Apply patch mission %s: Executing real patching", mission.mission_id)
        
        patch_id = mission.parameters.get("patch_id")
        service = mission.parameters.get("service")
        target_version = mission.parameters.get("target_version")
        new_version = mission.parameters.get("new_version")
        cves_fixed = mission.parameters.get("cves_fixed", [])
        dry_run = mission.parameters.get("dry_run", True)  # Default to dry run for safety
        
        if not all([patch_id, service, target_version, new_version]):
            log.error("Patch failed: Missing required parameters")
            return {"status": "failed", "error": "Missing required patch parameters"}
        
        try:
            if PATCH_MANAGER_AVAILABLE and self.patch_manager:
                # Create patch info
                patch = PatchInfo2(
                    patch_id=patch_id,
                    name=f"{service}-{new_version}",
                    description=f"Security patch for {service}",
                    target_service=service,
                    target_version=target_version,
                    new_version=new_version,
                    cves_fixed=cves_fixed,
                    changes={},
                    requires_restart=True,
                )
                
                # Apply patch
                log.debug(f"Applying patch {patch_id} to {service} (dry_run={dry_run})")
                result = self.patch_manager.apply_patch(
                    patch=patch,
                    dry_run=dry_run,
                    backup_only=False
                )
                
                patch_data = result.to_dict()
                
                log.info(f"Patch result: {result.status}, CVEs fixed: {len(result.applied_changes)}")
                
                # Store patch result
                mission.data["patch_result"] = patch_data
                mission.data["status"] = result.status
                mission.data["rollback_available"] = result.rollback_available
                
                # Deduct economic cost
                self._deduct_patch_cost(len(cves_fixed), len(result.errors))
                
                return patch_data
            else:
                return self._unavailable_result(mission, "PatchManager")
        
        except Exception as e:
            log.error(f"Patch mission failed: {e}")
            mission.data["status"] = "failed"
            mission.data["error"] = str(e)
            return {"status": "failed", "error": str(e)}
    
    def _unavailable_result(self, mission: Mission, component: str) -> Dict:
        error = f"{component} is unavailable"
        log.error("Mission %s failed: %s", mission.mission_id, error)
        mission.data["status"] = "failed"
        mission.data["error"] = error
        return {"status": "failed", "error": error}
    
    def _deduct_patch_cost(self, cves_fixed: int, error_count: int) -> None:
        """Deduct economic cost for patching."""
        # Base cost: 100 units
        # Reward: 50 units per CVE fixed
        # Penalty: 20 units per error
        cost = 100 + (error_count * 20) - (cves_fixed * 50)
        vault = getattr(getattr(self.kernel, "economic", None), "vault", None)
        if vault:
            try:
                vault.deduct_balance(self.codename, max(0, cost), "patch_cost")
                log.debug(f"Applied {cves_fixed} patches (cost: {cost} units)")
            except Exception as e:
                log.warning(f"Failed to deduct cost: {e}")

    def _unknown_task(self, mission: Mission):
        log.error("Unknown task: %s", mission.name)
        mission.data["status"] = "failed"
        mission.data["error"] = f"Unknown task type: {mission.name}"

    # ------------------------------------------------------------------
    def shutdown(self):
        """Clean shutdown."""
        self.running = False
        self.executor.shutdown(wait=True)
        
        # Clean up Phase 2.4 executors
        if self.file_monitor:
            log.debug("File monitor stopped")
        if self.patch_manager:
            log.debug("Patch manager stopped")
        
        log.info("Mission agent shut down.")
    
    # --- Phase 2.5: Mission Approval Helper Methods ---
    def _get_enforcer(self) -> Optional[Any]:
        """Get the EnforcerAgent from the kernel."""
        try:
            agents = getattr(self.kernel, "agents", {})
            if isinstance(agents, dict):
                enforcer = agents.get("ENF-ENFORCER") or agents.get("ENF")
            else:
                # If agents is a list or other container
                enforcer = None
                for agent in getattr(agents, "__iter__", []):
                    if hasattr(agent, "codename") and agent.codename == "ENF-ENFORCER":
                        enforcer = agent
                        break
            return enforcer
        except Exception as e:
            log.debug(f"Could not get enforcer: {e}")
            return None
    
    def _check_mission_approval(self, enforcer: Any, mission: Mission) -> Dict[str, Any]:
        """
        Check if mission is approved by the enforcer.
        
        Returns:
            Dict with "approved" boolean and optional "reason" string
        """
        try:
            # Extract mission parameters
            mission_id = mission.mission_id
            mission_name = mission.name
            agent_id = self.codename
            params = mission.parameters or {}
            
            # Call enforcer's evaluation (synchronous wrapper)
            if hasattr(enforcer, "mission_approval_engine") and enforcer.mission_approval_engine:
                approval_decision = enforcer.mission_approval_engine.evaluate_mission(
                    mission_id=mission_id,
                    mission_name=mission_name,
                    agent_id=agent_id,
                    mission_parameters=params,
                )
                
                # Check approval status
                from laws.mission_approval import ApprovalStatus
                
                is_approved = approval_decision.status in [
                    ApprovalStatus.APPROVED,
                ]  # Exclude CONDITIONAL_APPROVAL for now
                
                reason = approval_decision.justification if not is_approved else None
                
                return {
                    "approved": is_approved,
                    "reason": reason,
                    "status": approval_decision.status.value,
                }
            
            # Fallback: approve if enforcer not fully configured
            return {"approved": True, "reason": "Approval engine not available"}
        
        except Exception as e:
            log.warning(f"Mission approval check failed: {e}, allowing mission")
            return {"approved": True, "reason": f"Approval check error (fallback): {str(e)}"}
    
    # --- Phase 2.4 Utility Methods ---
    def get_mission_status(self) -> Dict:
        """Get current mission executor status."""
        return {
            "mission_executor": "available" if MISSION_EXECUTOR_AVAILABLE else "unavailable",
            "file_monitor": "available" if FILE_MONITOR_AVAILABLE else "unavailable",
            "patch_manager": "available" if PATCH_MANAGER_AVAILABLE else "unavailable",
            "workers": self.workers,
            "retry_limit": self.retry_limit,
        }
    
    def get_patch_history(self) -> List[Dict]:
        """Get history of applied patches."""
        if self.patch_manager:
            return self.patch_manager.get_patch_history()
        return []