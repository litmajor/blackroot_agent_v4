# Relevant Files Inventory - Credential Extraction & Forensics

## Directory Structure Overview

```
e:\repos\litmajor\blackroot_agent_v4\
├── agents/                           # Agent implementations
│   ├── privesc.py                   # ✅ Privilege escalation engine
│   ├── post_escalation.py           # ✅ Post-escalation orchestrator
│   ├── post_escalation_agent.py     # ✅ **LSASS/SAM dumping** (real implementation)
│   ├── post_escalation_agent_v2.py  # ✅ Enhanced version
│   ├── infiltrator.py               # 🟡 Browser detection + system profiling (no extraction)
│   ├── infiltrator_advanced.py      # 🟡 Advanced detection methods (no credential extraction)
│   ├── anti_analysis.py             # ✅ VM/Sandbox/EDR/Debugger detection
│   └── [other agents...]
│
├── black_vault.py                   # ✅ **Artifact encryption & storage**
├── ghost_layer.py                   # 🟡 Shellcode operations + entropy calc
├── payload_engine.py                # 🟡 Payload deployment (stubs for some features)
│
├── recon/                           # Reconnaissance modules
│   ├── scanner.py                   # 🟡 Web reconnaissance (not local)
│   ├── jwt_gql.py                   # 🟡 JWT/GraphQL scanning
│   └── [passive reconnaissance]
│
├── exploit/                         # Exploitation modules
│   ├── veil_payload.js              # 🟡 Browser payload (form capture only, no local extraction)
│   └── [XSS/SQLi scanners]
│
├── mimic/                           # Site cloning
│   ├── site_cloner.py               # 🟡 Puppeteer-based site cloning (not for browser extraction)
│   └── polyglot_builder.py          # 🟡 Payload polyglot builder
│
└── [Documentation files - see below]
```

---

## 🟢 FULLY IMPLEMENTED FILES

### 1. **agents/privesc.py** (Privilege Escalation Engine)
- **Purpose**: Detect and exploit Windows privilege escalation vectors
- **Status**: ✅ Operational (90%)
- **Key Methods**:
  - `WindowsPrivesc.detect_all_vectors()` - Scan for UAC bypass, SeImpersonate, weak services, etc.
  - `WindowsPrivesc._detect_uac_bypass()` - Find UAC bypass opportunities
  - `WindowsPrivesc._find_weak_services()` - Enumerate services with weak permissions
  - `WindowsPrivesc._find_dll_hijack_paths()` - DLL hijacking candidates
  - `WindowsPrivesc._check_seimpersonate()` - Check for token impersonation capability
- **MITRE ATT&CK**: Maps to T1548.002, T1134.002, T1543.003, T1574.002
- **Integration**: Works with GhostLayer for post-injection escalation

---

### 2. **agents/post_escalation_agent.py** (Post-Exploitation - REAL IMPLEMENTATION)
- **Purpose**: Execute system-level exploitation after privilege escalation
- **Status**: ✅ Operational (85% - REAL LSASS/SAM dumping)
- **Key Implemented Methods**:
  - `_dump_lsass()` - **Real LSASS dumping using MiniDumpWriteDump API**
    ```python
    # Line 295+: Real token dup for LSASS access
    # Finds LSASS.exe via tasklist
    # Uses kernel32.OpenProcess with PROCESS_ALL_ACCESS
    # Calls Dbghelp.MiniDumpWriteDump
    # Dump file: lsass_<timestamp>.dmp
    ```
  - `_dump_sam()` - **SAM registry hive extraction**
    ```python
    # Real registry access and dumping
    # HKEY_LOCAL_MACHINE\SAM access
    # Writes to encrypted storage in BlackVault
    ```
  - `_dup_token()` - Token duplication & impersonation
  - `_inject_code()` - Code/shellcode injection
- **Credential Extraction**:
  - ✅ LSASS memory dump (T1003.001)
  - ✅ SAM hash extraction (T1003.002)
  - 🟡 Credential parsing (comment references mimikatz, but not integrated)
- **Storage**: All dumps encrypted via BlackVault
- **Integration**: Direct integration with BlackVault for artifact storage

---

### 3. **agents/post_escalation_agent_v2.py** (Enhanced Version)
- **Purpose**: Newer variant with enhanced LSASS dump verification
- **Status**: ✅ Operational (for testing/enhancement)
- **Differences**:
  - Better error handling for LSASS access
  - Additional registry parsing attempts
  - Version 2 approach to SAM extraction

---

### 4. **black_vault.py** (Secure Artifact Storage)
- **Purpose**: Encrypt and store sensitive artifacts (LSASS dumps, SAM hashes, recon data)
- **Status**: ✅ Operational (95%)
- **Key Features**:
  - AES-GCM encryption (preferred) with fallback AES-CBC+HMAC
  - Per-artifact DEK (Data Encryption Key)
  - Master key derivation via PBKDF2-HMAC-SHA256
  - Key rotation (every 7 days by default)
  - Artifact TTL support (expiration)
- **Methods**:
  - `store(name, data)` - Encrypt & store artifact
  - `retrieve(name)` - Decrypt artifact
  - `list_artifacts()` - List stored items
  - `wipe_all()` - Forensic wipe on detection
- **Integration**: All agents store sensitive data via BlackVault

---

### 5. **anti_analysis.py** (Environmental Detection)
- **Purpose**: Detect analysis/sandboxing environments to evade detection
- **Status**: ✅ Operational (95%)
- **Detection Capabilities**:
  - ✅ Debugger detection (IsDebuggerPresent, `/proc/self/status`)
  - ✅ VM detection (VMware, VirtualBox, Hyper-V, KVM, Xen)
  - ✅ Sandbox detection (Cuckoo, Joe Sandbox markers)
  - ✅ EDR/AV detection (Windows Defender, McAfee, etc.)
  - ✅ Analysis tools detection (Wireshark, Burp, IDA, Ghidra, Frida, PETools)
  - ✅ Privilege level check (Admin/System context)
- **Integration**: Called from main.py PHASE_3 anti-analysis assessment

---

## 🟡 PARTIALLY IMPLEMENTED FILES

### 6. **agents/infiltrator_advanced.py** (System Profiling - Detection Only)
- **Purpose**: Advanced system profiling and capability detection
- **Status**: 🟡 Partial (Detection 90%, Extraction 10%)
- **What's Implemented**:
  - ✅ `_detect_browsers()` - Detects Chrome/Firefox/Edge paths
  - ✅ `_get_browser_paths()` - Maps browser profile directories
  - ✅ `_find_credentials()` - Finds credential file locations (`.ssh`, `.aws`, `.docker`, etc.)
  - ✅ `_enum_certificates()` - Certificate store enumeration
  - ✅ `_get_cached_logons()` - Session enumeration
  - ✅ System profiling (software, services, processes, users, network, etc.)
  - ✅ `_detect_dev_tools()` - Development tool detection
  - ✅ `_detect_debuggers()` - Debugger tool detection
  - ✅ `_detect_virtualization()` - VM detection
  - ✅ `_detect_sandbox()` - Sandbox detection
  - ✅ `_detect_analysis_tools()` - Analysis tool detection
- **What's NOT Implemented**:
  - ❌ Credential extraction (only paths enumerated)
  - ❌ Browser password vault dumping
  - ❌ Browser database parsing (SQLite)
  - ❌ Cookie/history extraction
  - ❌ Credential file parsing (`.aws/credentials`, `.docker/config.json`, etc.)
- **Integration**: Called from infiltrator.py assessment methods

---

### 7. **agents/infiltrator.py** (Browser Detection - No Extraction)
- **Purpose**: Infiltration capability assessment
- **Status**: 🟡 Partial (Detection 95%, Extraction 5%)
- **Implemented**:
  - ✅ `AdvancedInfiltrationMethods._detect_browsers()` - Browser enumeration
  - ✅ `AdvancedInfiltrationMethods._get_browser_paths()` - Profile path mapping
  - ✅ Full system profiling integration
  - 🟡 `enum` command for credential type enumeration (lists but doesn't extract)
- **NOT Implemented**:
  - ❌ Password vault extraction
  - ❌ SQLite database access
  - ❌ Encryption key retrieval
  - ❌ Cookie/history parsing

---

### 8. **agents/post_escalation.py** (Post-Escalation Orchestrator)
- **Purpose**: Orchestrate post-escalation actions
- **Status**: 🟡 Partial (Orchestration 80%, Execution stubbed)
- **Implemented**:
  - ✅ Action enumeration (5 priority phases)
  - ✅ MITRE mapping (T1003.001 for LSASS, T1003.002 for SAM)
  - ✅ Action scheduling
  - 🟡 Action execution (references post_escalation_agent.py for real work)
- **Structure**:
  ```python
  PostEscalationPhase.PHASE_1_INITIAL  # Initial access
  PostEscalationPhase.PHASE_2_CRED_THEFT  # Credential harvesting
  PostEscalationPhase.PHASE_3_PERSISTENCE  # Persistence
  PostEscalationPhase.PHASE_4_EXFIL  # Data exfiltration
  PostEscalationPhase.PHASE_5_STEALTH  # Anti-forensics
  ```

---

### 9. **ghost_layer.py** (Shellcode Operations - Partial)
- **Purpose**: Manage payload operations with encryption
- **Status**: 🟡 Partial (Encryption 90%, Credential extraction 10%)
- **Key Methods**:
  - ✅ `extract_secrets()` - Extract secrets from HTML/JS (web-based, not local)
  - ✅ `fingerprint_tech()` - Technology fingerprinting
  - ✅ `encrypt_shellcode()` - Shellcode encryption via BlackVault
  - ✅ `decrypt_shellcode()` - Shellcode decryption
  - 🟡 `_calculate_entropy()` - Used for shellcode validation, not file encryption detection
  - ✅ `_has_sedebug_privilege()` - Check for SeDebugPrivilege (required for LSASS)
  - ❌ `_extract_secrets()` - Module-level secret extraction (web recon only, not local)

---

## ❌ NOT IMPLEMENTED FILES

### Missing Implementations:

**File**: [encryption_detector.py] ❌ (DOES NOT EXIST)
- Purpose: Should detect file encryption (BitLocker, VeraCrypt, ransomware)
- What's needed:
  - BitLocker detection via WMI Win32_LogicalDisk.Encrypted
  - VeraCrypt container detection
  - Encrypted file signature scanning
  - Ransomware note detection

**File**: [browser_dumper.py] ❌ (DOES NOT EXIST)
- Purpose: Should extract browser passwords/cookies/history
- What's needed:
  - Chrome database parsing (Local State, Login Data, History)
  - Firefox profile.ini parsing
  - Encryption key extraction
  - SQL decryption

**File**: [credential_extractor.py] ❌ (DOES NOT EXIST)
- Purpose: Should parse credential file formats
- What's needed:
  - `.aws/credentials` INI parsing
  - `.docker/config.json` JSON parsing
  - `.ssh/id_rsa` PEM key extraction
  - `.git-credentials` store parsing
  - Environment variable credential extraction

**File**: [lsa_secrets_dumper.py] ❌ (DOES NOT EXIST)
- Purpose: Should extract LSA Secrets registry
- What's needed:
  - HKLM\Security\Policy\Secrets access
  - DPAPI decryption
  - Cached domain credential extraction

---

## 📊 Integration Summary

### Current Data Flow (Credential Extraction):

```
┌─────────────────┐
│ Infiltrator     │
│ (detect env)    │
└────────┬────────┘
         │
    ✅ Detect:
    - Browsers
    - Credentials (paths)
    - System config
         │
         ├──────────────────────────────────┐
         │                                  │
    ✅ Integration                   ❌ Missing
    - Anti-analysis pass             - Extraction
    - Privilege escalation           - Parsing
         │                            - Decryption
         ▼
  ┌──────────────┐
  │ Privesc      │  ✅ Detect escalation vectors
  │              │
  └──────┬───────┘
         │
    ✅ Exploit found vectors
         │
         ▼
  ┌──────────────────────┐
  │ Post-Escalation      │  ✅ LSASS dump
  │                      │  ✅ SAM dump
  │ (real implementation)│  ❌ LSA Secrets
  │                      │  ❌ Vault extraction
  └──────┬───────────────┘
         │
         ▼
  ┌──────────────────────┐
  │ BlackVault           │  ✅ AES-GCM encryption
  │ (encrypted storage)  │  ✅ Key rotation
  │                      │  ✅ TTL support
  └──────────────────────┘
```

---

## 🔗 Key Integration Points

### To Enable Browser Password Extraction:

**Step 1** - Add to `infiltrator_advanced.py`:
```python
@staticmethod
def _dump_chrome_passwords() -> List[Dict]:
    """Extract Chrome passwords from encrypted database"""
    # Need: SQLite parsing + encryption key extraction
    pass

@staticmethod
def _dump_firefox_passwords() -> List[Dict]:
    """Extract Firefox passwords from profile database"""
    # Need: SQLite parsing + master password bypass
    pass
```

**Step 2** - Call from `infiltrator.py`:
```python
def extract_credentials(self):
    # ... existing code ...
    browser_creds = AdvancedInfiltrationMethods._dump_chrome_passwords()
    browser_creds.extend(AdvancedInfiltrationMethods._dump_firefox_passwords())
    # Store in vault
```

**Step 3** - Store via `black_vault.py`:
```python
vault.store("browser_passwords", json.dumps(browser_creds).encode())
```

---

## 📈 Implementation Priority

**Highest ROI / Easiest First**:
1. Environment variable credential harvesting (infiltrator_advanced.py) - Easy
2. AWS/Docker credential file parsing (new file: credential_file_parser.py) - Easy  
3. SSH key extraction (infiltrator_advanced.py) - Easy
4. Browser password vault extraction (new file: browser_dumper.py) - Medium
5. BitLocker/Encryption detection (new file: encryption_detector.py) - Medium
6. Windows Credential Manager vault (infiltrator.py extension) - Medium-Hard
7. LSA Secrets extraction (post_escalation_agent.py extension) - Hard

---

**Document Generated**: April 3, 2026  
**Scope**: e:\repos\litmajor\blackroot_agent_v4  
**Last Updated**: Complete audit of credential extraction, browser forensics, and encryption detection modules
