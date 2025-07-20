# BLACKROOT.Agent v5 — Shadow Intelligence Swarm

## Overview
BLACKROOT.Agent is a modular, extensible offensive security toolkit designed for automated reconnaissance, exploitation, site cloning, and C2 (Command & Control) operations. It is intended for penetration testers, red teamers, and security researchers.

---

## Features
- **Reconnaissance**: Scans target for common endpoints, parses robots.txt, fingerprints server, and discovers API references in JavaScript.
- **XSS Scanning**: Automated detection of XSS injection vectors on discovered endpoints.
- **Site Cloning**: Clones target websites for phishing or analysis.
- **Replication**: Deploys payloads or agents to compromised targets.
- **C2 Operations**: Initiates command-and-control sessions for post-exploitation.
- **Dashboard**: Provides a dashboard for monitoring and control.

---

## Installation

1. **Clone the repository:**
   ```powershell
   git clone <repo-url>
   cd blackroot_agent_v4
   ```
2. **Install Python 3.8+ (recommended: 3.10+)**
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
   Or, if requirements.txt is missing, manually install:
   ```powershell
   pip install requests beautifulsoup4
   ```

---

## Usage

### Basic Command
```powershell
python main.py --target <TARGET_URL>
```

### Options
- `--target` (required): The target URL (e.g., `https://victim.com`)
- `--silent`: Run the site cloner in silent mode (no output)

### Example
```powershell
python main.py --target https://example.com
```

---

## User Guide

### 1. Reconnaissance
- The tool scans the target for common endpoints (e.g., `/admin`, `/login`, `/api`).
- It parses `robots.txt` for hidden/disallowed paths.
- JavaScript files are scanned for possible API endpoints.
- Server fingerprinting is performed.

### 2. XSS Scanning
- All discovered endpoints are tested for XSS vulnerabilities.
- If injection vectors are found, the tool proceeds to exploitation.

### 3. Site Cloning
- The target site is cloned for further analysis or phishing operations.
- Use `--silent` to suppress output during cloning.

### 4. Replication
- Payloads or agents are deployed to the target if possible.

### 5. Command & Control (C2)
- A C2 session is initiated for post-exploitation control.

### 6. Dashboard
- A dashboard is started for real-time monitoring and management.

---

## Extending BLACKROOT.Agent
- Add new modules in the appropriate folders (`core/`, `exploit/`, `mimic/`, `recon/`).
- Follow the structure of existing modules for integration.

---

## Disclaimer
**For educational and authorized security testing only.** Unauthorized use against systems you do not own or have explicit permission to test is illegal.

---

## Support
For issues, feature requests, or contributions, please open an issue or pull request on the repository.
