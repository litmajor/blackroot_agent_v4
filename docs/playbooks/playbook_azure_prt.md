# Playbook: Azure AD Primary Refresh Token (PRT) Extraction & Relay

## 1. Overview
* **Artifact Type**: `azure_prt_cache`
* **Base Value**: 300 ROOT
* **Severity**: Critical
* **Prerequisites**: User context on Azure AD / Entra ID joined Windows workstation (Local admin required for DPAPI-SYSTEM or LSASS unprotect).

The Primary Refresh Token (PRT) is an authentication token issued to devices registered with Microsoft Entra ID. It allows seamless Single Sign-On (SSO) across enterprise applications, Microsoft 365, SharePoint, Exchange Online, and connected enterprise SAML/OIDC apps (NetSuite, Workday, Salesforce).

---

## 2. Artifact Locations & Identification

### Registry Check (Target Identification)
```powershell
# Verify Entra ID Join status
Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\CDJ" -ErrorAction SilentlyContinue | Select-Object TenantId, JoinType
```

### Filesystem Storage
* **CloudAP Plugin Cache**:
  - `%LOCALAPPDATA%\Microsoft\Windows\CloudAP\`
  - `C:\Windows\System32\config\systemprofile\AppData\Local\Microsoft\Windows\CloudAP\`
* **Web Account Manager (WAM) TokenBroker**:
  - `%LOCALAPPDATA%\Microsoft\TokenBroker\Cache\`

---

## 3. Extraction & Unprotect Workflow

### Step 1: WAM Cache Blob Harvest
Blackroot Phase 5 collector locates all `.bin` and MSAL cache files in the TokenBroker directory.

### Step 2: Decryption / Token Acquisition
1. **User-level Token Request (No Admin required via WAM COM API)**:
   Invoke the Windows Web Account Manager API directly to request tokens for target resource IDs (e.g., Azure Portal, Microsoft Graph) on behalf of the signed-in user without re-authenticating:
   ```powershell
   # Concept via ROADtoken / AADInternals
   $tenantId = "..."
   # WAM broker returns access_token + refresh_token
   ```
2. **DPAPI CloudAP Decryption (Admin / SYSTEM context)**:
   The PRT itself is encrypted with a session key protected by the host's DPAPI Master Key.
   - Dump DPAPI master key for the user or SYSTEM.
   - Decrypt CloudAP cached tokens to recover the plaintext PRT and Transport Key (`tb_proof_of_possession`).

---

## 4. Downstream Abuse & SSO Relay

Once the PRT and session keys are obtained:
1. **Bypass Conditional Access / MFA**:
   - The PRT inherently satisfies Device Compliance and Multi-Factor Authentication claims that were satisfied when the user initially logged in.
2. **Exchange PRT for Target Application Tokens**:
   ```bash
   # Using AADInternals or curl with Proof-of-Possession header
   POST https://login.microsoftonline.com/common/oauth2/token
   grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer
   request=<PRT_SIGNED_JWT>
   resource=https://graph.microsoft.com
   ```
3. **Target High-Leverage Cloud Assets**:
   - `https://portal.azure.com` -> View cloud infrastructure, Key Vaults, subscription settings.
   - `https://admin.microsoft.com` -> Enterprise user administration.
   - Enterprise SSO portals (Workday, ADP, NetSuite) configured with Entra ID SSO.
