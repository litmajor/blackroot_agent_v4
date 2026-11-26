# Playbook: Corporate Banking Batches & Treasury File Analysis

## 1. Overview
* **Artifact Type**: `banking_batch_file`
* **Base Value**: 220 ROOT
* **Severity**: Critical
* **Prerequisites**: Read access to user downloads, desktop, corporate shared folders, or accounting export drops.

Corporate finance and payroll personnel regularly generate payment instruction batches in standard banking interchange formats (NACHA ACH in the US, SEPA ISO 20022 in the EU, SWIFT MT940 globally). These files contain complete, unencrypted corporate bank account numbers, employee/vendor bank details, and payment schedules.

---

## 2. Supported Formats & Byte Layouts

### A. NACHA ACH Files (.ach, .nacha)
Fixed 94-character ASCII record lines:
* **Record 1 (File Header)**:
  - `[03:13]`: Immediate Destination (Routing transit number of receiving bank / Federal Reserve)
  - `[13:23]`: Immediate Origin (Company tax ID / Originator ID)
* **Record 5 (Company / Batch Header)**:
  - `[04:20]`: Company Name
  - `[50:53]`: Standard Entry Class (SEC) code (e.g., `PPD` for payroll, `CCD` for corporate credit/debit)
* **Record 6 (Entry Detail)**:
  - `[01:03]`: Transaction Code (`22` Checking Credit, `27` Checking Debit)
  - `[03:11]`: RDFI Routing Transit Number (8 digits) + `[11:12]` Check Digit
  - `[12:29]`: DFI Account Number (recipient bank account)
  - `[29:39]`: Amount in cents (10 digits)
  - `[54:76]`: Individual / Payee Name

### B. SEPA XML / ISO 20022 (.xml, .pain)
XML-based European and international credit transfers:
* `//DbtrAcct/Id/IBAN`: Originating corporate debtor account
* `//CdtTrfTxInf/Amt/InstdAmt`: Settlement amount and currency (`Ccy="EUR"`)
* `//Cdtr/Nm`: Vendor or employee payee name
* `//CdtrAcct/Id/IBAN`: Beneficiary IBAN
* `//CdtrAgt/FinInstnId/BIC`: Beneficiary BIC/SWIFT code

---

## 3. Operational Extraction & Triage

1. **Automated Batch Ingestion**:
   - `ghost_layer_parsers.parse_nacha_ach()` and `parse_sepa_iso20022()` automatically aggregate total batch sums, individual payroll salaries, and routing maps.
2. **Corporate Financial Profiling**:
   - High total amounts (e.g. >$500,000) identify prime corporate operational accounts.
   - Frequent vendor disbursements highlight primary supplier relationships.

---

## 4. Exploitation & Abuse Chains

1. **Vendor Impersonation / Wire Fraud**:
   - Use extracted supplier details, historical payment amounts, and payment cadence to prepare targeted invoice redirection requests.
2. **Direct Debit / ACH Debiting**:
   - Corporate checking accounts lacking ACH Debit Blocks can be directly debited using the captured routing and account numbers.
3. **Payroll Manipulation**:
   - Modify Entry Detail 6 records prior to bank portal upload to redirect salary tranches.
