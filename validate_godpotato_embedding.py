#!/usr/bin/env python3
"""
Validate that GODPOTATO_BINARY is correctly embedded in privesc.py
Useful for CI/CD verification before running full pipeline.
"""

import sys
import os
import base64
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def validate_embedding():
    """Check that GODPOTATO_BINARY constant is present and decodable."""
    
    logger.info("=" * 70)
    logger.info("GodPotato Embedding Validation")
    logger.info("=" * 70)
    
    # Step 1: Import WindowsPrivesc
    logger.info("\n[1/4] Importing WindowsPrivesc...")
    try:
        from agents.privesc import WindowsPrivesc
        logger.info("[✓] Successfully imported WindowsPrivesc")
    except ImportError as e:
        logger.error(f"[✗] Failed to import: {e}")
        return False
    
    # Step 2: Check GODPOTATO_BINARY constant exists
    logger.info("\n[2/4] Checking GODPOTATO_BINARY constant...")
    if not hasattr(WindowsPrivesc, 'GODPOTATO_BINARY'):
        logger.error("[✗] GODPOTATO_BINARY not found in WindowsPrivesc")
        logger.error("    Make sure build_embed_godpotato.py was run successfully")
        return False
    
    binary_data = getattr(WindowsPrivesc, 'GODPOTATO_BINARY', None)  # type: ignore
    if binary_data is None:
        logger.error("[✗] GODPOTATO_BINARY is None")
        return False
    logger.info(f"[✓] GODPOTATO_BINARY found: {len(binary_data)} bytes ({len(binary_data)/1024:.1f} KB)")
    
    # Step 3: Validate it's actually a PE executable
    logger.info("\n[3/4] Validating PE header...")
    if not binary_data.startswith(b'MZ'):
        logger.error(f"[✗] Invalid PE header (expected 'MZ', got {binary_data[:2]})")
        return False
    
    logger.info(f"[✓] Valid PE header: MZ...")
    
    # Step 4: Check imports
    logger.info("\n[4/4] Checking escalate_via_godpotato method...")
    if not hasattr(WindowsPrivesc, 'escalate_via_godpotato'):
        logger.error("[✗] escalate_via_godpotato method not found")
        return False
    
    logger.info(f"[✓] escalate_via_godpotato method found")
    
    # Success summary
    logger.info("\n" + "=" * 70)
    logger.info("[✓] VALIDATION PASSED")
    logger.info("=" * 70)
    logger.info(f"\nBinary Details:")
    logger.info(f"  Size: {len(binary_data)} bytes ({len(binary_data)/1024:.1f} KB)")
    logger.info(f"  PE Type: {'x64' if len(binary_data) > 50000 else 'x86'} (estimated)")
    logger.info(f"  Ready for: Phase 5 (GodPotato Priority 1)")
    logger.info(f"\nNext: Run full pipeline:")
    logger.info(f"  python test_propagation_pipeline.py")
    logger.info("\nExpect in Phase 5 output:")
    logger.info(f"  [PRIORITY 1] Attempting GodPotato...")
    logger.info(f"  🍠 Attempting GodPotato (SeImpersonate → SYSTEM)...")
    logger.info(f"\nLook for success indicator:")
    logger.info(f"  ✓✓✓ GodPotato SUCCESS → SYSTEM achieved")
    
    return True

def check_import_base64():
    """Verify base64 import is present in privesc.py."""
    privesc_path = 'agents/privesc.py'
    
    logger.info("\n[Extra] Checking for 'import base64'...")
    if not os.path.exists(privesc_path):
        logger.warning(f"    [!] {privesc_path} not found, skipping import check")
        return
    
    with open(privesc_path, 'r') as f:
        content = f.read()
    
    if 'import base64' in content:
        logger.info("[✓] 'import base64' found")
    else:
        logger.warning("[!] 'import base64' NOT found")
        logger.warning("    Add this line near the top of privesc.py:")
        logger.warning("    import base64")

if __name__ == '__main__':
    try:
        success = validate_embedding()
        check_import_base64()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"[✗] Validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
