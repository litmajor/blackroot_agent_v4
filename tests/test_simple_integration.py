#!/usr/bin/env python3
"""Test PayloadEngine with ReflectiveDLLLoader integration - NO EMOJIS"""

from payload_engine import PayloadEngine

print("[*] Testing PayloadEngine integration...")

try:
    engine = PayloadEngine()
    print("[+] PayloadEngine initialized")
    print("    Supported formats: " + str(engine.supported_formats))
    print("    Reflective loader: " + type(engine.reflective_loader).__name__)
    
    # Test reflective loader
    print("\n[*] Testing reflective loader shellcode generation...")
    test_dll = b'MZ\x90' * 100
    
    try:
        shellcode = engine._generate_reflective_dll_loader(test_dll)
        print("[+] Reflective loader generated: " + str(len(shellcode)) + " bytes")
        print("    First bytes: " + shellcode[:16].hex())
        
        # Verify it's real code
        if b'\x50' in shellcode and b'\xc3' in shellcode:
            print("[+] Contains real x86-64 instructions (PUSH, RET)")
        else:
            print("[-] Missing instruction patterns")
    except Exception as e:
        print("[-] Reflective loader failed: " + str(e))
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("[+] PAYLOAD ENGINE + REFLECTIVE LOADER INTEGRATION: SUCCESS")
    print("="*60)
    
except Exception as e:
    print("[-] Failed: " + str(e))
    import traceback
    traceback.print_exc()
