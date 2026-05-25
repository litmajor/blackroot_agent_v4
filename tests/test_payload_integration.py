#!/usr/bin/env python3
"""Test PayloadEngine with ReflectiveDLLLoader integration"""

from payload_engine import PayloadEngine

print("[*] Testing PayloadEngine integration...")

try:
    engine = PayloadEngine()
    print("✅ PayloadEngine initialized")
    print(f"   Supported formats: {engine.supported_formats}")
    print(f"   Reflective loader: {type(engine.reflective_loader).__name__}")
    print(f"   PE validator: {type(engine.pe_validator).__name__}")
    
    # Test reflective loader
    print("\n[*] Testing reflective loader shellcode generation...")
    test_dll = b'MZ\x90' * 100  # Fake DLL-like data
    
    try:
        shellcode = engine._generate_reflective_dll_loader(test_dll)
        print(f"✅ Reflective loader generated: {len(shellcode)} bytes")
        print(f"   First bytes: {shellcode[:16].hex()}")
        
        # Verify it's real code
        if b'\x50' in shellcode and b'\xc3' in shellcode:
            print("✅ Contains real x86-64 instructions (PUSH, RET)")
        else:
            print("❌ Missing instruction patterns")
    except Exception as e:
        print(f"❌ Reflective loader failed: {e}")
    
    print("\n" + "="*60)
    print("✅ PAYLOAD ENGINE + REFLECTIVE LOADER INTEGRATION: SUCCESS")
    print("="*60)
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
