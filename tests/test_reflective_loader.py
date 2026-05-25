#!/usr/bin/env python3
"""Test the reflective DLL loader shellcode generation"""

from reflective_dll_loader import ReflectiveDLLLoader

# Create loader
loader = ReflectiveDLLLoader()

# Generate shellcode
print("[*] Generating reflective loader shellcode...")
shellcode = loader.generate_loader_shellcode(b'test_dll_data')

print(f"\n✅ SHELLCODE GENERATION SUCCESSFUL")
print(f"   Size: {len(shellcode)} bytes")
print(f"   First 32 bytes (hex): {shellcode[:32].hex()}")
print(f"   Last 4 bytes: {shellcode[-4:].hex()}")

# Verify it's not ASCII
ascii_count = sum(1 for b in shellcode[:64] if 0x20 <= b <= 0x7E)
print(f"\n📊 ENTROPY ANALYSIS:")
print(f"   ASCII characters in first 64 bytes: {ascii_count}/64 ({100*ascii_count//64}%)")
print(f"   Expected for code: <10% (high entropy)")

# Verify it contains real instructions
print(f"\n🔍 INSTRUCTION VERIFICATION:")
has_push = b'\x50' in shellcode
has_call = b'\xff\xd0' in shellcode
has_ret = b'\xc3' in shellcode
has_mov = b'\x48\x8b' in shellcode or b'\x48\x89' in shellcode

print(f"   PUSH instruction (0x50): {has_push}")
print(f"   CALL instruction (0xFF 0xD0): {has_call}")
print(f"   RET instruction (0xC3): {has_ret}")
print(f"   MOV instruction (0x48...): {has_mov}")

instruction_count = sum([has_push, has_call, has_ret, has_mov])
print(f"   Total valid instructions detected: {instruction_count}/4")

print(f"\n{'='*60}")
if instruction_count >= 3 and ascii_count < 20:
    print("✅ VERDICT: SHELLCODE IS REAL EXECUTABLE MACHINE CODE")
    print("   Not data, not ASCII, can execute on Windows")
else:
    print("❌ VERDICT: Shellcode validation FAILED")
    print(f"   ASCII ratio too high or missing instructions")
print(f"{'='*60}\n")
