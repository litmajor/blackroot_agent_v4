@echo off
REM Mock nmap executable for testing
REM This allows python-nmap to work without the real nmap binary

setlocal enabledelayedexpansion

if "%1"=="--version" (
    echo Nmap version 7.93 ^( https://nmap.org ^)
    echo Platform: win32
    exit /b 0
)

if "%1"=="-V" (
    echo Nmap version 7.93
    exit /b 0
)

REM For actual scans, call Python mock scanner
python "%~dp0mock_nmap_scanner.py" %*
exit /b %ERRORLEVEL%
