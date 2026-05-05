@echo off
setlocal EnableDelayedExpansion
title UAV Flight Analyzer

cd /d "%~dp0"

echo.
echo  ==============================================
echo   UAV Flight Analyzer
echo  ==============================================
echo.

:: ---------------------------------------------------------------
:: Step 1: Detect Python 3.10-3.13
:: Python 3.14 alpha has a known DLL bug with pydantic_core.
:: ---------------------------------------------------------------
set PYTHON=
set PYTHON_VER=

for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PYTHON (
        py -%%V --version >nul 2>&1
        if not errorlevel 1 (
            set PYTHON=py -%%V
            set PYTHON_VER=%%V
        )
    )
)

if defined PYTHON goto :python_ok

echo  [!] No compatible Python found (3.10-3.13 required).
echo      Python 3.14 is excluded: known pydantic_core DLL bug on Windows.
echo.
echo  [..] Installing Python 3.13 via winget...
echo.
winget install Python.Python.3.13 --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :winget_failed
echo.
echo  [OK] Python 3.13 installed.
echo.
echo  Please CLOSE this window and double-click run.bat again.
echo.
pause
exit /b 0

:winget_failed
echo.
echo  [ERROR] Auto-install failed.
echo.
echo  Install Python 3.13 manually from:
echo    https://www.python.org/downloads/
echo.
echo  Check "Add Python to PATH" during setup, then re-run this file.
echo.
pause
exit /b 1

:: ---------------------------------------------------------------
:: Step 2: Create virtual environment
:: ---------------------------------------------------------------
:python_ok
echo  [OK] Python %PYTHON_VER% detected.

if exist ".venv\Scripts\python.exe" goto :check_deps

echo  [..] Creating virtual environment in .venv ...
%PYTHON% -m venv .venv
if errorlevel 1 goto :venv_failed
echo  [OK] Virtual environment created.
goto :check_deps

:venv_failed
echo.
echo  [ERROR] Could not create virtual environment.
echo          Try reinstalling Python 3.13 from python.org.
echo.
pause
exit /b 1

:: ---------------------------------------------------------------
:: Step 3: Install or refresh dependencies
::  Sentinel: .venv\.installed tracks last successful install.
::  Reinstall when missing OR when requirements.txt is newer.
:: ---------------------------------------------------------------
:check_deps
if not exist ".venv\.installed" goto :install_deps

powershell -NoProfile -Command "exit [int]([System.IO.File]::GetLastWriteTime('requirements.txt') -gt [System.IO.File]::GetLastWriteTime('.venv\.installed'))" >nul 2>&1
if errorlevel 1 goto :install_deps
goto :start_server

:install_deps
echo  [..] Installing dependencies...
echo       First run takes 1-2 minutes. Subsequent starts are instant.
echo.
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt --only-binary=pymavlink
if errorlevel 1 goto :deps_failed
echo installed > ".venv\.installed"
echo.
echo  [OK] Dependencies installed.
goto :start_server

:deps_failed
echo.
echo  [ERROR] Dependency installation failed.
echo.
echo  Fix options:
echo    1. Check your internet connection and try again.
echo    2. Delete the .venv folder, then re-run this file.
echo.
pause
exit /b 1

:: ---------------------------------------------------------------
:: Step 4: Start server and open browser
:: ---------------------------------------------------------------
:start_server
netstat -ano | findstr ":5000 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo  [WARN] Port 5000 is already in use. Server may not start.
    echo         Edit run_web.py to change the port, then re-run.
    echo.
    pause
)
echo.
echo  [..] Starting server at http://localhost:5000
echo       Browser opens automatically. Press Ctrl+C here to stop.
echo.

start "" powershell -WindowStyle Hidden -Command "Start-Sleep 4; Start-Process 'http://localhost:5000'"

.venv\Scripts\python.exe run_web.py

echo.
echo  Server stopped.
pause