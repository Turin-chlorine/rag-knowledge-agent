@echo off
REM ============================================
REM One-click start script for Windows
REM Auto: create venv -> install deps -> start server
REM NOTE: Keep this file ASCII-only. Chinese chars
REM break cmd batch parsing (GBK vs UTF-8 issue).
REM ============================================

REM Switch to the script's own directory so relative paths work
cd /d "%~dp0"

REM First run: create virtual environment and install dependencies
if not exist venv (
    echo [1/3] First run. Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Please install Python 3.10+ and add it to PATH.
        pause
        exit /b 1
    )
    echo [2/3] Installing dependencies. This may take a few minutes...
    venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Check your network and retry.
        pause
        exit /b 1
    )
)

REM Create .env from template if missing, then ask user to fill in the API key
if not exist .env (
    copy .env.example .env >nul
    echo.
    echo [NOTICE] A .env file has been created.
    echo Please open .env, fill in your DEEPSEEK_API_KEY, then run this script again.
    pause
    exit /b 0
)

REM Check whether port 8000 is already occupied by an old server
netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo [ERROR] Port 8000 is already in use. An old server may still be running.
    echo Please close the old server window first, then run this script again.
    pause
    exit /b 1
)

echo [3/3] Starting server...
echo After startup, open http://127.0.0.1:8000 in your browser.
echo.
venv\Scripts\python.exe -m uvicorn backend.main:app --reload

REM Keep the window open after the server exits so errors stay visible
echo.
echo Server stopped. If there was an error, see the messages above.
pause
