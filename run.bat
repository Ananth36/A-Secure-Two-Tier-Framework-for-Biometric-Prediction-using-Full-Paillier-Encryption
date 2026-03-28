@echo off
echo --- Secure Hospital Portal Launcher ---
echo.

REM --- 1. Check Python Version ---
echo [1/8] Checking global Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python and add it to your PATH.
    pause
    exit /b
)
echo.

REM --- 2. Create Python Virtual Environment ---
echo [2/8] Setting up Python Virtual Environment in '.venv'...
python -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Could not create virtual environment.
    pause
    exit /b
)
echo.

REM --- 3. Install Dependencies (with Verbose logging) ---
echo [3/8] Installing Dependencies (this may take a few minutes)...
echo "Running pip install with verbose logging..."
call .venv\Scripts\python.exe -m pip install -v -r requirements.txt
echo "Pip install finished with error level: %errorlevel%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements. Check your internet connection or requirements.txt.
    pause
    exit /b
)
echo.

REM --- 4. Run Setup Scripts ---
echo [4/8] Running Initial Setup - Generating Keys...
call .venv\Scripts\python.exe encryption_utils.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to generate keys.
    pause
    exit /b
)
echo.

echo [5/8] Running Initial Setup - Encrypting Datasets...
call .venv\Scripts\python.exe encrypt_datasets.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to encrypt datasets.
    pause
    exit /b
)
echo.

echo [6/8] Running Initial Setup - Training Models on Encrypted Data...
call .venv\Scripts\python.exe train_models.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to train models.
    pause
    exit /b
)
echo.

REM --- 5. Start Servers ---
echo [7/8] Starting Main Server (Backend) on port 5001...
start "MainServer (Backend)" .venv\Scripts\python.exe main_server.py

echo.
echo [8/8] Starting App Server (Frontend) on port 5000...
start "AppServer (Frontend)" .venv\Scripts\python.exe app.py

echo.
echo --- LAUNCH COMPLETE ---
echo Your servers are starting.
echo Please wait about 10 seconds for them to initialize.
echo.
echo **Then, open this URL in your browser:**
echo **http://127.0.0.1:5000**
echo.
pause

