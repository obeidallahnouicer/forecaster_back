@echo off
REM Quick setup script for Windows users
REM This script sets up the forecaster project automatically

echo ========================================
echo Sales Forecaster - Quick Setup (Windows)
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Python detected: 
python --version
echo.

REM Run the setup script
echo [2/5] Running setup script...
python setup.py
if errorlevel 1 (
    echo [ERROR] Setup script failed!
    pause
    exit /b 1
)
echo.

REM Install dependencies
echo [3/5] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install
    echo Please check the error messages above
)
echo.

REM Check if .env exists
echo [4/5] Checking configuration...
if not exist .env (
    echo [WARNING] .env file not found!
    echo Please create .env from .env.example and add your API keys
) else (
    echo [OK] .env file exists
)
echo.

REM Final instructions
echo [5/5] Setup complete!
echo.
echo ========================================
echo Next Steps:
echo ========================================
echo 1. Edit .env file and add your GROQ_API_KEY
echo 2. Run the server:    python main.py
echo 3. Or run dashboard:  streamlit run streamlit_app.py
echo ========================================
echo.

pause
