@echo off
echo 🚀 Starting ML Trading API Server...
echo 📊 Loading trained models from user_data/models/
echo 🌐 API will be available at http://localhost:8000
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if models directory exists
if not exist "user_data\models" (
    echo ❌ Models directory not found: user_data/models/
    echo Please make sure your trained models are in the correct location.
    pause
    exit /b 1
)

REM Install requirements if needed
if not exist "requirements.txt" (
    echo ❌ requirements.txt not found
    pause
    exit /b 1
)

echo 📦 Installing Python dependencies...
pip install -r requirements.txt

echo.
echo 🤖 Starting FastAPI server...
echo 📈 ML Trading Engine with XGBoost models
echo 🔗 Webapp should connect to: http://localhost:8000
echo.

REM Start the API server
python trading_api.py

pause
