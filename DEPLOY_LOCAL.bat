@echo off
color 0A
echo ========================================
echo    ABSA Platform - Local Deployment
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Checking Python environment...
if not exist "venv_new\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run INSTALL_AND_RUN.bat first.
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv_new\Scripts\activate.bat

echo [3/4] Installing/Updating dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo [4/4] Starting ABSA Server...
echo.
echo ========================================
echo    Server Configuration
echo ========================================
echo Backend API: http://127.0.0.1:8000
echo Frontend UI: http://127.0.0.1:8000/static/index_advanced.html
echo API Docs: http://127.0.0.1:8000/docs
echo ========================================
echo.
echo Server is starting... Please wait for model to load.
echo Press Ctrl+C to stop the server.
echo.

python -m uvicorn app_final:app --host 127.0.0.1 --port 8000 --reload

pause
