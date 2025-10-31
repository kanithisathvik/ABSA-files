@echo off
cls
color 0B
echo ========================================
echo    ABSA Platform - Quick Deploy
echo ========================================
echo.
echo Starting server...
echo.

cd /d "%~dp0"
python -m uvicorn app_final:app --host 0.0.0.0 --port 8000 --reload

pause
