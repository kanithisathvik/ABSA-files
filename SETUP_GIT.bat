@echo off
color 0E
cls
echo ========================================
echo    GitHub Deployment Script
echo ========================================
echo.

cd /d "%~dp0"

echo [1/5] Checking Git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo Git found!

echo.
echo [2/5] Initializing Git repository...
if not exist ".git" (
    git init
    echo Git repository initialized!
) else (
    echo Git repository already exists!
)

echo.
echo [3/5] Adding files...
git add .
echo Files added!

echo.
echo [4/5] Creating commit...
set /p commit_msg="Enter commit message (or press Enter for default): "
if "%commit_msg%"=="" set commit_msg=Update: Enhanced ABSA Platform with dark theme
git commit -m "%commit_msg%"

echo.
echo [5/5] Ready to push to GitHub!
echo.
echo ========================================
echo    Next Steps:
echo ========================================
echo 1. Create a new repository on GitHub
echo 2. Copy the repository URL
echo 3. Run: git remote add origin YOUR_REPO_URL
echo 4. Run: git push -u origin main
echo.
echo Or run PUSH_TO_GITHUB.bat after setting up remote
echo ========================================

pause
