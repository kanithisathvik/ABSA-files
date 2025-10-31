@echo off
color 0B
cls
echo ========================================
echo    Push to GitHub
echo ========================================
echo.

cd /d "%~dp0"

echo Enter your GitHub repository URL
echo Example: https://github.com/username/absa-platform.git
echo.
set /p repo_url="Repository URL: "

if "%repo_url%"=="" (
    echo ERROR: No URL provided!
    pause
    exit /b 1
)

echo.
echo [1/3] Adding remote...
git remote add origin %repo_url% 2>nul
if errorlevel 1 (
    echo Remote already exists, updating...
    git remote set-url origin %repo_url%
)

echo.
echo [2/3] Creating main branch...
git branch -M main

echo.
echo [3/3] Pushing to GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo ========================================
    echo    ERROR: Push failed!
    echo ========================================
    echo.
    echo Possible reasons:
    echo 1. Authentication failed - Configure Git credentials
    echo 2. Repository doesn't exist - Create it on GitHub first
    echo 3. No commits to push - Make sure files are committed
    echo.
    echo Try: git push -u origin main --force
    echo.
) else (
    echo.
    echo ========================================
    echo    SUCCESS! Code pushed to GitHub
    echo ========================================
    echo.
    echo View your repository at:
    echo %repo_url%
    echo.
    echo Next: Deploy to Vercel!
    echo 1. Go to https://vercel.com
    echo 2. Click "Import Project"
    echo 3. Select your repository
    echo 4. Click "Deploy"
    echo.
)

pause
