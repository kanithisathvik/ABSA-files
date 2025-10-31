@echo off
color 0A
cls
echo ========================================
echo    Complete Deployment Wizard
echo ========================================
echo.
echo This wizard will help you deploy your ABSA Platform
echo.
echo Choose deployment method:
echo.
echo 1. GitHub + Vercel (Quick, but size limits)
echo 2. GitHub + Railway (Recommended for ML)
echo 3. GitHub + Render (Free tier available)
echo 4. Local deployment only
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto vercel
if "%choice%"=="2" goto railway
if "%choice%"=="3" goto render
if "%choice%"=="4" goto local

:vercel
echo.
echo ========================================
echo    Vercel Deployment
echo ========================================
echo.
echo Step 1: Install Node.js and Vercel CLI
echo Download Node.js from: https://nodejs.org
echo Then run: npm i -g vercel
echo.
echo Step 2: Setup Git and push to GitHub
call SETUP_GIT.bat
echo.
echo Step 3: Deploy to Vercel
echo Run: vercel login
echo Then: vercel
echo.
pause
exit /b

:railway
echo.
echo ========================================
echo    Railway Deployment (Recommended)
echo ========================================
echo.
echo Step 1: Setup Git and push to GitHub
call SETUP_GIT.bat
echo.
echo Step 2: Deploy to Railway
echo 1. Go to https://railway.app
echo 2. Sign up with GitHub
echo 3. Click "New Project"
echo 4. Select "Deploy from GitHub repo"
echo 5. Choose your ABSA repository
echo 6. Railway will auto-deploy!
echo.
echo Your app will be live at: your-app.railway.app
echo.
pause
exit /b

:render
echo.
echo ========================================
echo    Render Deployment
echo ========================================
echo.
echo Step 1: Setup Git and push to GitHub
call SETUP_GIT.bat
echo.
echo Step 2: Deploy to Render
echo 1. Go to https://render.com
echo 2. Sign up with GitHub
echo 3. Click "New +" -> "Web Service"
echo 4. Connect your repository
echo 5. Configure:
echo    - Name: absa-platform
echo    - Environment: Python 3
echo    - Build: pip install -r requirements.txt
echo    - Start: uvicorn app_final:app --host 0.0.0.0 --port $PORT
echo 6. Click "Create Web Service"
echo.
echo Free tier includes:
echo - 512MB RAM
echo - Automatic SSL
echo - Custom domains
echo.
pause
exit /b

:local
echo.
echo ========================================
echo    Local Deployment
echo ========================================
echo.
echo Starting local server...
call START_SERVER.bat
exit /b

:end
