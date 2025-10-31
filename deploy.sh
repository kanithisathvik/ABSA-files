#!/bin/bash

echo "========================================="
echo "   Deploy to Railway/Render"
echo "========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: ABSA Platform"
fi

echo "Choose deployment platform:"
echo "1. Railway (Recommended)"
echo "2. Render"
echo "3. Heroku"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Deploying to Railway..."
        echo "1. Go to: https://railway.app"
        echo "2. Click 'New Project'"
        echo "3. Select 'Deploy from GitHub repo'"
        echo "4. Choose your repository"
        echo "5. Railway will auto-deploy!"
        ;;
    2)
        echo ""
        echo "Deploying to Render..."
        echo "1. Go to: https://render.com"
        echo "2. Click 'New +' -> 'Web Service'"
        echo "3. Connect your GitHub repository"
        echo "4. Use these settings:"
        echo "   - Build Command: pip install -r requirements.txt"
        echo "   - Start Command: uvicorn app_final:app --host 0.0.0.0 --port \$PORT"
        echo "5. Click 'Create Web Service'"
        ;;
    3)
        echo ""
        echo "Deploying to Heroku..."
        echo "Creating Procfile..."
        echo "web: uvicorn app_final:app --host 0.0.0.0 --port \$PORT" > Procfile
        echo ""
        echo "Run these commands:"
        echo "heroku login"
        echo "heroku create your-app-name"
        echo "git push heroku main"
        ;;
esac

echo ""
echo "========================================="
echo "Deployment guide created!"
echo "========================================="
