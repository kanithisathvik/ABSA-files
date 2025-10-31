# Deployment Guide

## 🚀 Deploy to GitHub & Vercel

### Step 1: Initialize Git Repository

```bash
cd "c:\Users\SATHVIK\Downloads\capstone absa complete"

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: ABSA Platform with dark theme"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click "New Repository"
3. Name it: `absa-platform` (or your preferred name)
4. Don't initialize with README (we already have one)
5. Click "Create repository"

### Step 3: Push to GitHub

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/absa-platform.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 4: Deploy to Vercel

#### Option A: Vercel Dashboard (Recommended)

1. Go to [Vercel](https://vercel.com)
2. Sign up/Login with GitHub
3. Click "New Project"
4. Import your `absa-platform` repository
5. Configure:
   - **Framework Preset:** Other
   - **Build Command:** (leave empty)
   - **Output Directory:** (leave empty)
   - **Install Command:** `pip install -r requirements.txt`
6. Add Environment Variables:
   ```
   PYTHON_VERSION=3.9
   ```
7. Click "Deploy"

#### Option B: Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
vercel

# Follow prompts:
# - Set up and deploy? Y
# - Which scope? (select your account)
# - Link to existing project? N
# - Project name? absa-platform
# - Directory? ./
# - Override settings? N

# Deploy to production
vercel --prod
```

## ⚙️ Vercel Configuration

The `vercel.json` is already configured with:
- Python runtime
- Static file serving
- Route handling
- Environment variables

## 🔧 Important Notes for Vercel

### Model Size Limitation
Vercel has a 50MB deployment size limit. The transformers model (~500MB) won't fit.

### Solution: Use Vercel KV or External Storage

1. **Option 1: Use a smaller model**
   ```python
   # In app_final.py, change model to:
   MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
   ```

2. **Option 2: Use external model hosting**
   - Host model on Hugging Face
   - Use API calls instead of local loading

3. **Option 3: Use Vercel Edge Functions** (Advanced)
   - Split into serverless functions
   - Cache model in Vercel KV

### Recommended: Use Railway/Render for Full Deployment

Since Vercel has size limits, consider these alternatives:

#### Deploy to Railway

1. Go to [Railway](https://railway.app)
2. Connect GitHub repository
3. Railway auto-detects Python
4. Environment variables set automatically
5. Deploy with one click

#### Deploy to Render

1. Go to [Render](https://render.com)
2. New Web Service
3. Connect repository
4. Configuration:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app_final:app --host 0.0.0.0 --port $PORT`
5. Deploy

## 📝 Quick Commands Reference

```bash
# Git Commands
git init
git add .
git commit -m "message"
git push origin main

# Vercel Commands
vercel login
vercel          # Deploy preview
vercel --prod   # Deploy production
vercel logs     # View logs
vercel env add  # Add environment variable

# Check deployment
curl https://your-app.vercel.app/status
```

## 🐛 Troubleshooting

### Large Model Size
- Use model quantization
- Implement lazy loading
- Use model API services

### Timeout Issues
- Increase timeout in vercel.json
- Use async processing
- Implement caching

### Memory Limits
- Reduce batch size
- Use streaming responses
- Optimize model loading

## 🎯 Best Practices

1. **Test locally first**
   ```bash
   python -m uvicorn app_final:app --reload
   ```

2. **Check all files are committed**
   ```bash
   git status
   ```

3. **Use environment variables for secrets**
   - Never commit API keys
   - Use Vercel's environment variable UI

4. **Monitor your deployment**
   - Check Vercel logs
   - Set up error tracking
   - Monitor API usage

## 📊 Post-Deployment

### Update README
Replace placeholders:
- `YOUR_USERNAME` with your GitHub username
- Add your live URL
- Update screenshots

### Add Badge
```markdown
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/absa-platform)
```

### Share Your Project
- Tweet about it
- Post on LinkedIn
- Add to your portfolio
- Share on Reddit r/MachineLearning

---

Need help? Check:
- [Vercel Docs](https://vercel.com/docs)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Railway Guide](https://docs.railway.app)
