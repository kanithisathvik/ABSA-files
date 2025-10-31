# ABSA Platform - Local Deployment Guide

## Quick Start

### Option 1: One-Click Deployment
Simply double-click `DEPLOY_LOCAL.bat` to start the server.

### Option 2: Manual Deployment

1. **Activate Virtual Environment:**
   ```cmd
   venv_new\Scripts\activate
   ```

2. **Start the Server:**
   ```cmd
   python -m uvicorn app_final:app --host 127.0.0.1 --port 8000 --reload
   ```

## Access Points

Once the server starts, access the application at:

- **Main Application:** http://127.0.0.1:8000/static/index_advanced.html
- **API Documentation:** http://127.0.0.1:8000/docs
- **API Endpoint:** http://127.0.0.1:8000/analyze

## Server Information

- **Backend Framework:** FastAPI
- **AI Model:** cardiffnlp/twitter-roberta-base-sentiment-latest
- **Port:** 8000
- **Hot Reload:** Enabled (auto-restarts on code changes)

## Features

✅ Single Review Analysis
✅ Batch Processing
✅ Review Comparison
✅ Analysis History & Insights
✅ 15 Product Categories
✅ Dark Theme UI
✅ Real-time Visualizations (4 chart types)
✅ Export (JSON/CSV)

## API Usage

### Analyze Endpoint
```bash
POST http://127.0.0.1:8000/analyze
Content-Type: application/json

{
  "review": "The camera quality is excellent but battery life is poor",
  "category": "smartphones",
  "aspects": ["camera", "battery"]
}
```

### Response
```json
[
  {
    "Aspect": "camera",
    "Sentiment": "Positive",
    "Score (1-10)": 8.5,
    "Confidence": 0.92,
    "Probabilities": {
      "Positive": 0.92,
      "Neutral": 0.05,
      "Negative": 0.03
    },
    "Reasoning": "Positive keywords: excellent"
  }
]
```

## Troubleshooting

### Model Loading Issues
If the model fails to load, it may need to be downloaded:
- First startup takes 2-5 minutes to download the AI model
- Subsequent starts are faster (model is cached)

### Port Already in Use
If port 8000 is busy:
```cmd
python -m uvicorn app_final:app --host 127.0.0.1 --port 8001 --reload
```
Then access at: http://127.0.0.1:8001/static/index_advanced.html

### Dependencies Issues
Reinstall dependencies:
```cmd
venv_new\Scripts\activate
pip install -r requirements.txt --force-reinstall
```

## Stopping the Server

Press `Ctrl + C` in the terminal window to stop the server.

## Production Deployment

For production deployment, consider:
- Using Gunicorn/Uvicorn workers
- Setting up NGINX reverse proxy
- Enabling HTTPS
- Using a process manager (PM2, Supervisor)
- Configuring CORS properly
- Setting up logging and monitoring

## Support

For issues or questions, check:
- API Documentation: http://127.0.0.1:8000/docs
- Server logs in the terminal
- README_ADVANCED.md for detailed information
