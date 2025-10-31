# ABSA Platform - Advanced Sentiment Analysis

A powerful Aspect-Based Sentiment Analysis platform with a modern dark-themed interface.

## 🚀 Live Demo

[Deploy to Vercel](https://vercel.com/import/project?template=https://github.com/YOUR_USERNAME/absa-platform)

## ✨ Features

- 🎯 Advanced ABSA with RoBERTa model
- 📊 4 Interactive visualizations (Bar, Radar, Pie, Heatmap)
- 🌙 Modern dark theme UI
- 📱 Responsive design
- 🔄 Real-time analysis
- 💾 Export to JSON/CSV
- 📈 15 Product categories
- 🎨 Beautiful gradient effects

## 🛠️ Tech Stack

- **Backend:** FastAPI + Uvicorn
- **AI Model:** cardiffnlp/twitter-roberta-base-sentiment-latest
- **Frontend:** Vanilla JavaScript + Plotly.js
- **Styling:** Custom CSS with dark theme

## 📋 Prerequisites

- Python 3.8+
- pip

## 🏃 Quick Start

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/absa-platform.git
   cd absa-platform
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server:**
   ```bash
   python -m uvicorn app_final:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Open in browser:**
   ```
   http://localhost:8000
   ```

## 🌐 Deploy to Vercel

### Option 1: One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/absa-platform)

### Option 2: Manual Deploy

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Deploy:**
   ```bash
   vercel
   ```

3. **Follow the prompts and your site will be live!**

## 📁 Project Structure

```
absa-platform/
├── app_final.py              # Main FastAPI application
├── requirements.txt          # Python dependencies
├── vercel.json              # Vercel configuration
├── static/
│   ├── index_advanced.html  # Main UI
│   ├── styles_advanced.css  # Dark theme styles
│   └── app_advanced.js      # Frontend logic
├── README.md
└── .gitignore
```

## 🎨 Features Overview

### Single Analysis
- Analyze individual reviews
- Auto-detect aspects or specify custom ones
- Real-time sentiment visualization

### Batch Processing
- Upload CSV files
- Process multiple reviews at once
- Bulk export results

### Compare Reviews
- Side-by-side comparison
- Comparative visualizations

### Insights & History
- View analysis history
- Track sentiment trends
- Export historical data

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):
```env
PORT=8000
HOST=0.0.0.0
RELOAD=true
```

### Supported Categories

- Smartphones
- Laptops
- Tablets
- Smartwatches
- Headphones
- Cameras
- Televisions
- Gaming Consoles
- Smart Speakers
- Routers
- Printers
- Monitors
- Keyboards
- Mice
- General

## 📊 API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Analyze Endpoint

```http
POST /analyze
Content-Type: application/json

{
  "review": "The camera is excellent but battery life is poor",
  "category": "smartphones",
  "aspects": ["camera", "battery"]
}
```

## 🎯 Performance

- First load: ~3-5 minutes (model download)
- Subsequent loads: <30 seconds
- Analysis speed: ~1-2 seconds per review
- Supports concurrent requests

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace for the transformer models
- FastAPI team for the amazing framework
- Plotly for visualization library

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/YOUR_USERNAME/absa-platform](https://github.com/YOUR_USERNAME/absa-platform)

## 🐛 Troubleshooting

### Model Loading Issues
- First run downloads ~500MB model
- Ensure stable internet connection
- Check disk space

### Port Already in Use
```bash
# Use different port
python -m uvicorn app_final:app --port 8001
```

### Memory Issues
- Model requires ~2GB RAM
- Close other applications if needed

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Custom model training
- [ ] Authentication & user accounts
- [ ] Advanced analytics dashboard
- [ ] API rate limiting
- [ ] Caching layer
- [ ] WebSocket support for real-time updates

---

Made with ❤️ by Your Name
