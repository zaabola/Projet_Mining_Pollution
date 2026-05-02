# Setup Instructions for PPE Detection Django Web App

## 📋 Project Overview

This is a **complete Django web application** for deploying your trained YOLO object detection models (Helmet, Gas Mask, Face Mask, and Fish detection) with:

✅ Web interface for image/video upload and real-time inference  
✅ Explainability features (EigenCAM heatmaps)  
✅ REST API endpoints  
✅ Support for webcam streaming  
✅ Production-ready configuration  
✅ Docker support  

---

## 🚀 Quick Start (5 minutes)

### Step 1: Navigate to web app folder
```powershell
cd web_app
```

### Step 2: Create and activate virtual environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install all dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the server
```powershell
python run_server.py
```

### Step 5: Open in browser
**http://localhost:8000**

---

## 📁 Project Structure

```
web_app/
│
├── manage.py                    # Django management CLI
├── run_server.py                # Startup script (recommended)
├── requirements.txt             # Python dependencies
├── config_manager.py            # Environment configuration
│
├── ppe_detector/                # Main Django project
│   ├── settings.py             # Settings & model paths
│   ├── urls.py                 # Main URL routing
│   ├── wsgi.py                 # Production WSGI
│   ├── __init__.py
│   ├── templates/
│   │   └── index.html          # Main web interface
│   └── static/
│       └── js/
│           └── app.js          # Frontend logic
│
├── detector/                    # Detection Django app
│   ├── views.py                # Request handlers
│   ├── urls.py                 # App URL routing
│   ├── inference.py            # Model loading & inference
│   ├── middleware.py           # Custom middleware
│   ├── utils.py                # Helper functions
│   ├── apps.py                 # App configuration
│   ├── admin.py                # Django admin
│   ├── models.py               # Database models (optional)
│   ├── tests.py                # Unit tests
│   ├── migrations/
│   └── __init__.py
│
├── media/                       # Uploaded files (auto-created)
├── staticfiles/                 # Collected static files
│
├── .gitignore
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── README.md                    # Full documentation
├── QUICKSTART.md                # Quick start guide
└── DEPLOYMENT.md                # Deployment guide
```

---

## 🎯 Features

### 1. **Image Detection**
- Upload JPG, PNG, BMP, GIF images
- Choose detection model (Helmet, Gas Mask, Mask, Fish)
- Adjust confidence threshold (0-100%)
- View annotated results with bounding boxes
- Generate explainability heatmaps

### 2. **Video Detection**
- Upload MP4, AVI, MOV, MKV videos
- Frame-by-frame object detection
- Detection statistics per video
- Configurable confidence threshold

### 3. **Webcam Stream**
- Real-time detection from webcam
- Configurable duration (5-120 seconds)
- Live detection statistics
- Works with any USB or integrated camera

### 4. **Explainability (EigenCAM)**
- Visualize what the model is "seeing"
- Understand model decision-making
- Gradient-free attention mechanism
- Heatmap overlay on original images

### 5. **REST API**
- `/api/detect/image/` - Image detection
- `/api/detect/video/` - Video detection
- `/api/detect/webcam/` - Webcam detection
- `/api/explain/heatmap/` - Generate heatmaps
- `/api/models/` - Get available models

---

## ⚙️ Configuration

### Model Paths (in `ppe_detector/settings.py`)
```python
MODELS_CONFIG = {
    'helmet': 'path/to/helmet_model.pt',
    'mask': 'path/to/mask_model.pt',
    'gasmask': 'path/to/gasmask_model.pt',
    'fish': 'path/to/fish_model.pt',
}
```

### Upload Limits
```python
MAX_UPLOAD_SIZE = 52428800  # 50MB
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
ALLOWED_VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv']
```

### Inference Settings
```python
INFERENCE_CONFIDENCE = 0.5  # Default confidence threshold (0-1)
```

---

## 🔧 Environment Variables (Optional)

Create a `.env` file in the `web_app` directory:

```env
DEBUG=False
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1,yourdomain.com
MAX_UPLOAD_SIZE=52428800
INFERENCE_CONFIDENCE=0.5

# Model paths
HELMET_MODEL=path/to/helmet.pt
MASK_MODEL=path/to/mask.pt
GASMASK_MODEL=path/to/gasmask.pt
FISH_MODEL=path/to/fish.pt
```

---

## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
docker build -t ppe-detector:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/media:/app/media \
  ppe-detector:latest

# Or use Docker Compose
docker-compose up -d
```

---

## 🌐 Production Deployment

### With Gunicorn + Nginx
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn ppe_detector.wsgi:application \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --timeout 120
```

See `DEPLOYMENT.md` for complete production setup guides.

---

## 📊 API Usage Examples

### Image Detection
```bash
curl -X POST http://localhost:8000/api/detect/image/ \
  -F "image=@photo.jpg" \
  -F "model=helmet" \
  -F "confidence=0.5" \
  -F "heatmap=true"
```

### Get Available Models
```bash
curl http://localhost:8000/api/models/
```

### Generate Heatmap
```bash
curl -X POST http://localhost:8000/api/explain/heatmap/ \
  -F "image=@photo.jpg" \
  -F "model=helmet"
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not loading | Check model paths in `settings.py` |
| Port 8000 in use | `python manage.py runserver 8001` |
| Out of memory | Reduce workers, use smaller models |
| Slow inference | Enable GPU, reduce image size, increase confidence |
| Upload fails | Check file size < 50MB, format is jpg/png/mp4/avi |

---

## 📚 Documentation Files

- **README.md** - Complete feature documentation
- **QUICKSTART.md** - Step-by-step setup guide
- **DEPLOYMENT.md** - Production deployment options
- **This file** - Project overview and setup

---

## 💡 Tips & Best Practices

1. **Development vs Production**
   - Keep `DEBUG=True` only for development
   - Change `SECRET_KEY` before deploying

2. **Model Selection**
   - Use smaller models (yolov8n) for faster inference
   - Use larger models (yolov8m/l) for better accuracy

3. **Performance**
   - Enable GPU for 10-50x faster inference
   - Batch process multiple images when possible
   - Monitor memory usage with large videos

4. **Security**
   - Validate all file uploads
   - Implement rate limiting for APIs
   - Use HTTPS in production
   - Add authentication if needed

---

## 🚀 Next Steps

1. **Verify Setup**
   ```bash
   python run_server.py
   ```

2. **Test Web Interface**
   - Open http://localhost:8000
   - Upload test images
   - Check model dropdown

3. **Test API**
   ```bash
   curl http://localhost:8000/api/models/
   ```

4. **Deploy to Production**
   - Follow DEPLOYMENT.md
   - Choose hosting platform
   - Set up monitoring

---

## 📞 Support

For issues:
1. Check console output for errors
2. Review log files in `/logs/` directory
3. Verify model paths and files exist
4. Check Python/PyTorch version compatibility

---

## 📝 License

This project uses YOLO models from Ultralytics and Django framework.
Ensure compliance with respective licenses.

---

**Happy Detecting! 🎯**

For questions or improvements, check the documentation files included.
