# Quick Start Guide for PPE Detection Web App

## Prerequisites
- Python 3.8+
- pip package manager
- CUDA 11.8+ (optional but recommended for GPU acceleration)

## Installation & Setup

### Step 1: Navigate to the web app directory
```powershell
cd web_app
```

### Step 2: Create virtual environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install dependencies
```powershell
pip install -r requirements.txt
```

**Note:** This may take a few minutes as it installs PyTorch and other heavy dependencies.

### Step 4: Configure model paths
Edit `ppe_detector/settings.py` and verify the model paths are correct:

```python
MODELS_CONFIG = {
    'helmet': '../../models/final/helmet_best_yolov8s.pt',
    'mask': '../../runs/detect/models/finetune/mask_kaggle_finetune/weights/best.pt',
    'gasmask': '../../runs/detect/models/finetune/gasmask_yolov8s_finetuned_SAFE/weights/best.pt',
    'fish': '../../runs/detect/runs/detect/models/yolo/yolov8_fish_v34/weights/best.pt',
}
```

### Step 5: Start the server
```powershell
python run_server.py
```

Or use the standard Django command:
```powershell
python manage.py runserver
```

### Step 6: Open in browser
Visit: **http://localhost:8000**

## Usage

### Web Interface
The web app has three main tabs:

1. **Image Detection**
   - Upload an image
   - Select detection model (Helmet/Gas Mask/Face Mask/Fish)
   - Adjust confidence threshold
   - Optionally generate explainability heatmap
   - View annotated results

2. **Video Detection**
   - Upload a video file
   - Process with selected model
   - Get frame-by-frame statistics

3. **Webcam Stream**
   - Start live webcam detection
   - Set recording duration
   - Get real-time statistics

### API Usage

#### Detect Objects in Image
```bash
curl -X POST http://localhost:8000/api/detect/image/ \
  -F "image=@test.jpg" \
  -F "model=helmet" \
  -F "confidence=0.5" \
  -F "heatmap=true"
```

#### Get Available Models
```bash
curl http://localhost:8000/api/models/
```

#### Generate Heatmap
```bash
curl -X POST http://localhost:8000/api/explain/heatmap/ \
  -F "image=@test.jpg" \
  -F "model=helmet"
```

## Configuration

### Settings File: `ppe_detector/settings.py`

Key settings to customize:

```python
# Maximum upload size (50MB)
MAX_UPLOAD_SIZE = 52428800

# Inference confidence threshold (0-1)
INFERENCE_CONFIDENCE = 0.5

# Allowed file types
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
ALLOWED_VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv']
```

## Troubleshooting

### Models not found
1. Check model paths in `ppe_detector/settings.py`
2. Ensure model files exist at specified paths
3. Check the console output when server starts for loading status

### Upload errors
- Verify file size is under 50MB
- Use supported formats (jpg, png, mp4, avi)

### Slow inference
- Ensure CUDA is properly installed for GPU support
- Try reducing confidence threshold
- Use smaller model variants if available

### Port 8000 already in use
```powershell
python manage.py runserver 8001
```

## Project Files

```
web_app/
├── manage.py                    # Django CLI
├── run_server.py                # Startup script
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
├── ppe_detector/                # Main project
│   ├── settings.py             # Configuration
│   ├── urls.py                 # URL routing
│   ├── wsgi.py                 # Production WSGI
│   └── templates/index.html    # Web UI
├── detector/                    # Detection app
│   ├── views.py                # Request handlers
│   ├── urls.py                 # App routes
│   ├── inference.py            # Model inference
│   ├── middleware.py           # Custom middleware
│   └── migrations/             # Database
└── media/                       # Uploads (auto-created)
```

## Next Steps

1. **Test with sample images** - Try uploading test images to verify detection
2. **Adjust confidence** - Fine-tune confidence threshold for your use case
3. **Use explainability** - Enable heatmaps to understand model predictions
4. **Deploy to production** - Use Gunicorn + Nginx for production deployment

## Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn ppe_detector.wsgi:application --bind 0.0.0.0:8000 --workers 4
```

### Docker
```bash
docker build -t ppe-detector .
docker run -p 8000:8000 ppe-detector
```

### Environment Variables (Production)
```bash
export DEBUG=False
export SECRET_KEY='your-secure-key-here'
export ALLOWED_HOSTS='yourdomain.com'
```

## Performance Optimization

1. **GPU Support** - Install CUDA and cuDNN for faster inference
2. **Batch Processing** - Process multiple items efficiently
3. **Model Caching** - Models are cached after first load
4. **Async Tasks** - Consider Celery for background jobs

## Support

For issues or questions:
1. Check the logs in console output
2. Review settings.py configuration
3. Verify model paths and files
4. Check Django documentation for common issues

Happy detecting! 🚀
