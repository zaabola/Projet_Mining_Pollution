# PPE Detection Web Application
# Django-based web app for deploying YOLO detection models

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Django Development Server
```bash
cd web_app
python manage.py runserver
```

The app will be available at: `http://localhost:8000`

## Features

### Image Detection
- Upload images and run YOLO inference
- Adjustable confidence threshold
- Annotated output with bounding boxes
- Explainability heatmaps (EigenCAM)

### Video Detection
- Process video files with YOLO detection
- Frame-by-frame analysis
- Detection statistics

### Webcam Stream
- Real-time detection from webcam
- Configurable duration
- Live statistics

### Model Support
- **Helmet Detection** - Detects safety helmets
- **Gas Mask Detection** - Identifies gas masks
- **Face Mask Detection** - Detects face masks
- **Fish Detection** - Recognizes fish species

## API Endpoints

### Image Detection
**POST** `/api/detect/image/`
- `image`: Image file (multipart/form-data)
- `model`: Model name (helmet, gasmask, mask, fish)
- `confidence`: Confidence threshold (0-1)
- `heatmap`: Generate heatmap (true/false)

### Video Detection
**POST** `/api/detect/video/`
- `video`: Video file (multipart/form-data)
- `model`: Model name
- `confidence`: Confidence threshold

### Webcam Detection
**POST** `/api/detect/webcam/`
- `model`: Model name
- `confidence`: Confidence threshold
- `duration`: Recording duration in seconds

### Explainability Heatmap
**POST** `/api/explain/heatmap/`
- `image`: Image file
- `model`: Model name

### Available Models
**GET** `/api/models/`
- Returns list of available and loaded models

## Project Structure

```
web_app/
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
├── ppe_detector/             # Django project settings
│   ├── settings.py          # Configuration
│   ├── urls.py              # URL routing
│   ├── wsgi.py              # WSGI application
│   ├── __init__.py
│   ├── templates/
│   │   └── index.html       # Web UI
│   └── static/
│       └── js/
│           └── app.js       # Frontend JavaScript
├── detector/                # Django app
│   ├── views.py            # Request handlers
│   ├── urls.py             # App URL routing
│   ├── inference.py        # Model inference service
│   ├── migrations/
│   └── __init__.py
└── media/                  # Uploaded files (auto-created)
```

## Model Paths Configuration

Edit `ppe_detector/settings.py` to set your model paths:

```python
MODELS_CONFIG = {
    'helmet': 'path/to/helmet_model.pt',
    'mask': 'path/to/mask_model.pt',
    'gasmask': 'path/to/gasmask_model.pt',
    'fish': 'path/to/fish_model.pt',
}
```

## Deployment

### Local Development
```bash
python manage.py runserver
```

### Production with Gunicorn
```bash
gunicorn ppe_detector.wsgi:application --bind 0.0.0.0:8000
```

### Docker (Optional)
Create a `Dockerfile` for containerized deployment.

## Explainability

The app uses **EigenCAM** for model explainability:
- Gradient-free attention mechanism
- Highlights dominant features in predictions
- Overlaid heatmap visualization
- Helps understand model decision-making

## Performance Tips

1. **GPU Support**: Ensure CUDA is installed for faster inference
2. **Batch Processing**: Process multiple images efficiently
3. **Model Caching**: Models are loaded once and cached
4. **Async Tasks**: Consider using Celery for long-running tasks

## Troubleshooting

### Models not loading
- Check model paths in `settings.py`
- Verify model files exist
- Check CUDA/torch compatibility

### Upload size limits
- Modify `MAX_UPLOAD_SIZE` in `settings.py`
- Default is 50MB

### Slow inference
- Check GPU availability
- Reduce confidence threshold for faster filtering
- Use smaller model variants

## License

This project uses YOLO models from Ultralytics.
