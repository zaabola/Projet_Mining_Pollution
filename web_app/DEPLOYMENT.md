# Deployment Guide - PPE Detection Web Application

## Overview
This guide covers deployment options for the PPE Detection Django Web App on different platforms.

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Production with Gunicorn](#production-with-gunicorn)
4. [Cloud Deployment](#cloud-deployment)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Local Development

### Quick Start
```bash
cd web_app
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
python run_server.py
```

### Access
- Web UI: http://localhost:8000
- API: http://localhost:8000/api/

### Debug Mode
Debug mode is enabled by default for development. Disable for testing:

Edit `ppe_detector/settings.py`:
```python
DEBUG = False
```

---

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Compose (optional but recommended)

### Build Image
```bash
docker build -t ppe-detector:latest .
```

### Run Container
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/media:/app/media \
  --name ppe-web \
  ppe-detector:latest
```

### Using Docker Compose
```bash
docker-compose up -d
```

### Access
- Web: http://localhost:8000

### Stop Container
```bash
docker stop ppe-web
docker rm ppe-web
```

---

## Production with Gunicorn

### Install Gunicorn
```bash
pip install gunicorn
```

### Start Server
```bash
gunicorn ppe_detector.wsgi:application \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class sync \
  --timeout 120 \
  --log-level info
```

### With Systemd (Linux)
Create `/etc/systemd/system/ppe-detector.service`:

```ini
[Unit]
Description=PPE Detection Django App
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/ppe-detector/web_app
ExecStart=/opt/ppe-detector/web_app/venv/bin/gunicorn \
    ppe_detector.wsgi:application \
    --bind unix:/run/ppe-detector.sock \
    --workers 4

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ppe-detector
sudo systemctl start ppe-detector
```

---

## Cloud Deployment

### Heroku

1. **Create Procfile**
```
web: gunicorn ppe_detector.wsgi:application
worker: python manage.py run_celery_worker
```

2. **Deploy**
```bash
heroku login
heroku create ppe-detector-app
git push heroku main
```

### AWS EC2

1. **SSH into instance**
```bash
ssh -i key.pem ec2-user@your-instance-ip
```

2. **Install dependencies**
```bash
sudo yum install python39 python39-pip
sudo yum install gcc python39-devel
```

3. **Setup app**
```bash
git clone your-repo
cd web_app
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. **Configure Nginx**
```bash
sudo yum install nginx
```

Edit `/etc/nginx/nginx.conf`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /media/ {
        alias /opt/ppe-detector/web_app/media/;
    }

    location /static/ {
        alias /opt/ppe-detector/web_app/staticfiles/;
    }
}
```

5. **Start services**
```bash
sudo systemctl start nginx
gunicorn ppe_detector.wsgi:application --bind 127.0.0.1:8000
```

### Railway.app

```bash
railway login
railway init
railway up
```

Create `railway.toml`:
```toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "python run_server.py"
```

### Google Cloud Run

1. **Create app.yaml**
```yaml
runtime: python310

env: standard

entrypoint: gunicorn ppe_detector.wsgi:application --bind 0.0.0.0:8080

runtime_config:
  cpu: 2
  memory_mb: 2048

env_variables:
  DEBUG: "False"
```

2. **Deploy**
```bash
gcloud run deploy ppe-detector \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2
```

---

## Performance Tuning

### Database Optimization
- Use PostgreSQL for production (instead of SQLite)
- Add database connection pooling

### Caching
Add Redis caching:

```python
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}
```

### Model Optimization
- Use quantized models for faster inference
- Implement model batching
- Use GPU acceleration

### Gunicorn Settings
```bash
gunicorn ppe_detector.wsgi \
  --workers 4 \
  --worker-class sync \
  --worker-connections 1000 \
  --keepalive 5 \
  --timeout 120 \
  --graceful-timeout 30
```

---

## Monitoring & Maintenance

### Logging
Configure logging in `settings.py`:

```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/var/log/ppe-detector/app.log',
        },
    },
    'root': {
        'handlers': ['file'],
        'level': 'INFO',
    },
}
```

### Health Check Endpoint
Add to `detector/urls.py`:

```python
path('health/', views.health_check, name='health-check'),
```

### Metrics
Monitor:
- Request latency
- Model inference time
- Error rates
- GPU usage
- Memory consumption

### Updates & Patches
```bash
pip install --upgrade -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput
systemctl restart ppe-detector
```

---

## Security Checklist

- [ ] Set `DEBUG = False`
- [ ] Generate new `SECRET_KEY`
- [ ] Set `ALLOWED_HOSTS` properly
- [ ] Use HTTPS/SSL
- [ ] Implement CSRF protection
- [ ] Add authentication for API
- [ ] Limit upload file sizes
- [ ] Scan dependencies for vulnerabilities
- [ ] Keep dependencies updated
- [ ] Use environment variables for secrets

### Enable HTTPS
```bash
# Let's Encrypt with Certbot
sudo certbot certonly --standalone -d your-domain.com
```

---

## Troubleshooting

### Out of Memory
- Reduce worker count
- Implement model unloading
- Add swap space

### Slow Inference
- Check GPU availability
- Reduce image resolution
- Batch process requests

### File Upload Issues
- Check disk space
- Verify permissions
- Increase upload limits in Nginx

### Model Loading Fails
- Verify model paths
- Check file permissions
- Ensure sufficient RAM

---

## Additional Resources

- [Django Deployment Docs](https://docs.djangoproject.com/en/4.2/howto/deployment/)
- [Gunicorn Docs](https://docs.gunicorn.org/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [YOLO Deployment](https://docs.ultralytics.com/guides/deploy/)

---

For more help, check application logs and Django documentation.
