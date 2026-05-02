"""
Django settings for PPE Detector web app.
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-your-secret-key-change-this-in-production'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'detector',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'detector.middleware.ModelLoadingMiddleware',
]

ROOT_URLCONF = 'ppe_detector.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'ppe_detector', 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.csrf',
            ],
        },
    },
]

WSGI_APPLICATION = 'ppe_detector.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# Media files (Uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Model paths
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODELS_CONFIG = {
    'helmet': os.path.join(PROJECT_ROOT, 'models/final/helmet_best_yolov8s.pt'),
    'mask': os.path.join(PROJECT_ROOT, 'runs/detect/models/finetune/mask_kaggle_finetune/weights/best.pt'),
    'gasmask': os.path.join(PROJECT_ROOT, 'runs/detect/models/finetune/gasmask_yolov8s_finetuned_SAFE/weights/best.pt'),
    'fish': os.path.join(PROJECT_ROOT, 'runs/detect/runs/detect/models/yolo/yolov8_fish_v34/weights/best.pt'),
    'animaux': os.path.join(PROJECT_ROOT, 'best_animaux_eya.pt'),
    'ahmed': os.path.join(PROJECT_ROOT, 'ahmed_best.pt'),
    'smoke': os.path.join(PROJECT_ROOT, 'best_smoke_ela.pt'),
}

# Inference settings
MAX_UPLOAD_SIZE = 52428800  # 50MB
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
ALLOWED_VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv']
INFERENCE_CONFIDENCE = 0.5

# Auth settings
LOGIN_URL = '/accounts/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/accounts/login/'
