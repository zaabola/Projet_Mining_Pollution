"""
WSGI config for PPE Detector.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ppe_detector.settings')

application = get_wsgi_application()
