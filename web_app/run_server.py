#!/usr/bin/env python
"""
Startup script for PPE Detection Django app
Run this to start the development server with automatic model loading
"""

import os
import sys
import django
from django.conf import settings
from django.core.management import execute_from_command_line

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ppe_detector.settings')
    
    # Setup Django
    django.setup()
    
    # Import and test model loading
    print("\n" + "="*50)
    print("PPE DETECTION WEB APPLICATION")
    print("="*50 + "\n")
    
    try:
        from detector.inference import inference_service
        print("[✓] Inference service initialized")
        print(f"[✓] Available models: {list(inference_service.models.keys())}")
        print(f"[✓] Loaded models: {[k for k, v in inference_service.models.items() if v is not None]}")
        failed = [k for k, v in inference_service.models.items() if v is None]
        if failed:
            print(f"[!] Failed to load: {failed}")
    except Exception as e:
        print(f"[!] Error loading models: {e}")
    
    print("\n" + "="*50)
    print("Starting Django development server...")
    print("="*50)
    print("\nAccess the web app at: http://localhost:8000/")
    print("Press CTRL+C to stop\n")
    
    # Start Django server
    execute_from_command_line([sys.argv[0], 'runserver', '0.0.0.0:8000'])
