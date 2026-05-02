"""
Custom middleware and configuration for Django app
"""

import os
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class ModelLoadingMiddleware:
    """Middleware to ensure models are loaded on app startup."""
    
    def __init__(self, get_response):
        self.get_response = get_response
        # Load models on initialization
        from .inference import inference_service
        self.inference_service = inference_service
        logger.info("Models loaded successfully")
    
    def __call__(self, request):
        response = self.get_response(request)
        return response
