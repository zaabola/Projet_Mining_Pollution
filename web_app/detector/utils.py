"""
Utility functions for PPE Detection Web App
"""

import os
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


def validate_image_file(file_obj):
    """
    Validate uploaded image file.
    
    Args:
        file_obj: Django UploadedFile object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    file_ext = file_obj.name.split('.')[-1].lower()
    if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}"
    
    # Check file size
    if file_obj.size > settings.MAX_UPLOAD_SIZE:
        max_size_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        return False, f"File too large. Maximum size: {max_size_mb:.1f} MB"
    
    return True, None


def validate_video_file(file_obj):
    """
    Validate uploaded video file.
    
    Args:
        file_obj: Django UploadedFile object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    file_ext = file_obj.name.split('.')[-1].lower()
    if file_ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)}"
    
    # Check file size
    if file_obj.size > settings.MAX_UPLOAD_SIZE:
        max_size_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        return False, f"File too large. Maximum size: {max_size_mb:.1f} MB"
    
    return True, None


def ensure_media_dir():
    """Ensure media directory exists."""
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)


def format_detection_result(detection_dict):
    """
    Format detection result for JSON response.
    
    Args:
        detection_dict: Detection dictionary from model
        
    Returns:
        Formatted dictionary
    """
    return {
        'class_id': detection_dict.get('class_id'),
        'class_name': detection_dict.get('class_name'),
        'confidence': round(detection_dict.get('confidence', 0), 4),
        'bbox': detection_dict.get('bbox'),
    }


def get_model_status():
    """
    Get current status of all loaded models.
    
    Returns:
        Dictionary with model status
    """
    from .inference import inference_service
    
    status = {
        'loaded': [],
        'failed': [],
        'total': 0,
    }
    
    for model_name, model in inference_service.models.items():
        status['total'] += 1
        if model is not None:
            status['loaded'].append(model_name)
        else:
            status['failed'].append(model_name)
    
    return status


def log_inference_request(model_name, image_path, detections_count):
    """Log inference request for analytics."""
    logger.info(f"Inference: model={model_name}, image={image_path}, detections={detections_count}")
