"""
Django Configuration Management
Utility to help manage Django settings without code changes
"""

import os
from pathlib import Path

class ConfigManager:
    """Manage Django configuration from environment variables."""
    
    @staticmethod
    def get_debug_mode():
        """Get DEBUG setting from environment."""
        return os.getenv('DEBUG', 'True').lower() == 'true'
    
    @staticmethod
    def get_secret_key():
        """Get SECRET_KEY from environment."""
        return os.getenv('SECRET_KEY', 'django-insecure-change-this-key')
    
    @staticmethod
    def get_allowed_hosts():
        """Get ALLOWED_HOSTS from environment."""
        hosts = os.getenv('ALLOWED_HOSTS', '*')
        return hosts.split(',') if hosts else ['*']
    
    @staticmethod
    def get_model_paths():
        """Get model paths from environment or use defaults."""
        return {
            'helmet': os.getenv('HELMET_MODEL', '../../models/final/helmet_best_yolov8s.pt'),
            'mask': os.getenv('MASK_MODEL', '../../runs/detect/models/finetune/mask_kaggle_finetune/weights/best.pt'),
            'gasmask': os.getenv('GASMASK_MODEL', '../../runs/detect/models/finetune/gasmask_yolov8s_finetuned_SAFE/weights/best.pt'),
            'fish': os.getenv('FISH_MODEL', '../../runs/detect/runs/detect/models/yolo/yolov8_fish_v34/weights/best.pt'),
        }
    
    @staticmethod
    def get_max_upload_size():
        """Get maximum upload size from environment."""
        return int(os.getenv('MAX_UPLOAD_SIZE', '52428800'))  # 50MB default
    
    @staticmethod
    def get_inference_confidence():
        """Get default inference confidence threshold."""
        return float(os.getenv('INFERENCE_CONFIDENCE', '0.5'))
