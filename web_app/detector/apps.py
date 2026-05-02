from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
    
    def ready(self):
        """Load models when app is ready."""
        try:
            from .inference import inference_service
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Detector app loaded - models initialized")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error initializing detector app: {e}")
