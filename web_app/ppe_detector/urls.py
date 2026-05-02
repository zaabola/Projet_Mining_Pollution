"""
Main URL Configuration for PPE Detector
"""

from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from detector import views as detector_views

urlpatterns = [
    path('accounts/', include('django.contrib.auth.urls')),
    path('accounts/register/', detector_views.register, name='register'),
    path('', include('detector.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
