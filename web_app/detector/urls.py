"""
URL Configuration for detector app
"""

from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.index, name='index'),
    path('ppe/', views.ppe_page, name='ppe-page'),
    path('fish/', views.fish_page, name='fish-page'),
    path('animaux/', views.animaux_page, name='animaux-page'),
    path('illegal-mining/', views.illegal_mining_page, name='illegal-mining'),
    path('smoke/', views.smoke_page, name='smoke-page'),
    path('api/animaux/compare/', views.animaux_compare, name='animaux-compare'),
    path('api/detect/image/', views.ImageUploadView.as_view(), name='detect-image'),
    path('api/explain/heatmap/', views.HeatmapView.as_view(), name='heatmap'),
    path('api/models/', views.ModelListView.as_view(), name='models'),
    
    # Video: upload then stream
    path('api/upload/video/', views.upload_video, name='upload-video'),
    path('api/stream/video/<str:session_id>/', views.stream_video, name='stream-video'),
    
    # Webcam: direct MJPEG stream
    path('api/stream/webcam/', views.stream_webcam, name='stream-webcam'),
    
    # Deforestation detection
    path('predict/', views.predict_page, name='predict-page'),
    path('api/deforestation/', views.predict_deforestation, name='predict-deforestation'),
    
    # Mining segmentation
    path('mining/', views.mining_page, name='mining-page'),
    path('api/mining/', views.segment_mining_sites, name='segment-mining'),
    
    # Mining Areas & Soil Health
    path('mining-areas/', views.mining_areas_page, name='mining-areas-page'),
    path('api/soil-health/', views.segment_soil_health_api, name='segment-soil-health'),

    # Chart data for dashboard
    path('api/chart-data/', views.chart_data_api, name='chart-data'),
    
    # Ollama Chat API
    path('api/chat/', views.chat_api, name='chat-api'),
    
    # User Profile & Settings
    path('profile/', views.profile_page, name='profile-page'),
    path('settings/', views.settings_page, name='settings-page'),
    
    # OCR & User Management
    path('api/ocr-id-card/', views.ocr_id_card, name='ocr-id-card'),
    path('user-management/', views.user_management_page, name='user-management'),
]
