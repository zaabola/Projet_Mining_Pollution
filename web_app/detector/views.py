"""
Views for PPE Detection Web App
"""

import os
import json
import logging
import cv2
import base64
import uuid
import requests
from io import BytesIO
from django.http import JsonResponse, FileResponse, StreamingHttpResponse
from django.views import View
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.conf import settings
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login as auth_login
from django.contrib import messages
from .inference import inference_service
from .fish_tracking import FishTracker
from .warning_logger import log_detection, get_summary
import numpy as np

try:
    from ml.model import detect_deforestation, MODEL_LOADED as DEFORESTATION_MODEL_LOADED
    DEFORESTATION_AVAILABLE = DEFORESTATION_MODEL_LOADED
    if not DEFORESTATION_MODEL_LOADED:
        logger_temp = logging.getLogger(__name__)
        logger_temp.warning("Deforestation model failed to load. Check web_app/ml/deforestation_unet_deploy.pth")
except (ImportError, ModuleNotFoundError) as e:
    DEFORESTATION_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.error(f"Cannot import deforestation model: {e}")

try:
    from ml.mining_segmentation import segment_image, MINING_MODEL_LOADED
    MINING_AVAILABLE = MINING_MODEL_LOADED
    if not MINING_MODEL_LOADED:
        logger_temp = logging.getLogger(__name__)
        logger_temp.warning("Mining segmentation model failed to load.")
except (ImportError, ModuleNotFoundError) as e:
    MINING_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.error(f"Cannot import mining segmentation model: {e}")

try:
    from ml.soil_health_ghada import segment_soil_health, SOIL_HEALTH_MODEL_LOADED
    SOIL_HEALTH_AVAILABLE = SOIL_HEALTH_MODEL_LOADED
    if not SOIL_HEALTH_MODEL_LOADED:
        logger_temp = logging.getLogger(__name__)
        logger_temp.warning("Soil health segmentation model (Ghada) failed to load.")
except (ImportError, ModuleNotFoundError) as e:
    SOIL_HEALTH_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.error(f"Cannot import soil health segmentation model: {e}")

logger = logging.getLogger(__name__)


class ImageUploadView(View):
    """Handle image upload and inference."""
    
    @method_decorator(require_http_methods(["POST"]))
    def post(self, request):
        try:
            if 'image' not in request.FILES:
                return JsonResponse({'error': 'No image provided'}, status=400)
            
            model_name = request.POST.get('model', 'helmet')
            conf_threshold = float(request.POST.get('confidence', settings.INFERENCE_CONFIDENCE))
            include_heatmap = request.POST.get('heatmap', 'false').lower() == 'true'
            
            uploaded_file = request.FILES['image']
            
            # Validate file
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
                return JsonResponse({'error': f'Invalid file type. Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}'}, status=400)
            
            if uploaded_file.size > settings.MAX_UPLOAD_SIZE:
                return JsonResponse({'error': f'File too large. Max size: {settings.MAX_UPLOAD_SIZE} bytes'}, status=400)
            
            mode = request.POST.get('mode', 'ppe')
            
            # Save uploaded file
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            with open(file_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            
            response_data = {}
            warn_count = 0
            det_count = 0
            class_breakdown = {}
            
            if mode == 'ppe':
                # Use combined logic for PPE image
                frame = cv2.imread(file_path)
                annotated_frame, detections, _ = inference_service.predict_frame_combined(frame, conf_threshold, {})
                
                annotated_path = file_path.replace('.', '_annotated_ppe.')
                cv2.imwrite(annotated_path, annotated_frame)
                
                response_data = detections.copy()
                response_data['success'] = True
                response_data['annotated_image_url'] = f'/media/{os.path.basename(annotated_path)}'
                
                det_count = detections.get('total_persons', 0)
                warn_count = detections.get('unsafe_count', 0)
                
                # Format for frontend display
                frontend_dets = []
                for p in detections.get('people', []):
                    status = p['status']
                    frontend_dets.append({
                        'class_name': f"Person: {status}",
                        'confidence': 1.0
                    })
                response_data['detections'] = frontend_dets
                
            else:
                # Standard single-model logic for others
                annotated_path, predictions = inference_service.predict_with_visualization(
                    file_path, model_name, conf_threshold
                )
                
                response_data = predictions.copy()
                if annotated_path and os.path.exists(annotated_path):
                    response_data['annotated_image_url'] = f'/media/{os.path.basename(annotated_path)}'
                
                dets = predictions.get('detections', [])
                det_count = len(dets)
                
                for d in dets:
                    cn = d.get('class_name', 'unknown')
                    class_breakdown[cn] = class_breakdown.get(cn, 0) + 1
                    # For smoke / illegal mining, every detection is a warning
                    if model_name in ('smoke', 'ahmed'):
                        warn_count += 1
                    elif mode == 'animaux':
                        warn_count += 1
            
            # Generate heatmap if requested
            if include_heatmap:
                heatmap_model = 'helmet' if mode == 'ppe' else model_name
                heatmap_result = inference_service.generate_explainability_heatmap(file_path, heatmap_model)
                if heatmap_result.get('success'):
                    response_data['heatmap_url'] = f'/media/{os.path.basename(heatmap_result["heatmap_path"])}'
                else:
                    response_data['heatmap_error'] = heatmap_result.get('error')
            
            # --- Log this detection for chart tracking ---
            module_map = {'helmet': 'ppe', 'mask': 'ppe', 'gasmask': 'ppe',
                          'fish': 'fish', 'animaux': 'animaux',
                          'ahmed': 'illegal_mining', 'smoke': 'smoke'}
            log_module = 'ppe' if mode == 'ppe' else module_map.get(model_name, model_name)
            
            log_detection(
                module=log_module,
                detections_count=det_count,
                warnings_count=warn_count,
                details={'classes': class_breakdown, 'model': model_name, 'source': 'image'}
            )
            
            return JsonResponse(response_data)
        
        except Exception as e:
            logger.error(f"Image upload error: {e}")
            return JsonResponse({'error': str(e)}, status=500)


# ==================== Video Upload + MJPEG Stream ====================

# Store uploaded video paths for streaming
_video_sessions = {}


@require_http_methods(["POST"])
def upload_video(request):
    """Upload video and return a stream URL for live processing."""
    try:
        if 'video' not in request.FILES:
            return JsonResponse({'error': 'No video provided'}, status=400)
        
        conf_threshold = float(request.POST.get('confidence', settings.INFERENCE_CONFIDENCE))
        uploaded_file = request.FILES['video']
        
        # Validate file
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
            return JsonResponse({'error': f'Invalid file type. Allowed: {settings.ALLOWED_VIDEO_EXTENSIONS}'}, status=400)
        
        if uploaded_file.size > settings.MAX_UPLOAD_SIZE:
            return JsonResponse({'error': f'File too large. Max size: {settings.MAX_UPLOAD_SIZE} bytes'}, status=400)
        
        # Save uploaded file with unique name
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        session_id = str(uuid.uuid4())[:8]
        file_name = f"video_{session_id}.{file_ext}"
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        with open(file_path, 'wb+') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        mode = request.POST.get('mode', 'ppe')
        
        # Store session
        _video_sessions[session_id] = {
            'path': file_path,
            'confidence': conf_threshold,
            'mode': mode,
        }
        
        stream_url = f'/api/stream/video/{session_id}/'
        
        return JsonResponse({
            'success': True,
            'stream_url': stream_url,
            'session_id': session_id,
        })
    
    except Exception as e:
        logger.error(f"Video upload error: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def stream_video(request, session_id):
    """Serve MJPEG stream of processed video frames with integrated PPE detection."""
    if session_id not in _video_sessions:
        return JsonResponse({'error': 'Session not found'}, status=404)
    
    session = _video_sessions[session_id]
    video_path = session['path']
    conf_threshold = session['confidence']
    mode = session.get('mode', 'ppe')
    
    def generate_frames():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        if mode == 'fish':
            fish_tracker = FishTracker()
            fish_model = inference_service.models.get('fish')
            if not fish_model:
                logger.error("Fish model not loaded")
                return
        else:
            person_histories = {}
        
        # Accumulate stats for warning logging
        total_detections = 0
        total_warnings = 0
        total_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                
                if mode == 'fish':
                    annotated_frame = fish_tracker.process_frame(fish_model, frame, conf_threshold)
                    # Count fish detections from the tracker
                    fish_res = fish_model.predict(frame, conf=conf_threshold, verbose=False)
                    if fish_res and fish_res[0].boxes is not None:
                        total_detections += len(fish_res[0].boxes)
                elif mode in ['animaux', 'ahmed', 'smoke']:
                    single_model = inference_service.models.get(mode)
                    if single_model:
                        results = single_model.predict(frame, conf=conf_threshold, verbose=False)
                        annotated_frame = results[0].plot()
                        frame_det_count = len(results[0].boxes) if results[0].boxes is not None else 0
                        total_detections += frame_det_count
                        # For smoke/ahmed, every detection is a warning
                        if mode in ['smoke', 'ahmed']:
                            total_warnings += frame_det_count
                        # For animaux, any detection is a warning (animals present)
                        elif mode == 'animaux' and frame_det_count > 0:
                            total_warnings += frame_det_count
                    else:
                        annotated_frame = frame
                else:
                    # PPE mode
                    annotated_frame, detections, person_histories = inference_service.predict_frame_combined(
                        frame, conf_threshold, person_histories
                    )
                    total_detections += detections.get('total_persons', 0)
                    total_warnings += detections.get('unsafe_count', 0)
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
            # Clean up session
            if session_id in _video_sessions:
                del _video_sessions[session_id]
            
            # Log accumulated warnings from this video session
            module_map = {'ppe': 'ppe', 'fish': 'fish', 'animaux': 'animaux',
                          'ahmed': 'illegal_mining', 'smoke': 'smoke'}
            log_detection(
                module=module_map.get(mode, mode),
                detections_count=total_detections,
                warnings_count=total_warnings,
                details={'source': 'video', 'frames': total_frames}
            )
    
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


# ==================== Webcam MJPEG Stream ====================

@csrf_exempt
def stream_webcam(request):
    """Serve MJPEG stream from webcam with integrated PPE detection."""
    conf_threshold = float(request.GET.get('confidence', settings.INFERENCE_CONFIDENCE))
    duration = int(request.GET.get('duration', 30))
    mode = request.GET.get('mode', 'ppe')
    
    def generate_frames():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not access webcam")
            return
        
        if mode == 'fish':
            fish_tracker = FishTracker()
            fish_model = inference_service.models.get('fish')
            if not fish_model:
                logger.error("Fish model not loaded")
                return
        else:
            person_histories = {}
            
        frame_count = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_limit = fps * duration
        
        # Accumulate stats for warning logging
        total_detections = 0
        total_warnings = 0
        
        try:
            while frame_count < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize for consistent processing
                frame = cv2.resize(frame, (640, 480))
                
                if mode == 'fish':
                    annotated_frame = fish_tracker.process_frame(fish_model, frame, conf_threshold)
                    fish_res = fish_model.predict(frame, conf=conf_threshold, verbose=False)
                    if fish_res and fish_res[0].boxes is not None:
                        total_detections += len(fish_res[0].boxes)
                elif mode in ['animaux', 'ahmed', 'smoke']:
                    single_model = inference_service.models.get(mode)
                    if single_model:
                        results = single_model.predict(frame, conf=conf_threshold, verbose=False)
                        annotated_frame = results[0].plot()
                        frame_det_count = len(results[0].boxes) if results[0].boxes is not None else 0
                        total_detections += frame_det_count
                        if mode in ['smoke', 'ahmed']:
                            total_warnings += frame_det_count
                        elif mode == 'animaux' and frame_det_count > 0:
                            total_warnings += frame_det_count
                    else:
                        annotated_frame = frame
                else:
                    # PPE mode
                    annotated_frame, detections, person_histories = inference_service.predict_frame_combined(
                        frame, conf_threshold, person_histories
                    )
                    total_detections += detections.get('total_persons', 0)
                    total_warnings += detections.get('unsafe_count', 0)
                
                frame_count += 1
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
            
            # Log accumulated warnings from this webcam session
            module_map = {'ppe': 'ppe', 'fish': 'fish', 'animaux': 'animaux',
                          'ahmed': 'illegal_mining', 'smoke': 'smoke'}
            log_detection(
                module=module_map.get(mode, mode),
                detections_count=total_detections,
                warnings_count=total_warnings,
                details={'source': 'webcam', 'frames': frame_count}
            )
    
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


class HeatmapView(View):
    """Generate explainability heatmaps."""
    
    @method_decorator(require_http_methods(["POST"]))
    def post(self, request):
        try:
            if 'image' not in request.FILES:
                return JsonResponse({'error': 'No image provided'}, status=400)
            
            model_name = request.POST.get('model', 'helmet')
            
            uploaded_file = request.FILES['image']
            
            # Validate file
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
                return JsonResponse({'error': 'Invalid file type'}, status=400)
            
            # Save uploaded file
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            with open(file_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            
            # Generate heatmap
            result = inference_service.generate_explainability_heatmap(file_path, model_name)
            
            if result.get('success'):
                result['heatmap_url'] = f'/media/{os.path.basename(result["heatmap_path"])}'
                del result['heatmap_path']
            
            return JsonResponse(result)
        
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            return JsonResponse({'error': str(e)}, status=500)


class ModelListView(View):
    """Get list of available models."""
    
    @method_decorator(require_http_methods(["GET"]))
    def get(self, request):
        models = {
            'available_models': list(inference_service.models.keys()),
            'loaded_models': [k for k, v in inference_service.models.items() if v is not None],
            'failed_models': [k for k, v in inference_service.models.items() if v is None],
        }
        return JsonResponse(models)


def index(request):
    """Dashboard home page."""
    summary = get_summary()
    return render(request, 'index.html', {
        'active_page': 'dashboard',
        'chart_summary_json': json.dumps(summary),
    })


def chart_data_api(request):
    """API endpoint returning chart data as JSON."""
    return JsonResponse(get_summary())


def ppe_page(request):
    """PPE Detection page."""
    return render(request, 'ppe.html', {'active_page': 'ppe'})


def fish_page(request):
    """Fish Tracking page."""
    return render(request, 'fish.html', {'active_page': 'fish'})


def animaux_page(request):
    """Animal Detection page."""
    return render(request, 'animaux.html', {'active_page': 'animaux'})


def illegal_mining_page(request):
    """Illegal Mining Detection page (Ahmed YOLO)."""
    return render(request, 'illegal_mining.html', {'active_page': 'illegal_mining'})


def smoke_page(request):
    """Smoke Detection page."""
    return render(request, 'smoke.html', {'active_page': 'smoke'})


@csrf_exempt
def animaux_compare(request):
    """Compare animals before and after."""
    if request.method == "POST":
        old_file = request.FILES.get("image_before")
        new_file = request.FILES.get("image_after")

        if old_file is None or new_file is None:
            return JsonResponse({"error": "Please send two images (before and after)."}, status=400)
            
        animaux_model = inference_service.models.get('animaux')
        if not animaux_model:
            return JsonResponse({"error": "Animal model not available."}, status=500)

        conf_threshold = float(request.POST.get('confidence', settings.INFERENCE_CONFIDENCE))

        try:
            # Read images as bytes
            old_bytes = np.frombuffer(old_file.read(), np.uint8)
            new_bytes = np.frombuffer(new_file.read(), np.uint8)

            image_old = cv2.imdecode(old_bytes, cv2.IMREAD_COLOR)
            image_new = cv2.imdecode(new_bytes, cv2.IMREAD_COLOR)

            if image_old is None or image_new is None:
                return JsonResponse({"error": "Invalid image file."}, status=400)

            # Predict
            res_old = animaux_model.predict(image_old, conf=conf_threshold, verbose=False)[0]
            res_new = animaux_model.predict(image_new, conf=conf_threshold, verbose=False)[0]

            count_old = len(res_old.boxes) if res_old.boxes else 0
            count_new = len(res_new.boxes) if res_new.boxes else 0

            # Encode to base64
            _, buffer_old = cv2.imencode('.jpg', res_old.plot())
            _, buffer_new = cv2.imencode('.jpg', res_new.plot())

            b64_old = base64.b64encode(buffer_old).decode()
            b64_new = base64.b64encode(buffer_new).decode()
            
            # Determine logic
            decreased = count_new < count_old
            warning = False
            message = ""
            
            if count_new > 0 or count_old > 0:
                warning = True
                message = "⚠ WARNING: There are animals here. DO NOT MINE HERE!"
            else:
                message = "✅ No animals detected. Safe to proceed."

            # --- Log this detection for chart tracking ---
            warn_count = 1 if warning else 0
            log_detection(
                module='animaux',
                detections_count=count_new + count_old,
                warnings_count=warn_count,
                details={'count_before': count_old, 'count_after': count_new, 'type': 'comparison'}
            )

            return JsonResponse({
                "status": "success",
                "count_before": count_old,
                "count_after": count_new,
                "decreased": decreased,
                "warning": warning,
                "message": message,
                "image_before": f"data:image/jpeg;base64,{b64_old}",
                "image_after": f"data:image/jpeg;base64,{b64_new}"
            })

        except Exception as e:
            logger.error(f"Animal compare error: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed."}, status=405)


# ==================== Deforestation Detection ====================

def predict_page(request):
    """Render deforestation prediction page."""
    return render(request, 'predict.html', {'active_page': 'deforestation'})


def predict_deforestation(request):
    """Detect deforestation by comparing before and after satellite images."""
    if request.method == "POST":
        old_file = request.FILES.get("image_old")
        new_file = request.FILES.get("image_new")

        if old_file is None or new_file is None:
            return JsonResponse({"error": "Please send two images (old and new)."}, status=400)

        if not DEFORESTATION_AVAILABLE:
            return JsonResponse({"error": "Deforestation model not available."}, status=500)

        try:
            # Read images as bytes
            old_bytes = np.frombuffer(old_file.read(), np.uint8)
            new_bytes = np.frombuffer(new_file.read(), np.uint8)

            # Decode images
            image_old = cv2.imdecode(old_bytes, cv2.IMREAD_COLOR)
            image_new = cv2.imdecode(new_bytes, cv2.IMREAD_COLOR)

            if image_old is None or image_new is None:
                return JsonResponse({"error": "Invalid image file."}, status=400)

            # Run deforestation detection
            result = detect_deforestation(image_old, image_new)

            # Convert masks to colored visualizations
            forest_old = result["forest_old"]
            forest_new = result["forest_new"]
            deforestation_mask = result["deforestation_mask"]

            # Create visualization: green for forest, red for deforestation
            height, width = forest_old.shape
            
            # Forest visualization (old)
            forest_old_vis = np.zeros((height, width, 3), dtype=np.uint8)
            forest_old_vis[forest_old == 1] = [0, 255, 0]  # Green for forest
            
            # Forest visualization (new)
            forest_new_vis = np.zeros((height, width, 3), dtype=np.uint8)
            forest_new_vis[forest_new == 1] = [0, 255, 0]  # Green for forest
            
            # Deforestation visualization (red)
            deforestation_vis = np.zeros((height, width, 3), dtype=np.uint8)
            deforestation_vis[deforestation_mask == 1] = [0, 0, 255]  # Red for deforestation

            # Encode masks to JPEG base64
            _, forest_old_buffer = cv2.imencode('.jpg', forest_old_vis)
            _, forest_new_buffer = cv2.imencode('.jpg', forest_new_vis)
            _, deforestation_buffer = cv2.imencode('.jpg', deforestation_vis)

            forest_old_b64 = base64.b64encode(forest_old_buffer).decode()
            forest_new_b64 = base64.b64encode(forest_new_buffer).decode()
            deforestation_b64 = base64.b64encode(deforestation_buffer).decode()

            deforestation_percent = round(result["percent"], 2)
            
            # --- Log this detection for chart tracking ---
            warn_count = 1 if deforestation_percent > 3.0 else 0
            log_detection(
                module='deforestation',
                detections_count=1,
                warnings_count=warn_count,
                details={'percent': deforestation_percent}
            )

            return JsonResponse({
                "deforestation_percent": deforestation_percent,
                "status": "success",
                "forest_old_image": f"data:image/jpeg;base64,{forest_old_b64}",
                "forest_new_image": f"data:image/jpeg;base64,{forest_new_b64}",
                "deforestation_image": f"data:image/jpeg;base64,{deforestation_b64}"
            })

        except Exception as e:
            logger.error(f"Deforestation detection error: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed."}, status=405)


# ==================== Mining Segmentation ====================

def mining_page(request):
    """Render mining segmentation page."""
    return render(request, 'mining.html', {'active_page': 'mining'})


def segment_mining_sites(request):
    """Segment mining and residential sites in satellite image."""
    if request.method == "POST":
        image_file = request.FILES.get("image")

        if image_file is None:
            return JsonResponse({"error": "Please upload a satellite image."}, status=400)

        if not MINING_AVAILABLE:
            return JsonResponse({"error": "Mining segmentation model not available."}, status=500)

        try:
            # Read image as bytes
            image_bytes = np.frombuffer(image_file.read(), np.uint8)

            # Decode image
            image_np = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if image_np is None:
                return JsonResponse({"error": "Invalid image file."}, status=400)

            # Run segmentation
            result = segment_image(image_np)

            # Encode results to base64
            _, colored_buffer = cv2.imencode('.jpg', result["colored_mask"])
            _, overlay_buffer = cv2.imencode('.jpg', result["overlay"])

            colored_b64 = base64.b64encode(colored_buffer).decode()
            overlay_b64 = base64.b64encode(overlay_buffer).decode()

            msite_percent = round(result["msite_percent"], 2)
            ressite_percent = round(result["ressite_percent"], 2)
            
            # --- Log this detection for chart tracking ---
            warn_count = 1 if msite_percent > 0 else 0
            log_detection(
                module='mining',
                detections_count=1,
                warnings_count=warn_count,
                details={'msite_percent': msite_percent, 'ressite_percent': ressite_percent}
            )

            return JsonResponse({
                "status": "success",
                "colored_mask": f"data:image/jpeg;base64,{colored_b64}",
                "overlay": f"data:image/jpeg;base64,{overlay_b64}",
                "ressite_percent": ressite_percent,
                "msite_percent": msite_percent,
                "ressite_pixels": result["ressite_pixels"],
                "msite_pixels": result["msite_pixels"],
                "height": result["height"],
                "width": result["width"]
            })

        except Exception as e:
            logger.error(f"Mining segmentation error: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed."}, status=405)


# ==================== Mining Areas & Soil Health ====================

def mining_areas_page(request):
    """Render mining areas and soil health page."""
    return render(request, 'mining_areas.html', {'active_page': 'mining_areas'})


def segment_soil_health_api(request):
    """Segment mining areas to determine soil health."""
    if request.method == "POST":
        image_file = request.FILES.get("image")

        if image_file is None:
            return JsonResponse({"error": "Please upload a satellite image."}, status=400)

        if not SOIL_HEALTH_AVAILABLE:
            return JsonResponse({"error": "Soil Health model not available."}, status=500)

        try:
            # Read image as bytes
            image_bytes = np.frombuffer(image_file.read(), np.uint8)

            # Decode image
            image_np = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if image_np is None:
                return JsonResponse({"error": "Invalid image file."}, status=400)

            # Run segmentation
            result = segment_soil_health(image_np)

            # Encode results to base64
            _, overlay_buffer = cv2.imencode('.jpg', result["overlay"])
            overlay_b64 = base64.b64encode(overlay_buffer).decode()

            mining_percent = round(result["mining_percent"], 2)
            
            # --- Log this detection for chart tracking ---
            warn_count = 1 if mining_percent > 0.5 else 0
            log_detection(
                module='mining', # reusing mining module logging
                detections_count=1,
                warnings_count=warn_count,
                details={'mining_percent': mining_percent, 'type': 'soil_health'}
            )

            return JsonResponse({
                "status": "success",
                "overlay": f"data:image/jpeg;base64,{overlay_b64}",
                "mining_percent": mining_percent,
                "mining_pixels": result["mining_pixels"],
                "height": result["height"],
                "width": result["width"]
            })

        except Exception as e:
            logger.error(f"Soil health segmentation error: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed."}, status=405)


# ==================== Ollama Chat Integration ====================

# Cache the dataset strings
_GABES_CROP_CSV = ""
_GABES_SOIL_CSV = ""

def load_gabes_data():
    global _GABES_CROP_CSV, _GABES_SOIL_CSV
    try:
        if not _GABES_CROP_CSV:
            crop_path = os.path.join(settings.PROJECT_ROOT, 'datasets', 'gabes_crop_recommendation.csv')
            if os.path.exists(crop_path):
                with open(crop_path, 'r') as f:
                    _GABES_CROP_CSV = f.read()
        
        if not _GABES_SOIL_CSV:
            soil_path = os.path.join(settings.PROJECT_ROOT, 'datasets', 'gabescsv', 'gabes_soil_dataset.csv')
            if os.path.exists(soil_path):
                with open(soil_path, 'r') as f:
                    _GABES_SOIL_CSV = f.read()
    except Exception as e:
        logger.error(f"Failed to load Gabes datasets for Ollama context: {e}")

@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            messages = data.get('messages', [])
            language = data.get('language', 'English')
            
            if not messages:
                return JsonResponse({'error': 'No messages provided'}, status=400)
            
            load_gabes_data()
            
            # Fetch live dashboard stats
            summary = get_summary()
            
            if language == 'French':
                stats_str = f"Détections totales: {summary.get('total_detections', 0)}\nAvertissements totaux (DANGER): {summary.get('total_warnings', 0)}\nSécurité totale (SAFE): {summary.get('total_safe', 0)}\n\nDétails par module:\n"
                for mod, stat in summary.get('module_stats', {}).items():
                    stats_str += f"- {mod.upper()}: {stat['warnings']} avertissements sur {stat['detections']} détections\n"
                
                system_prompt = f"""Tu es un assistant IA environnemental pour EcoGuard.
IMPORTANT: Tu dois répondre EXCLUSIVEMENT en Français! Ne parle jamais anglais.

Statistiques du tableau de bord (résumé des détections):
{stats_str}

Règles de formatage:
1. GARDER TRÈS COURT: 2 ou 3 phrases maximum.
2. COULEURS: Utilise exactement la syntaxe [DANGER: mot] pour les choses dangereuses et [SAFE: mot] pour les bonnes choses. N'enveloppe JAMAIS une phrase entière, seulement les mots ou les nombres! Exemple: [DANGER: 335 avertissements] et [SAFE: 64 sécurité]
3. EMOJIS: Utilise beaucoup d'emojis pertinents (🏭, 💧, ⚠️, 🌱, 📊) pour rendre le texte vivant.
"""
            elif language == 'Arabic':
                stats_str = f"إجمالي الاكتشافات: {summary.get('total_detections', 0)}\nإجمالي التحذيرات (DANGER): {summary.get('total_warnings', 0)}\nإجمالي الحالات الآمنة (SAFE): {summary.get('total_safe', 0)}\n\nتفاصيل كل وحدة:\n"
                for mod, stat in summary.get('module_stats', {}).items():
                    stats_str += f"- {mod.upper()}: {stat['warnings']} تحذيرات من أصل {stat['detections']} اكتشافات\n"
                    
                system_prompt = f"""أنت مساعد بيئي ذكي لمنصة EcoGuard.
هام جدا: يجب عليك الرد باللغة العربية فقط! لا تستخدم اللغة الإنجليزية أبداً.

إحصائيات لوحة التحكم (ملخص الاكتشافات):
{stats_str}

قواعد التنسيق:
1. الإيجاز: اجعل إجابتك قصيرة جدا (جملتين أو ثلاث كحد أقصى).
2. الألوان: استخدم بالضبط التنسيق [DANGER: كلمة] للأشياء الخطيرة و [SAFE: كلمة] للأشياء الآمنة. لا تضع الجمل الكاملة داخل القوسين أبدا، بل فقط الأرقام أو الكلمات القصيرة! مثال: [DANGER: 335 تحذير] و [SAFE: 64 آمن]
3. الرموز التعبيرية: استخدم الرموز التعبيرية (🏭، 💧، ⚠️، 🌱، 📊) لجعل النص أكثر حيوية.
"""
            else:
                stats_str = f"Total System Detections: {summary.get('total_detections', 0)}\nTotal System Warnings (DANGER): {summary.get('total_warnings', 0)}\nTotal System Safe (SAFE): {summary.get('total_safe', 0)}\n\nBreakdown per module:\n"
                for mod, stat in summary.get('module_stats', {}).items():
                    stats_str += f"- {mod.upper()}: {stat['warnings']} warnings out of {stat['detections']} detections\n"
                    
                system_prompt = f"""You are an environmental AI assistant for the EcoGuard Platform.
IMPORTANT: You must respond in English.

Dashboard Statistics:
{stats_str}

Formatting Rules:
1. BREVITY: Keep it extremely short (2-3 sentences max).
2. COLORS: Use exactly [DANGER: word] for bad things and [SAFE: word] for good things. NEVER wrap an entire sentence, only short words/numbers! Example: [DANGER: 335 warnings] and [SAFE: 64 safe]
3. EMOJIS: Use relevant emojis abundantly (🏭, 💧, ⚠️, 🌱, 📊) to make the text engaging.
"""
            
            # Common knowledge base to append to system prompt
            system_prompt += f"""
KNOWLEDGE BASE (Use these as examples of pollution/agriculture if needed):
{_GABES_CROP_CSV}
{_GABES_SOIL_CSV}
"""
            
            # Massive English datasets cause English bias. Re-assert language forcefully at the very end of the prompt.
            if language == 'French':
                system_prompt += "\n\nRAPPEL CRITIQUE FINALE: TU DOIS RÉPONDRE EXCLUSIVEMENT EN FRANÇAIS. L'ANGLAIS EST STRICTEMENT INTERDIT."
            elif language == 'Arabic':
                system_prompt += "\n\nتذكير نهائي هام جدا: يجب عليك الرد باللغة العربية حصراً. يمنع منعاً باتاً استخدام اللغة الإنجليزية."
            
            # Prepend system prompt to messages
            ollama_messages = [{"role": "system", "content": system_prompt}] + messages
            
            # Send to local Ollama instance (using mistral)
            response = requests.post(
                'http://localhost:11434/api/chat',
                json={
                    "model": "mistral",
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.5
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return JsonResponse({
                    "status": "success",
                    "message": result['message']
                })
            else:
                logger.error(f"Ollama API Error: {response.text}")
                return JsonResponse({"error": "Failed to generate response from Ollama"}, status=500)
                
        except Exception as e:
            logger.error(f"Chat API exception: {e}")
            return JsonResponse({"error": str(e)}, status=500)
            
    return JsonResponse({"error": "Method not allowed"}, status=405)


# ==================== Auth Views ====================

def register(request):
    from .models import UserProfile
    from .ocr_utils import generate_email
    
    if request.user.is_authenticated:
        return redirect('/')
        
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False  # Inactive until admin approves
            user.save()
            
            # Extract OCR data from hidden fields
            extracted_first = request.POST.get('extracted_first_name', '')
            extracted_last = request.POST.get('extracted_last_name', '')
            gen_email = request.POST.get('generated_email', '')
            
            if not gen_email and extracted_first:
                gen_email = generate_email(extracted_first, extracted_last)
            
            # Save profile
            profile = UserProfile.objects.create(
                user=user,
                role='employee',
                is_approved=False,
                extracted_first_name=extracted_first,
                extracted_last_name=extracted_last,
                generated_email=gen_email,
            )
            
            # Save ID card image
            if 'id_card' in request.FILES:
                profile.id_card_image = request.FILES['id_card']
                profile.save()
            
            user.email = gen_email
            user.first_name = extracted_first
            user.last_name = extracted_last
            user.save()
            
            return render(request, 'accounts/pending.html', {
                'username': user.username,
                'generated_email': gen_email,
            })
    else:
        form = UserCreationForm()
    
    return render(request, 'accounts/register.html', {'form': form})


@csrf_exempt
def ocr_id_card(request):
    """API endpoint to extract name from uploaded ID card image."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    
    if 'id_card' not in request.FILES:
        return JsonResponse({'error': 'No image'}, status=400)
    
    import tempfile
    from .ocr_utils import extract_name_from_id, generate_email
    
    try:
        uploaded = request.FILES['id_card']
        # Save to temp file for OCR
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        first_name, last_name = extract_name_from_id(tmp_path)
        email = generate_email(first_name, last_name)
        
        # Generate a secure random password
        import string
        import random
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        password = "Eco" + "".join(random.choice(chars) for _ in range(10))
        
        # Clean up
        import os
        os.unlink(tmp_path)
        
        return JsonResponse({
            'success': True,
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'password': password,
        })
    except Exception as e:
        logger.error(f"OCR API error: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def user_management_page(request):
    """Admin-only user management page."""
    from .models import UserProfile
    
    if not request.user.is_authenticated or not request.user.is_superuser:
        return redirect('/')
    
    # Handle POST actions (approve, reject, delete)
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        action = request.POST.get('action')
        
        try:
            from django.contrib.auth.models import User
            target_user = User.objects.get(id=user_id)
            profile = UserProfile.objects.get(user=target_user)
            
            if action == 'approve':
                profile.is_approved = True
                profile.save()
                target_user.is_active = True
                target_user.save()
                messages.success(request, f'User "{target_user.username}" has been approved!')
                
            elif action == 'reject':
                username = target_user.username
                target_user.delete()
                messages.warning(request, f'User "{username}" has been rejected and removed.')
                
            elif action == 'delete':
                username = target_user.username
                target_user.delete()
                messages.error(request, f'User "{username}" has been deleted.')
                
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')
        
        return redirect('/user-management/')
    
    # Ensure all users have profiles
    from django.contrib.auth.models import User
    for user in User.objects.all():
        UserProfile.objects.get_or_create(
            user=user,
            defaults={
                'role': 'admin' if user.is_superuser else 'employee',
                'is_approved': True if user.is_superuser else False,
            }
        )
    
    all_profiles = UserProfile.objects.select_related('user').all()
    pending = all_profiles.filter(is_approved=False)
    approved = all_profiles.filter(is_approved=True)
    admins = all_profiles.filter(role='admin')
    
    return render(request, 'user_management.html', {
        'active_page': 'user_management',
        'all_users': all_profiles,
        'pending_users': pending,
        'pending_count': pending.count(),
        'approved_count': approved.count(),
        'total_count': all_profiles.count(),
        'admin_count': admins.count(),
    })


def profile_page(request):
    """User profile page with password change."""
    from django.contrib.auth.forms import PasswordChangeForm
    from django.contrib.auth import update_session_auth_hash

    if not request.user.is_authenticated:
        return redirect('/accounts/login/')

    if request.method == 'POST':
        password_form = PasswordChangeForm(request.user, request.POST)
        if password_form.is_valid():
            user = password_form.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Your password was updated successfully!')
            return redirect('/profile/')
    else:
        password_form = PasswordChangeForm(request.user)

    return render(request, 'profile.html', {
        'active_page': 'profile',
        'password_form': password_form,
    })


def settings_page(request):
    """Settings page with dark mode toggle."""
    if not request.user.is_authenticated:
        return redirect('/accounts/login/')
    return render(request, 'settings.html', {'active_page': 'settings'})


