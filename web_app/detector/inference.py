"""
Model Inference Service for PPE Detection
Handles loading YOLO models and running inference with explainability
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import threading
from django.conf import settings
import logging
from collections import deque, Counter
import math
import torch

logger = logging.getLogger(__name__)


class YOLOEigenCAM:
    """
    Produces an EigenCAM heatmap from a YOLOv8 backbone.
    Uses Principal Component Analysis (SVD) on feature maps.
    """
    def __init__(self, model: YOLO):
        self.model = model
        self.activations = None
        self._hook_handle = None

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _find_target_layer(self):
        target = None
        for m in self.model.model.model[15].modules():
            if isinstance(m, torch.nn.Conv2d):
                target = m
                break
        return target

    def generate(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """Generate EigenCAM heatmap for the image."""
        target = self._find_target_layer()
        if target is None:
            return None
            
        self._hook_handle = target.register_forward_hook(self._save_activation)
        
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            device = next(self.model.model.parameters()).device
            tensor = (
                torch.from_numpy(img_resized)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )
            
            with torch.no_grad():
                _ = self.model.model(tensor)
            
            if self.activations is None:
                return None
            
            batch_size, num_channels, h, w = self.activations.shape
            activations_cpu = self.activations.cpu().numpy()
            
            U, S, Vh = np.linalg.svd(
                activations_cpu[0].reshape(num_channels, h * w),
                full_matrices=False
            )
            
            eig_cam = np.abs(U[:, 0]).reshape(h, w)
            eig_cam = cv2.resize(eig_cam, (640, 640))
            eig_cam = (eig_cam - eig_cam.min()) / (eig_cam.max() - eig_cam.min() + 1e-7)
            
            return eig_cam
            
        finally:
            if self._hook_handle:
                self._hook_handle.remove()


# ==================== PPE Analysis Helpers ====================

def stable_vote(history, default_value):
    """Get most common value from history."""
    if len(history) == 0:
        return default_value
    return Counter(history).most_common(1)[0][0]


def box_center(box):
    """Get center coordinates of a bounding box."""
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def box_distance(c1, c2):
    """Calculate Euclidean distance between two centers."""
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def make_person_box_from_face(face_box, frame_w, frame_h):
    """Expand face box to full person box."""
    x1, y1, x2, y2 = face_box[:4]
    fw = x2 - x1
    fh = y2 - y1
    
    px1 = max(0, x1 - int(fw * 0.8))
    py1 = max(0, y1 - int(fh * 1.2))
    px2 = min(frame_w, x2 + int(fw * 0.8))
    py2 = min(frame_h, y2 + int(fh * 2.8))
    
    return (px1, py1, px2, py2)


def get_person_id(person_box):
    """Generate simple ID for person based on grid position."""
    cx, cy = box_center(person_box)
    return f"{cx//50}_{cy//50}"


# ==================== ModelInferenceService ====================

class ModelInferenceService:
    """
    Central service for all model inference operations.
    Manages model loading, inference, and explainability.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.models = {}
        self.cam_generators = {}
        self._load_models()
        self._initialized = True
    
    def _load_models(self):
        """Load all YOLO models from disk."""
        try:
            for model_name, model_path in settings.MODELS_CONFIG.items():
                try:
                    logger.info(f"Loading {model_name} model from {model_path}")
                    model = YOLO(model_path)
                    self.models[model_name] = model
                    self.cam_generators[model_name] = YOLOEigenCAM(model)
                    logger.info(f"Successfully loaded {model_name} model")
                except Exception as e:
                    logger.error(f"Failed to load {model_name} model: {e}")
                    self.models[model_name] = None
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict(self, image_path: str, model_name: str, conf_threshold: float = None) -> dict:
        """
        Run inference on an image.
        
        Args:
            image_path: Path to input image
            model_name: Name of the model to use ('helmet', 'mask', 'gasmask', 'fish')
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary with detections and metadata
        """
        if conf_threshold is None:
            conf_threshold = settings.INFERENCE_CONFIDENCE
        
        if model_name not in self.models or self.models[model_name] is None:
            return {'error': f'Model {model_name} not available'}
        
        try:
            model = self.models[model_name]
            results = model.predict(image_path, conf=conf_threshold, verbose=False)
            
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        detections.append({
                            'class_id': int(box.cls),
                            'class_name': model.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': box.xyxy[0].tolist(),
                        })
            
            return {
                'success': True,
                'model': model_name,
                'detections': detections,
                'image_shape': r.orig_shape if results else None,
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}
    
    def predict_with_visualization(self, image_path: str, model_name: str, conf_threshold: float = None) -> tuple:
        """
        Run inference and return annotated image.
        
        Returns:
            Tuple of (annotated_image_path, detections_dict)
        """
        if conf_threshold is None:
            conf_threshold = settings.INFERENCE_CONFIDENCE
        
        if model_name not in self.models or self.models[model_name] is None:
            return None, {'error': f'Model {model_name} not available'}
        
        try:
            model = self.models[model_name]
            
            if model_name == 'ahmed':
                image = cv2.imread(image_path)
                h, w = image.shape[:2]
                patch_size = 640
                overlap = 0.2
                stride = int(patch_size * (1 - overlap))
                
                all_boxes = []
                all_scores = []
                all_class_ids = []
                
                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        y1, x1 = y, x
                        y2, x2 = min(h, y + patch_size), min(w, x + patch_size)
                        
                        if y2 - y1 < 100 or x2 - x1 < 100:
                            continue
                            
                        patch = image[y1:y2, x1:x2]
                        res = model.predict(patch, conf=conf_threshold, imgsz=patch_size, verbose=False)[0]
                        
                        if res.boxes is not None and len(res.boxes) > 0:
                            for box, conf, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                                bx1, by1, bx2, by2 = box
                                all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                                all_scores.append(float(conf))
                                all_class_ids.append(int(cls))
                
                detections = []
                annotated_img = image.copy()
                if all_boxes:
                    cv_boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in all_boxes]
                    indices = cv2.dnn.NMSBoxes(cv_boxes, all_scores, conf_threshold, 0.4)
                    
                    if len(indices) > 0:
                        for i in indices.flatten():
                            box = all_boxes[i]
                            score = all_scores[i]
                            cls_id = all_class_ids[i]
                            cls_name = model.names[cls_id]
                            
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                            detections.append({
                                'class_id': cls_id,
                                'class_name': cls_name,
                                'confidence': score,
                                'bbox': [x1, y1, x2, y2]
                            })
                
                output_path = image_path.replace('.', f'_annotated_{model_name}.')
                cv2.imwrite(output_path, annotated_img)
                
                return output_path, {
                    'success': True,
                    'model': model_name,
                    'detections': detections,
                    'image_shape': image.shape,
                }
            
            else:
                results = model.predict(image_path, conf=conf_threshold, verbose=False)
                
                detections = []
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            detections.append({
                                'class_id': int(box.cls),
                                'class_name': model.names[int(box.cls)],
                                'confidence': float(box.conf),
                                'bbox': box.xyxy[0].tolist(),
                            })
                    
                    # Save annotated image
                    annotated_img = r.plot()
                    output_path = image_path.replace('.', f'_annotated_{model_name}.')
                    cv2.imwrite(output_path, annotated_img)
                
                return output_path, {
                    'success': True,
                    'model': model_name,
                    'detections': detections,
                    'image_shape': r.orig_shape if results else None,
                }
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return None, {'error': str(e)}

    def predict_video(self, video_path: str, model_name: str, conf_threshold: float = None, 
                     output_path: str = None) -> dict:
        """
        Run inference on a video file.
        """
        if conf_threshold is None:
            conf_threshold = settings.INFERENCE_CONFIDENCE
        
        if model_name not in self.models or self.models[model_name] is None:
            return {'error': f'Model {model_name} not available'}
        
        try:
            model = self.models[model_name]
            results = model.predict(video_path, conf=conf_threshold, save=True, verbose=False)
            
            detection_summary = {
                'total_frames': len(results),
                'frames_with_detections': 0,
                'model': model_name,
            }
            
            return {
                'success': True,
                'summary': detection_summary,
            }
        except Exception as e:
            logger.error(f"Video prediction error: {e}")
            return {'error': str(e)}
    
    def generate_explainability_heatmap(self, image_path: str, model_name: str) -> dict:
        """
        Generate EigenCAM heatmap for explainability.
        """
        if model_name not in self.cam_generators or self.cam_generators[model_name] is None:
            return {'error': f'CAM generator for {model_name} not available'}
        
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                return {'error': 'Could not read image'}
            
            cam_generator = self.cam_generators[model_name]
            heatmap = cam_generator.generate(img_bgr)
            
            if heatmap is None:
                return {'error': 'Could not generate heatmap'}
            
            # Convert heatmap to color
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Overlay on original image
            h, w = img_bgr.shape[:2]
            heatmap_resized = cv2.resize(heatmap_color, (w, h))
            overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_resized, 0.4, 0)
            
            # Save visualization
            output_path = image_path.replace('.', f'_heatmap_{model_name}.')
            cv2.imwrite(output_path, overlay)
            
            return {
                'success': True,
                'heatmap_path': output_path,
                'model': model_name,
            }
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            return {'error': str(e)}
    
    def predict_frame_combined(self, frame: np.ndarray, conf_threshold: float = None, 
                              person_histories: dict = None) -> tuple:
        """
        Run combined helmet + mask + gasmask detection on a single frame.
        Analyzes PPE compliance per person.
        
        Args:
            frame: Input frame (numpy array)
            conf_threshold: Confidence threshold
            person_histories: Dictionary to track PPE status over time
            
        Returns:
            Tuple of (annotated_frame, detections, person_statuses)
        """
        if conf_threshold is None:
            conf_threshold = settings.INFERENCE_CONFIDENCE
        
        if person_histories is None:
            person_histories = {}
        
        output = frame.copy()
        h, w, _ = frame.shape
        
        device = 0 if torch.cuda.is_available() else 'cpu'
        use_half = torch.cuda.is_available()
        
        # Run all 3 models using GPU and FP16 for speed
        helmet_results = self.models['helmet'].predict(frame, conf=conf_threshold, verbose=False, device=device, half=use_half)
        mask_results = self.models['mask'].predict(frame, conf=conf_threshold, verbose=False, device=device, half=use_half)
        gasmask_results = self.models['gasmask'].predict(frame, conf=conf_threshold, verbose=False, device=device, half=use_half)
        
        helmet_hats = []
        mask_detections = []
        gasmask_boxes = []
        
        # ==================== Extract Helmet Detections ====================
        for result in helmet_results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf >= conf_threshold and cls_id == 0:  # hat
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        helmet_hats.append((x1, y1, x2, y2, conf))
        
        # ==================== Extract Mask Detections ====================
        # Class mapping: 0=with_mask, 1=without_mask, 2=incorrect_mask
        for result in mask_results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf >= conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        mask_detections.append((x1, y1, x2, y2, cls_id, conf))
        
        # ==================== Extract Gas Mask Detections ====================
        for result in gasmask_results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf >= conf_threshold and cls_id in [0, 1]:  # 0=Oxygen_tube, 1=gasmask
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        gasmask_boxes.append((x1, y1, x2, y2, conf))
        
        # ==================== Build Person Boxes ====================
        person_candidates = []
        
        # From mask detections
        for m in mask_detections:
            person_candidates.append(make_person_box_from_face(m, w, h))
        
        # From gasmask detections
        for g in gasmask_boxes:
            person_candidates.append(make_person_box_from_face(g, w, h))
        
        # From helmet detections
        for hbox in helmet_hats:
            x1, y1, x2, y2, _ = hbox
            hw = x2 - x1
            hh = y2 - y1
            
            px1 = max(0, x1 - int(hw * 0.5))
            py1 = max(0, y1 - int(hh * 0.2))
            px2 = min(w, x2 + int(hw * 0.5))
            py2 = min(h, y2 + int(hh * 2.2))
            
            person_candidates.append((px1, py1, px2, py2))
        
        # Remove near-duplicate person boxes
        final_persons = []
        for candidate in person_candidates:
            cx1, cy1 = box_center(candidate)
            too_close = False
            
            for existing in final_persons:
                cx2, cy2 = box_center(existing)
                if box_distance((cx1, cy1), (cx2, cy2)) < 120:
                    too_close = True
                    break
            
            if not too_close:
                final_persons.append(candidate)
        
        # ==================== Analyze Per-Person PPE Status ====================
        person_statuses = []
        safe_count = 0
        warning_count = 0
        danger_count = 0
        
        for person_box in final_persons:
            px1, py1, px2, py2 = person_box
            person_id = get_person_id(person_box)
            
            # Initialize person history if needed
            if person_id not in person_histories:
                person_histories[person_id] = {
                    'helmet': deque(maxlen=10),
                    'mask': deque(maxlen=10),
                    'gasmask': deque(maxlen=10),
                    'safe': deque(maxlen=10)
                }
            
            hist = person_histories[person_id]
            
            current_helmet = "No Helmet"
            current_mask = "No Mask"
            current_gasmask = "No Gas Mask"
            
            # Match helmet to this person
            for (hx1, hy1, hx2, hy2, hconf) in helmet_hats:
                cx, cy = box_center((hx1, hy1, hx2, hy2))
                if px1 <= cx <= px2 and py1 <= cy <= py1 + (py2 - py1) * 0.55:
                    current_helmet = "Helmet"
                    break
            
            # Match mask to this person
            best_mask_conf = 0
            best_mask_label = "No Mask"
            
            for (mx1, my1, mx2, my2, mcls, mconf) in mask_detections:
                cx, cy = box_center((mx1, my1, mx2, my2))
                if px1 <= cx <= px2 and py1 <= cy <= py1 + (py2 - py1) * 0.75:
                    if mconf > best_mask_conf:
                        best_mask_conf = mconf
                        if mcls == 0:
                            best_mask_label = "With Mask"
                        elif mcls == 1:
                            best_mask_label = "Without Mask"
                        elif mcls == 2:
                            best_mask_label = "Incorrect Mask"
            
            current_mask = best_mask_label
            
            # Match gasmask to this person
            for (gx1, gy1, gx2, gy2, gconf) in gasmask_boxes:
                cx, cy = box_center((gx1, gy1, gx2, gy2))
                if px1 <= cx <= px2 and py1 <= cy <= py2: # Relaxed bounding box check
                    current_gasmask = "Gas Mask"
                    break
            
            # Update history
            hist['helmet'].append(current_helmet)
            hist['mask'].append(current_mask)
            hist['gasmask'].append(current_gasmask)
            
            helmet_status = stable_vote(hist['helmet'], "No Helmet")
            mask_status = stable_vote(hist['mask'], "No Mask")
            gasmask_status = stable_vote(hist['gasmask'], "No Gas Mask")
            
            # Determine overall status logic:
            # - Gas mask: DANGER
            # - Helmet + Mask: SAFE
            # - Helmet only / Mask only / Nothing: WARNING
            if gasmask_status == "Gas Mask":
                current_status = "DANGER"
            elif helmet_status == "Helmet" and mask_status == "With Mask":
                current_status = "SAFE"
            else:
                current_status = "WARNING"
            
            hist['safe'].append(current_status)
            overall_status = stable_vote(hist['safe'], 'WARNING')
            
            if overall_status == 'SAFE':
                color = (0, 255, 0)
                safe_count += 1
            elif overall_status == 'WARNING':
                color = (0, 165, 255) # Orange
                warning_count += 1
            else: # DANGER
                color = (0, 0, 255) # Red
                danger_count += 1
            
            # Draw person box
            cv2.rectangle(output, (px1, py1), (px2, py2), color, 3)
            
            # Draw labels
            label_x = px1
            label_y = max(30, py1 - 80)
            
            cv2.putText(output, overall_status, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            cv2.putText(output, helmet_status, (label_x, label_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(output, mask_status, (label_x, label_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(output, gasmask_status, (label_x, label_y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            person_statuses.append({
                'id': person_id,
                'status': overall_status,
                'helmet': helmet_status,
                'mask': mask_status,
                'gasmask': gasmask_status,
                'color': 'safe' if overall_status == 'SAFE' else ('danger' if overall_status == 'DANGER' else 'warning')
            })
        
        # Draw global status
        total_persons = len(final_persons)
        
        if danger_count > 0:
            global_status = "🚨 DANGER ZONE"
            global_color = (0, 0, 255)
        elif warning_count > 0:
            global_status = "⚠️ WARNING"
            global_color = (0, 165, 255)
        elif safe_count > 0:
            global_status = "✅ SAFE"
            global_color = (0, 255, 0)
        else:
            global_status = "⚠ NO PERSON"
            global_color = (128, 128, 128)
        
        cv2.putText(output, global_status, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, global_color, 3)
        
        stats_text = f"Persons: {total_persons} | Safe: {safe_count} | Warning: {warning_count} | Danger: {danger_count}"
        cv2.putText(output, stats_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        detections = {
            'total_persons': total_persons,
            'safe_count': safe_count,
            'unsafe_count': warning_count + danger_count, # Both are unsafe for charts
            'warning_count': warning_count,
            'danger_count': danger_count,
            'global_status': global_status,
            'people': person_statuses
        }
        
        return output, detections, person_histories


# Singleton instance
inference_service = ModelInferenceService()
