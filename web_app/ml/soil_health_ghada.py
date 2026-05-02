import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp

# =========================
# DEVICE & MODEL
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = torch.cuda.is_available()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "best_unet_ghada.pth")

SOIL_HEALTH_MODEL_LOADED = False
model = None

try:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    # weights_only=False needed because standard PyTorch save format with classes might require it, but we can try True first
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    # Strip _orig_mod. prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"[OK] Soil Health (Ghada) model loaded successfully from {MODEL_PATH}")
    SOIL_HEALTH_MODEL_LOADED = True
except Exception as e:
    print(f"[ERROR] Error loading Soil Health model: {e}")
    SOIL_HEALTH_MODEL_LOADED = False
    model = None

# =========================
# PREPROCESSING
# =========================
def normalize_image(image):
    """Normalize grayscale image using values from soil detector training."""
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.0], dtype=np.float32)  
    std = np.array([0.1], dtype=np.float32)   
    return (image - mean) / std

def create_overlay(image_np, mask_binary):
    """Create a red overlay for detected mining areas."""
    overlay = image_np.copy()
    red_mask = np.zeros_like(image_np)
    red_mask[mask_binary == 1] = [0, 0, 255] # BGR
    
    # Apply alpha blending
    cv2.addWeighted(red_mask, 0.5, overlay, 1.0, 0, overlay)
    return overlay

# =========================
# SEGMENTATION
# =========================
def segment_soil_health(image_data, tile_size=512):
    """
    Segment image using the Ghada UNet model.
    """
    if not SOIL_HEALTH_MODEL_LOADED or model is None:
        raise RuntimeError("Soil Health model is not loaded.")
    
    # Convert bytes to numpy if needed
    if isinstance(image_data, bytes):
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    else:
        image_np = image_data
    
    # Convert to Grayscale
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    h, w = image_gray.shape[:2]
    
    # Resize image to be multiple of 32 for UNet
    resized_gray = cv2.resize(image_gray, (512, 512))
    
    # Expand dims to make it (512, 512, 1) before transpose
    resized_gray_exp = np.expand_dims(resized_gray, axis=-1)
    
    normalized = normalize_image(resized_gray_exp).transpose(2, 0, 1)
    input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION):
            output = model(input_tensor)
            prob = torch.sigmoid(output)
            pred_mask = (prob > 0.5).float().cpu().numpy()[0, 0] # shape (512, 512)
    
    # Resize mask back to original resolution
    full_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    full_mask = full_mask.astype(np.uint8)
    
    # Create colored visualization
    h_orig, w_orig = full_mask.shape
    colored_mask = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    colored_mask[full_mask == 1] = [0, 0, 255] # Red BGR
    
    # Create overlay
    overlay = create_overlay(image_np, full_mask)
    
    # Calculate area percentages
    total_pixels = h_orig * w_orig
    mining_pixels = (full_mask == 1).sum()
    mining_percent = (mining_pixels / total_pixels) * 100
    
    return {
        "mask": full_mask,
        "colored_mask": colored_mask,
        "overlay": overlay,
        "mining_percent": float(mining_percent),
        "mining_pixels": int(mining_pixels),
        "height": h_orig,
        "width": w_orig
    }
