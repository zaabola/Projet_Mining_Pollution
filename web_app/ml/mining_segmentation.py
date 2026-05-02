import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# =========================
# DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = torch.cuda.is_available()

# =========================
# CLASS LABELS
# =========================
CLASS_LABELS = {
    'background': 0,
    'ressite': 1,      # Residential site - Green
    'msite': 2         # Mining site - Red
}

CLASS_COLORS = {
    0: [0, 0, 0],           # Background - Black
    1: [0, 255, 0],         # Residential - Green
    2: [255, 0, 0]          # Mining - Red
}

# =========================
# ARCHITECTURE
# =========================
class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], dropout=0.3):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f

        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], features[-1] * 2),
            nn.Dropout2d(dropout)
        )

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[-(idx // 2 + 1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x)


# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iyed_best.pth")

try:
    model = UNet(in_channels=3, out_channels=3, features=[64, 128, 256, 512])
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    # Strip _orig_mod. prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"[OK] Mining segmentation model loaded successfully from {MODEL_PATH}")
    MINING_MODEL_LOADED = True
except Exception as e:
    print(f"[ERROR] Error loading mining segmentation model: {e}")
    MINING_MODEL_LOADED = False
    model = None


# =========================
# PREPROCESSING
# =========================
def normalize_image(image):
    """Normalize image using ImageNet statistics."""
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean) / std


def create_colored_mask(mask):
    """Convert class indices mask to colored visualization."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in CLASS_COLORS.items():
        colored[mask == class_idx] = color
    
    return colored


# =========================
# SEGMENTATION
# =========================
def segment_image(image_data, tile_size=512):
    """
    Segment image using tiling at full resolution with batched inference.
    
    Processes tiles in batches for GPU efficiency while maintaining
    full-resolution accuracy.
    
    Args:
        image_data: numpy array or bytes (image data)
        tile_size: size of tiles for processing
        
    Returns:
        dict with segmentation results
    """
    if not MINING_MODEL_LOADED or model is None:
        raise RuntimeError("Mining segmentation model is not loaded.")
    
    # Convert bytes to numpy if needed
    if isinstance(image_data, bytes):
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    else:
        image_np = image_data
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    # --- Collect all tile patches ---
    tiles = []  # (y0, x0, patch)
    stride = tile_size
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = min(y + tile_size, h)
            x1 = min(x + tile_size, w)
            y0 = max(0, y1 - tile_size)
            x0 = max(0, x1 - tile_size)
            
            patch = image_rgb[y0:y1, x0:x1]
            
            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                continue
            
            tiles.append((y0, x0, patch))
    
    # --- Process tiles in batches for GPU efficiency ---
    full_mask = np.zeros((h, w), dtype=np.uint8)
    batch_size = 4  # Process 4 tiles at once
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            
            # Build batch tensor
            batch_tensors = []
            for _, _, patch in batch_tiles:
                normalized = normalize_image(patch).transpose(2, 0, 1)
                batch_tensors.append(normalized)
            
            batch_input = torch.tensor(
                np.stack(batch_tensors), dtype=torch.float32
            ).to(DEVICE)
            
            # Run inference on entire batch
            with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION):
                batch_output = model(batch_input)
            
            # Extract predictions
            batch_preds = batch_output.argmax(1).cpu().numpy().astype(np.uint8)
            
            for j, (y0, x0, _) in enumerate(batch_tiles):
                full_mask[y0:y0 + tile_size, x0:x0 + tile_size] = batch_preds[j]
    
    # Create colored visualization
    colored_mask = create_colored_mask(full_mask)
    
    # Create overlay
    overlay = cv2.addWeighted(image_np, 0.7, colored_mask, 0.3, 0)
    
    # Calculate area percentages
    total_pixels = h * w
    ressite_pixels = (full_mask == CLASS_LABELS['ressite']).sum()
    msite_pixels = (full_mask == CLASS_LABELS['msite']).sum()
    
    ressite_percent = (ressite_pixels / total_pixels) * 100
    msite_percent = (msite_pixels / total_pixels) * 100
    
    return {
        "mask": full_mask,
        "colored_mask": colored_mask,
        "overlay": overlay,
        "ressite_percent": float(ressite_percent),
        "msite_percent": float(msite_percent),
        "ressite_pixels": int(ressite_pixels),
        "msite_pixels": int(msite_pixels),
        "height": h,
        "width": w
    }

