import os
import cv2
import torch
import torch.nn as nn
import numpy as np


# =========================
# DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# ARCHITECTURE EXACTE DU MODELE
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.down1 = DoubleConv(in_channels, 64, dropout=0.0)
        self.down2 = DoubleConv(64, 128, dropout=0.05)
        self.down3 = DoubleConv(128, 256, dropout=0.10)
        self.down4 = DoubleConv(256, 512, dropout=0.20)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256, dropout=0.10)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128, dropout=0.05)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64, dropout=0.0)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        return self.final(x)


# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deforestation_unet_deploy.pth")

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    BEST_THRESHOLD = float(checkpoint["best_threshold"])
    IMG_SIZE = int(checkpoint["img_size"])
    
    print(f"[OK] Deforestation model loaded successfully from {MODEL_PATH}")
    MODEL_LOADED = True
    
except Exception as e:
    print(f"[ERROR] Error loading deforestation model: {e}")
    MODEL_LOADED = False
    BEST_THRESHOLD = 0.5
    IMG_SIZE = 256
    model = None


# =========================
# PREPROCESSING
# =========================
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor.to(DEVICE)

    return tensor


# =========================
# PREDICTION MASK FORET
# =========================
def predict_forest_mask(image):
    tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output)[0, 0].cpu().numpy()
        mask = (prob > BEST_THRESHOLD).astype(np.uint8)

    return mask, prob


# =========================
# DETECTION DEFORESTATION
# =========================
def detect_deforestation(image_old, image_new):
    if not MODEL_LOADED or model is None:
        raise RuntimeError("Deforestation model is not loaded. Check the model file.")
    
    forest_old, prob_old = predict_forest_mask(image_old)
    forest_new, prob_new = predict_forest_mask(image_new)

    deforestation_mask = ((forest_old == 1) & (forest_new == 0)).astype(np.uint8)

    forest_before = forest_old.sum()
    forest_lost = deforestation_mask.sum()

    if forest_before > 0:
        percent = (forest_lost / forest_before) * 100
    else:
        percent = 0.0

    return {
        "forest_old": forest_old,
        "forest_new": forest_new,
        "deforestation_mask": deforestation_mask,
        "percent": float(percent),
    }