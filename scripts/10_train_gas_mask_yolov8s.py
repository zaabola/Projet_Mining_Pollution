from ultralytics import YOLO
import torch
import os
import shutil

def main():
    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    model = YOLO("yolov8s.pt")

    results = model.train(
        data="datasets/gas_mask_dataset/data.yaml",
        epochs=10,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project="models/yolo",
        name="yolov8s_gas_mask",
        exist_ok=True
    )

    # Save clean final copy
    os.makedirs("models/final", exist_ok=True)

    possible_paths = [
        "models/yolo/yolov8s_gas_mask/weights/best.pt",
        "runs/detect/models/yolo/yolov8s_gas_mask/weights/best.pt",
        "runs/detect/yolov8s_gas_mask/weights/best.pt"
    ]

    for src in possible_paths:
        if os.path.exists(src):
            dst = "models/final/gas_mask_best_yolov8s.pt"
            shutil.copy(src, dst)
            print(f"✅ Final gas mask model copied to: {dst}")
            return

    print("⚠️ best.pt was not found automatically. Please copy it manually.")

if __name__ == "__main__":
    main()