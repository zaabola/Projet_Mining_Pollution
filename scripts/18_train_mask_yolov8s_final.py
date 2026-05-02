from ultralytics import YOLO
import torch
import os
import shutil

def main():
    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    model = YOLO("yolov8s.pt")

    results = model.train(
        data="datasets/mask_yolo_clean/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project="models/yolo",
        name="yolov8s_mask_final",
        exist_ok=True
    )

    os.makedirs("models/final", exist_ok=True)

    possible_paths = [
        "models/yolo/yolov8s_mask_final/weights/best.pt",
        "runs/detect/models/yolo/yolov8s_mask_final/weights/best.pt",
        "runs/detect/yolov8s_mask_final/weights/best.pt"
    ]

    for src in possible_paths:
        if os.path.exists(src):
            shutil.copy(src, "models/final/mask_best_yolov8s.pt")
            print("✅ Final mask model saved!")
            return

    print("⚠️ Copy best.pt manually if needed.")

if __name__ == "__main__":
    main()