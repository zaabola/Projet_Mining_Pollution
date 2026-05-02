from ultralytics import YOLO
import torch

def main():
    print("GPU available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    model = YOLO("yolov8s.pt")

    results = model.train(
        data="datasets/fish_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,          # 🔥 SAFE for 3050 Ti laptop
        device=0,
        workers=2,        # if crash → set to 0
        amp=True,         # 🔥 IMPORTANT (mixed precision = faster + less VRAM)
        cache=False,      # avoid RAM overload
        project="models/yolo",
        name="yolov8s_fish_base",
        exist_ok=True
    )

if __name__ == "__main__":
    main()