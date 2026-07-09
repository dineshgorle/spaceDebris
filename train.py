from ultralytics import YOLO

# 👉 FIRST define model
model = YOLO("yolov8n.pt")

# 👉 THEN train
model.train(
    data="C:/Users/DINESH/Desktop/yolo2/debris_dataset/data.yaml",
    epochs=20,
    imgsz=320,
    batch=2,
    device="cpu"
)