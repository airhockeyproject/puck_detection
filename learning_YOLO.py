from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="test1-dprxq-1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="results",
    name="test1_run"
)
