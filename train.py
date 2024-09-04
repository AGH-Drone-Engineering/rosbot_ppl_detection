from ultralytics import YOLO

model = YOLO("yolov9c.pt")

results = model.train(data="data.yaml", epochs=100, imgsz=640)

model.save("model.pt")
