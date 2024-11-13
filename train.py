from ultralytics import YOLO

model = YOLO('yolo11s-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='config.yaml', epochs=1, imgsz=640)
