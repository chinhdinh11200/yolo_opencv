from ultralytics import YOLO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

model = YOLO('yolo11s-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='config.yaml', epochs=1, imgsz=640, name="train_epoch")

# model.va
