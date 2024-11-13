from ultralytics import YOLO

import cv2
import numpy as np
import time

model_path = '/home/chinh/tuto-2024-06/opencv/yolo/runs/segment/train5/weights/last.pt'
image_path = '/home/chinh/tuto-2024-06/opencv/yolo/data/test/linhnd-cropped.png'

ts = time.time();
print('start', ts)
img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

boxes = results[0].boxes.xyxy.tolist()

for result in results:
    for j, mask in enumerate(result.masks.data):

        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./output1.png', mask)

mask = cv2.imread('./output1.png', 0);
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

result = cv2.bitwise_and(img, img, mask=binary_mask)

# Lưu ảnh kết quả hoặc hiển thị
cv2.imwrite('extracted_person6.jpg', result)

ts = time.time();
print('end', ts)
