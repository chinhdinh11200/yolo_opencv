from ultralytics import YOLO

import cv2
import numpy as np
import time

model_path = '/home/chinh/tuto-2024-06/opencv/yolo/runs/segment/train_epoch/weights/last.pt'
# image_path = '/home/chinh/tuto-2024-06/opencv/yolo/data/test/14005.jpg'
image_path = '/home/chinh/tuto-2024-06/opencv/yolo/data/test/image-20241107-074303.png'
# 

ts = time.time();
print('start', ts)
img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

boxes = results[0].boxes.xyxy.tolist()[0]
print('boxes', tuple(boxes))
for result in results:
    for j, mask in enumerate(result.masks.data):
        mask = mask.cpu().numpy() * 255

        mask = cv2.resize(mask, (W, H)) #.astype(np.uint8)
        # cv2.imwrite(f'./results/output_mask_{j}.png', mask)
        # _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

        # rgba_image = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        # rgba_image[:, :, 3] = binary_mask

        # fgModel = np.zeros((1, 65), dtype="float")
        # bgModel = np.zeros((1, 65), dtype="float")
        # (mask, bgModel, fgModel) = cv2.grabCut(img, mask, (187, 0, 369, 238), bgModel,
        #     fgModel, 2, mode=cv2.GC_INIT_WITH_RECT)

        cv2.imwrite(f'./results/output_mask_{j}.png', mask)
        mask = cv2.imread(f'./results/output_mask_{j}.png', 0);
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        result = cv2.bitwise_and(img, img, mask=binary_mask)
        cv2.imwrite(f'./results/output_img{j}.png', result)

# mask = cv2.imread('./output1.png', 0);
# _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# result = cv2.bitwise_and(img, img, mask=binary_mask)

# # Lưu ảnh kết quả hoặc hiển thị
# cv2.imwrite('extracted_person7.jpg', result)

ts = time.time();
print('end', ts)
