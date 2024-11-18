import os
import cv2
from rembg import remove 
from PIL import Image

input_dir = './tmp/images'
output_dir = './tmp/remove-bg'


# for file in os.listdir(input_dir):
#     img = cv2.imread(os.path.join(input_dir, file), 1)
#     tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
#     b, g, r = cv2.split(img)
#     rgba = [b, g, r, alpha]
#     dst = cv2.merge(rgba, 4)
#     cv2.imwrite(os.path.join(output_dir, file), dst)

for file in os.listdir(input_dir):
    input = Image.open(os.path.join(input_dir, file)) 
    # Removing the background from the given Image 
    output = remove(input) 
    output = output.convert("RGB")
    #Saving the image in the given path 
    output.save(os.path.join(output_dir, file)) 




