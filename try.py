import cv2
import numpy as np
import os

input_dir = 'data/annotations/test'
output_dir = 'data/annotations/test_modified'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

kernel = np.ones((3, 3), np.uint8)  # 定义结构元素

for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 随机选择腐蚀或膨胀
        if np.random.rand() > 0.5:
            modified_img = cv2.erode(img, kernel, iterations=1)
        else:
            modified_img = cv2.dilate(img, kernel, iterations=1)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, modified_img)
