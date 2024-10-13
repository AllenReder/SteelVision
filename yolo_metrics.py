import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO  # 确保安装了ultralytics库
import time

# 设置路径
test_dir = 'data/images/test'
label_dir = 'data/annotations/test'
pred_dir = 'yolo_pred_mask'

# 计算每个类别的IoU和mIoU


start_time = time.time()
TP = {1: 0, 2: 0, 3: 0}
FP = {1: 0, 2: 0, 3: 0}
FN = {1: 0, 2: 0, 3: 0}

for img_name in os.listdir(test_dir):
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        # 读取预测掩码和真实掩码
        pred_mask_path = os.path.join(
            pred_dir, img_name.replace('.jpg', '.png'))
        label_mask_path = os.path.join(
            label_dir, img_name.replace('.jpg', '.png'))

        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        label_mask = cv2.imread(label_mask_path, cv2.IMREAD_GRAYSCALE)

        # 确保 pred_mask 是一个 NumPy 数组
        pred_mask = np.array(pred_mask)

        for cls in range(1, 4):
            TP[cls] += np.sum((pred_mask == cls) & (label_mask == cls))
            FP[cls] += np.sum((pred_mask == cls) & (label_mask != cls))
            FN[cls] += np.sum((pred_mask != cls) & (label_mask == cls))

end_time = time.time()
fps = len(os.listdir(test_dir)) / (end_time - start_time)

iou = {1: 0, 2: 0, 3: 0}
# 计算 mIoU
for c in range(1, 4):
    iou[c] = TP[c] / (TP[c] + FP[c] + FN[c])
    print(f"Class {c} IoU: {iou[c]}")
mIoU = sum(iou.values()) / len(iou)
print(f"Mean IoU (mIoU): {mIoU:.4f}")
print(f"Frames Per Second (FPS): {fps:.2f}")
