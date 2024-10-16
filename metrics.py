import os
import cv2
import numpy as np

# 设置路径
label_dir = 'data/annotations/test'
pred_dir = 'data/annotations/test_modified'

# 计算每个类别的IoU和mIoU
inter = {1: 0, 2: 0, 3: 0}
union = {1: 0, 2: 0, 3: 0}

for img_name in os.listdir(label_dir):
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
            inter[cls] += np.sum((pred_mask == cls) & (label_mask == cls))
            union[cls] += np.sum((pred_mask == cls) | (label_mask == cls))


iou = {1: 0, 2: 0, 3: 0}
# 计算 mIoU
for c in range(1, 4):
    iou[c] = inter[c] / union[c]
    print(f"Class {c} IoU: {iou[c]}")
mIoU = sum(iou.values()) / len(iou)
print(f"Mean IoU (mIoU): {mIoU:.4f}")
