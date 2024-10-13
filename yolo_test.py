import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO  # 确保安装了ultralytics库

# 设置路径
test_dir = 'data/images/test'
label_dir = 'data/annotations/test'
pred_dir = 'yolo_pred_mask'
os.makedirs(pred_dir, exist_ok=True)

# 加载YOLO模型
model = YOLO('best.pt')

# 遍历测试集中的图像
for img_name in os.listdir(test_dir):
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        img_path = os.path.join(test_dir, img_name)

        # 读取图像
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # 推理并获取结果
        result = model(img)[0]
        pred_mask = np.zeros((height, width), dtype=np.uint8)

        # 获取类别索引和分割掩码
        classes = result.boxes.cls.int().tolist()  # 获取类别信息
        masks = result.masks.xy  # 获取掩码的多边形坐标

        for i in range(len(classes)):
            cls = classes[i] + 1  # 类别对应灰度值1, 2, 3
            mask = masks[i]
            mask = np.array(mask, dtype=np.int32)
            cv2.fillPoly(pred_mask, [mask], cls)

        # 保存结果
        mask_save_path = os.path.join(
            pred_dir, img_name.replace('.jpg', '.png'))
        cv2.imwrite(mask_save_path, pred_mask)
