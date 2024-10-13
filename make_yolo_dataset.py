import os
import cv2
import numpy as np

# 定义类别映射
class_mapping = {1: 0, 2: 1, 3: 2}


def process_masks(image_dir, mask_dir, label_dir):
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # 遍历掩码文件
    for mask_name in os.listdir(mask_dir):
        if mask_name.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_name)
            image_path = os.path.join(
                image_dir, mask_name.replace('.png', '.jpg'))
            label_path = os.path.join(
                label_dir, mask_name.replace('.png', '.txt'))

            # 读取图像和掩码
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            height, width = mask.shape

            with open(label_path, 'w') as f:
                # 处理每个类别
                for class_id in [1, 2, 3]:
                    # 创建类别掩码
                    class_mask = np.zeros_like(mask)
                    class_mask[mask == class_id] = 255

                    # 提取轮廓
                    contours, _ = cv2.findContours(
                        class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    for contour in contours:
                        if len(contour) >= 3:  # 确保轮廓至少有三个点
                            # 获取轮廓坐标并归一化
                            contour = contour.squeeze()
                            x = contour[:, 0] / width
                            y = contour[:, 1] / height
                            coords = np.vstack((x, y)).T.flatten()
                            coords_str = ' '.join(map(str, coords))

                            # 写入标注文件
                            f.write(
                                f"{class_mapping[class_id]} {coords_str}\n")


process_masks('./data/images/test', './data/annotations/test',
              './yolo_dataset/labels/val')
process_masks('./data/images/training',
              './data/annotations/training', './yolo_dataset/labels/train')
