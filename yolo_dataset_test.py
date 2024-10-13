import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
# 定义颜色映射
color_map = {
    0: (255, 0, 0),  # 类别1 - 红色
    1: (0, 255, 0),  # 类别2 - 绿色
    2: (0, 0, 255)   # 类别3 - 蓝色
}


def draw_masks(image_path, label_path):
    print(image_path)
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 读取标签文件
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # 绘制掩码
    for label in labels:
        parts = label.strip().split()
        category = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
        points = list(map(float, parts[5:]))

        # 计算边界框
        img_h, img_w = image.shape[:2]
        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h

        # 计算多边形顶点
        polygon = [(points[i] * img_w, points[i + 1] * img_h)
                   for i in range(0, len(points), 2)]

        # 绘制多边形
        cv2.polylines(image, [np.array(polygon, np.int32)],
                      isClosed=True, color=color_map[category], thickness=1)

    # 显示图像
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# 示例使用
image_folder = './yolo_dataset/images/val'
label_folder = './yolo_dataset/labels/val'

for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        label_path = os.path.join(
            label_folder, filename.replace('.jpg', '.txt'))
        draw_masks(image_path, label_path)
