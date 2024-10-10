import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device():
    return device


def show_images(images=[], titles=[], save_path=None, convert=True, show=True):
    plt.figure(figsize=(len(images) * 5, 5))
    for i in range(len(images)):
        if convert and titles[i] != 'image':
            # 将灰度1,2,3映射到RGB色彩
            color_map = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}  # 红色、绿色、蓝色
            colored_mask = np.zeros((*images[i].shape, 3), dtype=np.float32)
            for gray_value, rgb in color_map.items():
                colored_mask[images[i] == gray_value] = rgb
            images[i] = colored_mask

        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    if save_path:
        plt.savefig(save_path)  # 保存拼接好的窗口图片
    if show:
        plt.show(block=True)  # 确保显示窗口保持打开状态
