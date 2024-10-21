"""
图像导入库与格式转换库 \n
与图像处理相关的库主要有2个，现在阐明其区别与联系： \n
1. `cv2`，图像格式为`np.ndarray`，作为向量格式，主要用于计算机视觉中对于图像的高级处理算法 \n
2. `Image`，图像格式为`ImageFile`或`Image`，作为文件格式，主要用于对图像的直接操作，如格式转换等 \n
而图片在各层级上存在如下区别与联系： \n
1. 屏幕显示：坐标原点在左上角，y轴向下，x轴向右 \n
2. `np.ndarray`：RGB图为三维，[H, W, C]，通道顺序为R, G, B；灰度图为二维，[H, W]；索引顺序为先y轴后x轴 \n
3. `torch.tensor`：卷积神经网络的期待输入为[C, H, W]，即使是灰度图也依然需要通道维度 \n
同时，各层级上图片的数值属性也存在区别： \n
1. `np.ndarray`：cv2读入或Image转化后默认为`np.uint8`类型，范围在$[0, 255]$ \n
2. `torch.tensor`：需要与待输入的网络保持格式一致，常用类型为`torch.float32`，范围在$[0, 1]$ \n
"""


import typing
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt


def import_image(path: str) -> ImageFile.ImageFile:
    """导入图像"""
    image = Image.open(path)
    return image


def show_image(image: ImageFile.ImageFile | Image.Image | np.ndarray, cmap=None) -> None:
    """
    显示图像； \n
    这里说明显示图像的三种方法
    plt.imshow可以接收Image(即ImageFile)和np.ndarray两种格式的输入，主要是用于显示数据图而非原始图像的；
    这是因为plt.imshow
    如果输入的是灰度图，且希望以灰度方式显示，则需要指定参数cmap='gray'，
    否则会以其它颜色映射方式显示(默认为'viridis'，蓝色到绿色再到黄色的渐变) \n
    """
    plt.imshow(image, cmap=cmap)
    plt.show()
    return


def save_image(image: ImageFile.ImageFile, path: str) -> None:
    """保存图像"""
    image.save(path)
    return


def image_to_gray(image: ImageFile.ImageFile) -> ImageFile.ImageFile:
    """将图片转换为灰度图"""
    # 灰度值 = 0.299 * R + 0.587 * G + 0.114 * B
    gray_image = image.convert('L')
    return gray_image


def draw_histogram(histogram, save: None | str = None, show: bool = True) -> None:
    """保存或显示直方图"""
    if save is not None or show is True:
        plt.figure(figsize=(10, 5))
        plt.bar(range(256), histogram, color='blue')
        plt.title('Grayscale Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        if save is not None:
            plt.savefig(save)
        if show is True:
            plt.show()


def draw_cdf(raw_cdf: typing.Sequence, fla_cdf: typing.Sequence, ide_cdf: typing.Sequence,
             show: bool = True, save: None | str = None) -> None:
    """保存或显示cdf"""
    if save is not None or show is True:
        plt.figure()
        plt.plot(range(256), raw_cdf, label='raw cdf', color='b')
        plt.plot(range(256), fla_cdf, label='flatten cdf', color='r')
        plt.plot(range(256), ide_cdf, label='ideal cdf', color='g')
        plt.title('CDF Comparison')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        if save is not None:
            plt.savefig(save)
        if show is True:
            plt.show()
