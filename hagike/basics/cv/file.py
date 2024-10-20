import typing

import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt


def import_image(path: str) -> ImageFile.ImageFile:
    """导入图像"""
    image = Image.open(path)
    return image


def show_image(image: ImageFile.ImageFile, cmap=None) -> None:
    """显示图像"""
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


def draw_histogram(histogram, save: None | str = None, show: bool = True):
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
             show: bool = True, save: None | str = None):
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
