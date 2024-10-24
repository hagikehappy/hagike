"""
图像直方图
"""


from .file import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import typing


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


def convert_histogram(im: ImStd) -> np.ndarray:
    """提取图片各通道的灰度直方图"""
    # 确保图片是灰度图
    im = im.to_gray().to_style(ImStyle.im_file)
    image = im.image
    histogram = np.array(image).astype('float')
    # 标准化直方图
    num_pixels = image.size[0] * image.size[1]
    histogram_nm = histogram / num_pixels
    return histogram_nm


def flatten_histogram(im: ImStd) -> tuple:
    """直方图均衡化"""
    # 初始化结构
    quan_size = 256
    raw_his = convert_histogram(im)
    ideal_cdf = np.linspace(0, 1, quan_size, dtype=float)
    raw_cdf = np.zeros(quan_size, dtype=float)
    fla_cdf = np.zeros(quan_size, dtype=float)
    map_sheet = np.zeros(quan_size, dtype=int)
    fla_his = np.zeros(quan_size)

    # 建立raw_cdf
    raw_cdf[0] = raw_his[0]
    for i in range(1, quan_size):
        raw_cdf[i] = raw_his[i] + raw_cdf[i - 1]

    # 遍历raw_his来产生fla_his
    j = 0   # j是fla_cdf指针
    for i in range(0, quan_size):
        # 处理饱和情况
        if j == quan_size:
            fla_his[j] += raw_his[i]
            map_sheet[i] = j
            fla_cdf[j] += raw_his[i]
            continue
        # 处理零情况
        if raw_his[i] == 0.0:
            if i >= 1:
                map_sheet[i] = map_sheet[i - 1]
            continue
        # 处理一般情况
        tmp_cdf = fla_cdf[j] + raw_his[i]
        n = j
        for n in range(j, quan_size):
            # 如果此时比理想值小，那么这就是映射的目标点，设置该点cdf和映射值，并退出循环
            # 这里使用的是优先填充cdf的策略
            # 第二个条件检查饱和情况（防止浮点数精度导致的误差）
            if tmp_cdf <= ideal_cdf[n] or n == quan_size - 1:
                fla_his[n] = raw_his[i]
                map_sheet[i] = n
                fla_cdf[n] = tmp_cdf
                break
            # 如果此时比理想值大，那么应继续寻找映射目标点，同时设置该点cdf
            else:
                fla_cdf[n] = fla_cdf[j]
        # 更新fla_cdf指针
        j = n

    im = im.to_style(ImStyle.im_ndarray).to_gray()
    image = im.image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = map_sheet[image[i, j]]
    image_fla = Image.fromarray(image)

    return image_fla, raw_his, fla_his, raw_cdf, fla_cdf, ideal_cdf










