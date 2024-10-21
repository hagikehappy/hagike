"""
图像直方图
"""


import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from .file import draw_cdf


def convert_histogram(image: ImageFile.ImageFile) -> np.ndarray:
    """提取图片各通道的灰度直方图"""
    # 确保图片是灰度图
    gray_image = image.convert('L')
    # 计算灰度直方图
    histogram = np.array(gray_image.histogram()).astype('float')
    # 标准化直方图
    num_pixels = gray_image.size[0] * gray_image.size[1]
    histogram_nm = histogram / num_pixels
    return histogram_nm


def flatten_histogram(image: ImageFile.ImageFile) -> tuple:
    """直方图均衡化"""
    # 初始化结构
    quan_size = 256
    raw_his = convert_histogram(image)
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

    image_np = np.array(image)
    for i in range(image_np.shape[0]):
        for j in range(image_np.shape[1]):
            image_np[i, j] = map_sheet[image_np[i, j]]
    image_fla = Image.fromarray(image_np)

    return image_fla, raw_his, fla_his, raw_cdf, fla_cdf, ideal_cdf










