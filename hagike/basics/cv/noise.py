"""
噪声添加
"""


import cv2
import numpy as np
from ...utils import *


@advanced_enum()
class NoiseType(SuperEnum):
    """噪声类型"""

    class gaussian(SuperEnum):
        """高斯噪声"""
        mean = 0
        var = 0.01

    class salt_pepper(SuperEnum):
        """椒盐噪声"""
        prob = 0.02

    class poisson(SuperEnum):
        """泊松噪声"""
        lam = 255


def add_noise_to_image(image, noise_type='gaussian', **kwargs):
    """
    向图像添加不同类型的噪声。

    :param image: 输入图像（numpy数组）。
    :param noise_type: 噪声类型，'gaussian'、'salt_and_pepper' 或 'poisson'。
    :param kwargs: 其他参数，例如高斯噪声的均值和方差，椒盐噪声的概率等。
    :return: 带噪声的图像。
    """
    if noise_type == 'gaussian':
        mean = kwargs.get('mean', 0)
        var = kwargs.get('var', 0.01)  # 方差
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, image.shape).astype('float32')
        gaussian = np.clip(gaussian * 255, 0, 255).astype('uint8')
        noisy = cv2.addWeighted(image, 1, gaussian, 1, 0)
    elif noise_type == 'salt_and_pepper':
        prob = kwargs.get('prob', 0.02)
        salt_pepper = np.random.rand(*image.shape) < prob
        salt_pepper = np.where(salt_pepper, 255, 0).astype('uint8')
        noisy = cv2.addWeighted(image, 1 - prob, salt_pepper, prob, 0)
    elif noise_type == 'poisson':
        lam = kwargs.get('lam', 255)  # λ为图像的平均亮度
        poisson = np.random.poisson(lam)
        noisy = cv2.poissonNoise(image, lam)

    else:
        raise ValueError("Unsupported noise type")
    return noisy


