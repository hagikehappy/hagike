"""
噪声添加
"""

import cv2
import numpy as np
from ...utils import *
from .file import *
from typing import Mapping, Any


class ImageNoiseError(Exception):
    """图像噪声异常"""

    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


@advanced_enum()
class ImNoise(SuperEnum):
    """
    噪声类型 \n
    .. important::
        参数是在图像归一化情况下的
    """

    class gaussian(SuperEnum):
        """高斯噪声"""
        mean: uuid_t = 0
        """均值"""
        var: uuid_t = 0.01
        """方差"""

    class salt_pepper(SuperEnum):
        """椒盐噪声"""
        prob_salt: uuid_t = 0.02
        """白色块概率"""
        prob_pepper: uuid_t = 0.02
        """黑色块概率"""

    class poisson(SuperEnum):
        """泊松噪声"""
        lam: uuid_t = 0.1
        """λ"""


def add_noise_to_image(image: cv_ndarray, noise: uuid_t, para: Mapping[uuid_t, Any] | None = None):
    """
    向图像添加指定类型的噪声 \n

    """
    ImNoise.check_in_(noise)
    noise_type = ImNoise.get_cls_(noise)
    noise_type.dict_(para, is_force=True)
    # 将图像转换到[0, 1]之间
    if noise == ImNoise.gaussian:
        sigma = para[ImNoise.gaussian.var] ** 0.5
        gaussian = np.random.normal(para[ImNoise.gaussian.mean], sigma, image.shape).astype('float32')
        gaussian = np.clip(gaussian * 255, 0, 255).astype('uint8')
        noisy = cv2.addWeighted(image, 1, gaussian, 1, 0)
    elif noise == ImNoise.salt_pepper:
        salt = np.random.rand(*image.shape) < para[ImNoise.salt_pepper.prob_salt]
        image[salt] = 255
        pepper = np.random.rand(*image.shape) < para[ImNoise.salt_pepper.prob_pepper]
        image[pepper] = 0
        noisy = image
    elif noise == ImNoise.poisson:
        poisson = np.random.poisson(para[ImNoise.poisson.lam], image.shape).astype('float32')
        noisy_image = np.clip(noisy_image, 0, 1)
    else:
        raise ImageNoiseError(
            f"ERROR: Noise {ImStyle.get_value_(noise)} is not implemented a 'to' function!!!")
    return noisy
