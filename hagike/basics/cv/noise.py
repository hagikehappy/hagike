"""
噪声添加
"""

import cv2
import numpy as np
from hagike.utils import *
from .file.image import *
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
        scale: uuid_t = 0.12
        """
        泊松分布是一种离散概率分布，用于预测在固定时间或空间内发生某事件的平均次数。 \n
        在图像处理中，泊松噪声通常用于模拟光子到达传感器的随机性，特别是在低光照条件下。 \n
        将图像的每个像素值除以一个尺度参数 `scale`，调整图像数据的“亮度”，以便与泊松分布的参数相匹配。 \n
        `scale` 参数在这里起到了调节噪声强度的作用，较小的 `scale` 值会导致更强烈的噪声 \n
        当 `scale` 足够大时，趋向于椒盐噪声中白点的效果果，且随着 `scale` 继续增大，噪声效果变小 \n
        因为每个像素值被“缩放”得更小，然后泊松分布会在这个更小的值上产生更大的随机变化。 \n
        最后将上一步生成的泊松随机数乘以scale，缩放回原尺度。 \n
        """
        alpha: uuid_t = 0.3
        """
        附加比例因子，用于控制噪声相比原图的占比
        """


def add_noise_to_image(im: ImStd, noise: uuid_t, para: Mapping[uuid_t, Any] | None = None) -> ImStd:
    """
    向图像添加指定类型的噪声
    """
    ImNoise.check_in_(noise)
    noise_type = ImNoise.get_cls_(noise)
    para = noise_type.dict_(para, is_force=True)
    # 保存旧值，这里需要指定 `is_new=True` 来确保操作过程对输入项没有影响
    scale = im.scale
    im = im.to_scale(ImScale.float_n, is_new=True)
    image = im.image
    # 将图像转换到[0, 1]之间
    if noise == ImNoise.gaussian.get_uuid_():
        sigma = para[ImNoise.gaussian.var] ** 0.5
        gaussian = np.random.normal(para[ImNoise.gaussian.mean], sigma, image.shape).astype(np.float32)
        noisy = np.clip(image + gaussian, 0.0, 1.0)
    elif noise == ImNoise.salt_pepper.get_uuid_():
        rand_mask = np.random.rand(*image.shape)
        salt = rand_mask < para[ImNoise.salt_pepper.prob_salt]
        pepper = rand_mask > 1 - para[ImNoise.salt_pepper.prob_pepper]
        image[salt] = 1.0
        image[pepper] = 0.0
        noisy = image
    elif noise == ImNoise.poisson.get_uuid_():
        lam = para[ImNoise.poisson.scale]
        poisson = np.random.poisson(image / lam).astype(np.float32) * lam * para[ImNoise.poisson.alpha]
        noisy = np.clip(image + poisson, 0.0, 1.0)
    else:
        raise ImageNoiseError(
            f"Noise {ImStyle.get_name_(noise)} is not implemented a noise function!!!")
    # 还原为输入时的数值范围
    im.image = noisy
    im.to_scale_(scale)
    return im
