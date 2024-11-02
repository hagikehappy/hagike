"""
图像变换
"""


from .file.image.im import *
from PIL import Image
from typing import Callable
import math


def transform_curve(im: ImStd, transform: Callable) -> ImStd:
    """对图像强度进行整体变换，仅可用于 `im_file`，本地转换"""
    im.image = Image.eval(im.image, transform)
    return im


def im_log(im: ImStd) -> ImStd:
    """对数变换，仅可用于 uint8，np，本地转换"""
    scale = 255 / math.log1p(255)
    im.image = np.log1p(im.image) * 255 / math.log1p(255)
    return im
