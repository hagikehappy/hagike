"""
图像变换
"""


from .file.image.im import *
from PIL import Image
from typing import Callable


def transform_curve(im: ImStd, transform: Callable) -> im_file:
    """对图像强度进行整体变换"""
    # 确保图片是灰度图
    im = im.to_color(ImColor.gray).to_style(ImStyle.im_file)
    image = im.image
    image_t = Image.eval(image, transform)
    return image_t
