import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from typing import Callable


def transform_curve(image: ImageFile.ImageFile, transform: Callable) -> ImageFile.ImageFile:
    """对图像强度进行整体变换"""
    # 确保图片是灰度图
    gray_image = image.convert('L')
    image_t = Image.eval(gray_image, transform)
    return image_t
