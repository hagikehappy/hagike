"""
***图像滤波*** \n
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from .file.image import *
from hagike.utils import *


class ImageFilterError(Exception):
    """图像滤波器异常"""

    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


@advanced_enum()
class ImSmooth(SuperEnum):
    """平滑滤波"""

    class mean(SuperEnum):
        """均值滤波"""
        ksize: uuid_t = (5, 5)

    class gaussian(SuperEnum):
        """高斯滤波"""
        sigma: uuid_t = 1


def apply_smoothing_filter(im: ImStd, smooth: uuid_t, para: Mapping[uuid_t, Any] | None = None) -> ImStd:
    """对图像应用平滑滤波"""
    ImSmooth.check_in_(smooth)
    smooth_type = ImSmooth.get_cls_(smooth)
    para = smooth_type.dict_(para, is_force=True)
    style, scale = im.style, im.scale
    im = im.to_style(ImStyle.cv_ndarray, is_new=True)
    im.to_scale_(ImScale.float_n)
    image = im.image

    if smooth == ImSmooth.mean.get_uuid_():
        image = cv2.blur(image, para[ImSmooth.mean.ksize])
    elif smooth == ImSmooth.gaussian.get_uuid_():
        image = cv2.GaussianBlur(image, (0, 0), para[ImSmooth.gaussian.sigma])
    else:
        raise ImageFilterError(
            f"Filter {ImSmooth.get_name_(smooth)} is not implemented a filter function!!!")

    im.image = np.clip(image, 0.0, 1.0)
    im.to_style_(style), im.to_scale_(scale)
    return im


@advanced_enum()
class ImSharp(SuperEnum):
    """锐化滤波"""

    class laplacian(SuperEnum):
        """拉普拉斯滤波"""
        ksize: uuid_t = 5

    class sharpening(SuperEnum):
        """锐化滤波"""
        alpha: uuid_t = 1.5
        beta: uuid_t = -0.5


def apply_sharpening_filter(im: ImStd, sharp: uuid_t, para: Mapping[uuid_t, Any] | None = None) -> ImStd:
    """对图像应用锐化滤波"""
    ImSharp.check_in_(sharp)
    sharp_type = ImSharp.get_cls_(sharp)
    para = sharp_type.dict_(para, is_force=True)
    style, scale = im.style, im.scale
    im = im.to_style(ImStyle.cv_ndarray, is_new=True)
    im.to_scale_(ImScale.float_n)
    image = im.image

    if sharp == ImSharp.laplacian.get_uuid_():
        laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=para[ImSharp.laplacian.ksize])
        image = image - laplacian
    elif sharp == ImSharp.sharpening.get_uuid_():
        image = para[ImSharp.sharpening.alpha] * image + para[ImSharp.sharpening.beta]
    else:
        raise ImageFilterError(
            f"Filter {ImSharp.get_name_(sharp)} is not implemented a filter function!!!")

    im.image = np.clip(image, 0.0, 1.0)
    im.to_style_(style), im.to_scale_(scale)
    return im


@advanced_enum()
class ImFreq(SuperEnum):
    """频域滤波"""

    class LPF(SuperEnum):
        """理想低通"""
        radius: uuid_t = 0.5
        """半径相比原图半径的比例"""

    class HPF(SuperEnum):
        """理想高通"""
        radius: uuid_t = 0.5
        """半径相比原图半径的比例"""


def frequency_domain_filtering(im: ImStd, freq: uuid_t, para: Mapping[uuid_t, Any] | None = None) -> ImStd:
    """频域滤波"""
    ImFreq.check_in_(freq)
    freq_type = ImFreq.get_cls_(freq)
    para = freq_type.dict_(para, is_force=True)
    style, scale, color = im.style, im.scale, im.color
    im = im.to_style(ImStyle.im_ndarray, is_new=True)
    im.to_scale_(ImScale.float_n), im.to_color_(ImColor.freq)
    image = im.image

    if freq == ImFreq.LPF.get_uuid_():
        rows, cols = image.shape
        c_row, c_col = rows // 2, cols // 2
        r_radius, c_radius = int(para[ImFreq.LPF.radius] * rows), int(para[ImFreq.LPF.radius] * cols)
        mask = np.zeros((rows, cols), np.uint8)
        cv2.ellipse(mask, (c_col, c_row), (c_radius, r_radius), 
                    angle=0, startAngle=0, endAngle=360, color=(1, 1, 1), thickness=-1)
        image = image * mask
    elif freq == ImFreq.HPF.get_uuid_():
        rows, cols = image.shape
        c_row, c_col = rows // 2, cols // 2
        r_radius, c_radius = int(para[ImFreq.HPF.radius] * rows), int(para[ImFreq.HPF.radius] * cols)
        mask = np.ones((rows, cols), np.uint8)
        cv2.ellipse(mask, (c_col, c_row), (c_radius, r_radius),
                    angle=0, startAngle=0, endAngle=360, color=(0, 0, 0), thickness=-1)
        image = image * mask
    else:
        raise ImageFilterError(
            f"Filter {ImFreq.get_name_(freq)} is not implemented a filter function!!!")

    im.image = image
    im.to_style_(style), im.to_scale_(scale), im.to_color_(color)
    return im
