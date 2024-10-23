"""
图像导入库与格式转换库 \n
与图像处理相关的库主要有2个，现在阐明其区别与联系： \n
1. `cv2`，图像格式为 `np.ndarray`，作为向量格式，主要用于计算机视觉中对于图像的高级处理算法 \n
2. `Image`，图像格式为 `ImageFile` 或 `Image`，作为文件格式，主要用于对图像的直接操作，如格式转换等；但可以转化为`np.ndarray` \n
而图片在各层级上存在如下区别与联系： \n
1. 屏幕显示：坐标原点在左上角，y轴向下，x轴向右 \n
3. `np.ndarray`： \n
    - `im_ndarray`：源于Image的导入后直接转换；彩图为三维，$[H, W, C]$，通道顺序为 `R, G, B` ；灰度图为二维，$[H, W]$； \n
    - `cv_ndarray`：源于cv2的导入，彩图的通道顺序为 `B, G, R` \n
4. `torch.Tensor`：卷积神经网络的期待输入为$[C, H, W]$，即使是灰度图也依然需要通道维度 \n
同时，各层级上图片的数值属性也存在区别： \n
1. `np.ndarray`：cv2读入或Image转化后默认为 `np.uint8` 类型，范围在$[0, 255]$ \n
2. `torch.Tensor`：需要与待输入的网络保持格式一致，默认类型为`torch.float32`，
    考虑到计算复杂度，初始时不进行归一化，保持$[0.0, 255.0]$，归一化过程在网络中定义 \n
值得注意的是，python中的对整型的除法 `/` 默认会将原类型转换为小数格式，这对 `np.ndarray` 和 `torch.Tensor` 同样均成立； \n
如果希望整除，这需要使用 `//` 操作 \n
下面在各个图像操作函数内会具体说明操作差异。 \n
"""

from __future__ import annotations
import typing
import numpy as np
import torch
from PIL import Image, ImageFile
import cv2
import matplotlib.pyplot as plt
from ...utils.enum import *


im_file = Image.Image | ImageFile.ImageFile
cv_ndarray = np.ndarray
im_ndarray = np.ndarray
im_tensor = torch.Tensor
im_all = im_file | cv_ndarray | im_ndarray | im_tensor


@advanced_enum()
class ImStyle(SuperEnum):
    """图像格式，值为须设计为可用于 `isinstance` 的类型"""
    im_file = (Image.Image, ImageFile.ImageFile)
    """Image格式"""
    cv_ndarray = cv_ndarray
    """cv2格式"""
    im_ndarray = im_ndarray
    """Image转array格式"""
    im_tensor = im_tensor
    """torch格式"""
    _auto = None
    """自动判断"""


@advanced_enum()
class ImColor(SuperEnum):
    """颜色类型"""
    gray = None
    """单色"""
    colored = None
    """彩色"""
    _auto = None
    """自动判断"""


class ImageStyleError(Exception):
    """图像类型与实际类型不符"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ImageColorError(Exception):
    """图像的颜色空间不支持或数据不符合预期"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ImStd:
    """标准图像容器"""

    def __init__(self, image: im_all, style: uuid_t = ImStyle._auto, color: uuid_t = ImColor._auto,
                 check_style: bool = False, check_color: bool = False) -> None:
        """
        初始化，`is_check` 指定是否进行检查与判断；
        如果未显式指定，且 `style` 或 `color` 为 `_auto` 时使用懒惰思路，即当有转换需求的时候进行判断
        """
        self._image, self._style, self._color = image, style, color
        if check_style:
            self._judge_style()
        if check_color:
            self._judge_color()

    @property
    def image(self):
        return self._image

    @property
    def style(self):
        return self._style

    @property
    def color(self):
        return self._color

    def _auto_style(self) -> None:
        """自动判断类型"""
        for uuid in ImStyle.iter():
            if isinstance(self._image, ImStyle.get_value(uuid)):
                self._style = uuid
                break
        else:
            raise ImageStyleError(f"ERROR: Image style {type(self._image)} is unknown!!!")

    def _judge_style(self) -> None:
        """检查图像类型"""
        ImStyle.check_in(self._style, all_or_index=True)
        if self._style == ImStyle._auto:
            self._auto_style()
        else:
            if not isinstance(self._image, ImStyle.get_value(self._style)):
                raise ImageStyleError(
                    f"ERROR: Image style {type(self._image)} is not as given, {ImStyle.get_value(self._style)}!!!")

    def _auto_color(self) -> None:
        """判断颜色类型，同时检查数据是否符合预期"""
        if self._style == ImStyle.im_file:
            if self._image.mode == 'L':
                self._color = ImColor.gray
            elif self._image.mode == 'RGB':
                self._color = ImColor.colored
            else:
                raise ImageColorError(f"ERROR: The image, {ImStyle.get_value(self._style)}, "
                                      f"has an unknown color format, f{self._image.mode}!!!")
        elif self._style == ImStyle.im_tensor:
            if len(self._image.shape) != 3:
                raise ImageColorError(f"ERROR: The image, {ImStyle.get_value(self._style)}, "
                                      f"has wrong shape, {self._image.shape}!!!")
            else:
                if self._image.shape[0] == 1:
                    self._color = ImColor.gray
                elif self._image.shape[0] == 3:
                    self._color = ImColor.colored
                else:
                    raise ImageColorError(f"ERROR: The image, {ImStyle.get_value(self._style)}, "
                                          f"has unknown numbers of color channel, {self._image.shape}!!!")
        elif self._style == ImStyle.im_ndarray or self._style == ImStyle.cv_ndarray:
            if len(self._image.shape) == 3:
                self._color = ImColor.colored
            elif len(self._image.shape) == 2:
                self._color = ImColor.gray
            else:
                raise ImageColorError(f"ERROR: The image, {ImStyle.get_value(self._style)}, "
                                      f"has unknown shapes, {self._image.shape}!!!")
        else:
            raise ImageStyleError(
                f"ERROR: Image style {type(self._image)} is not as given, {ImStyle.get_value(self._style)}!!!")

    def _judge_color(self) -> None:
        """检查颜色类型"""
        ImColor.check_in(self._color, all_or_index=True)
        if self._color == ImColor._auto:
            self._auto_color()
        else:
            color = self._color, self._auto_color()
            if color != self._color:
                raise ImageColorError(f"ERROR: The image's color is different from the auto-detection!!!")

    def to(self, style: uuid_t) -> None:
        """
        格式转换为指定 `style` \n
        这里将 `im_ndarray` 作为中间格式用于转换 \n
        """
        # 判定是否为标准格式


def import_image(path: str, style: uuid_t = ImStyle.im_file) -> ImStd:
    """
    从路径中导入图像 \n
    `style` 指定了导入方式 \n
    """
    ImStyle.check_in(style, all_or_index=False)
    if style == ImStyle.cv_ndarray:
        image = cv2.imread(path)
    else:
        image = Image.open(path)
        if style == ImStyle.im_file:
            pass
        else:
            image = np.array(image)
            if style == ImStyle.im_ndarray:
                pass
            elif style == ImStyle.im_tensor:
                image = torch.tensor(image, dtype=torch.float32, device='cpu')
                if len(image.shape) == 2:
                    image.unsqueeze_(0)
    image = ImStd(image, style)
    return image



def show_image(image: im_file, cmap=None) -> None:
    """
    显示图像； \n
    这里说明显示图像的三种方法： \n
    1.
    plt.imshow可以接收Image(即ImageFile)和np.ndarray两种格式的输入，主要是用于显示数据图而非原始图像的；
    这是因为plt.imshow
    如果输入的是灰度图，且希望以灰度方式显示，则需要指定参数cmap='gray'，
    否则会以其它颜色映射方式显示(默认为'viridis'，蓝色到绿色再到黄色的渐变) \n
    """
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
