"""
***图像导入库与格式转换库*** \n
与**图像处理相关的库**主要有2个，现在阐明其区别与联系： \n
1. `cv2`，图像格式为 `np.ndarray`，作为向量格式，主要用于计算机视觉中对于图像的高级处理算法 \n
2. `Image`，图像格式为 `ImageFile` 或 `Image`，作为文件格式，主要用于对图像的直接操作，如格式转换等；但可以转化为`np.ndarray` \n
而**图片**在各层级上存在如下区别与联系： \n
1. 屏幕显示：坐标原点在左上角，y轴向下，x轴向右 \n
2. `np.ndarray`： \n
    - `im_ndarray`：源于Image的导入后直接转换；彩图为三维，$[H, W, C]$，通道顺序为 `R, G, B` ；灰度图为二维，$[H, W]$； \n
    - `cv_ndarray`：源于cv2的导入，彩图的通道顺序为 `B, G, R` \n
3. `torch.Tensor`：卷积神经网络的期待输入为$[C, H, W]$，即使是灰度图也依然需要通道维度 \n
同时，各层级上**图片的数值属性**也存在区别： \n
1. `np.ndarray`：cv2读入或Image转化后默认为 `np.uint8` 类型，范围在$[0, 255]$ \n
2. `torch.Tensor`：需要与待输入的网络保持格式一致，默认类型为`torch.float32`，
    考虑到计算复杂度，初始时不进行归一化，保持$[0.0, 255.0]$，归一化过程在网络中定义 \n
值得**注意**的是，python中的对整型的除法 `/` 默认会将原类型转换为小数格式，这对 `np.ndarray` 和 `torch.Tensor` 同样均成立； \n
如果希望整除，这需要使用 `//` 操作 \n
下面在各个图像操作函数内会具体说明操作差异。 \n
**term** \n
    im - 图像容器
    image - 图像文件
    style - 格式类型
    color - 颜色类型
    show - 显示方式
.. todo::
    1. 目前开放并未开放某一图片格式的自定义参数输入，而只能使用默认方式 \n
    2. 目前ImStd容器集成了对各格式的具体处理，后续需要将格式特定部分分散到具体格式的封装中 \n
"""

from __future__ import annotations
import typing
from datetime import datetime
import numpy as np
import torch
from PIL import Image, ImageFile
import cv2
import matplotlib.pyplot as plt
from ...utils import *
from copy import deepcopy


im_file = Image.Image | ImageFile.ImageFile
cv_ndarray = np.ndarray
im_ndarray = np.ndarray
im_tensor = torch.Tensor
im_all = im_file | cv_ndarray | im_ndarray | im_tensor


@advanced_enum()
class ImStyle(SuperEnum):
    """图像格式，值为须设计为可用于 `isinstance` 的类型"""
    im_file: uuid_t = (Image.Image, ImageFile.ImageFile)
    """Image格式"""
    cv_ndarray: uuid_t = cv_ndarray
    """cv2格式"""
    im_ndarray: uuid_t = im_ndarray
    """Image转array格式"""
    im_tensor: uuid_t = im_tensor
    """torch格式"""
    auto__: uuid_t = None
    """自动判断"""


@advanced_enum()
class ImColor(SuperEnum):
    """颜色类型"""
    gray: uuid_t = None
    """单色"""
    colored: uuid_t = None
    """彩色"""
    auto__: uuid_t = None
    """自动判断"""


@advanced_enum()
class ImShow(SuperEnum):
    """图片显示方式"""
    inner = None
    """
    ***调用IDE嵌入查看器*** \n
    `plt.imshow` 可以接收 `Image` 和 `np.ndarray` 两种格式的输入，主要是用于显示数据图而非原始图像的； \n
    在显示或转换时，如果类型不是uint8类型的，容器会给出警告，因为自动转换可能是不可靠的 \n
    但优势在于，Pycharm等IDE中有嵌入的查看器，无需打开额外窗口，且是非阻塞的；同时，图像会带有坐标轴 \n
    在使用上，如果输入的是灰度图，且希望以灰度方式显示，则需要指定参数cmap='gray'，
    否则会以其它颜色映射方式显示(默认为'viridis'，即从蓝色到绿色再到黄色的渐变) \n
    .. warning::
        `plt.imshow` 总是会根据 `np.ndarray` 或 `Image` 转换后得到的 `np.ndarray` 的 `min` 和 `max` 自动缩放数据到$[0, 255]$
    """
    system = None
    """
    ***调用系统图片查看器*** \n
    `Image.show` ，`Image` 格式自带方式，非阻塞，但要与系统默认应用，且要打开额外窗口，较为麻烦 \n
    .. warning::
        在某些IDE的运行环境中，运行环境与外界环境是隔离的，这时会出现警告信息，且窗口不会被打开，但是也不会中断运行 \n
        但如果直接在系统内使用python运行，则不会有问题
    """
    windows = None
    """
    ***调用窗口查看器*** \n
    `cv2.imshow` 会启动cv2的私有窗口，但是是阻塞的，直到进行特定按键窗口才会被销毁 \n
    """


class ImageStyleError(Exception):
    """图像类型与实际类型不符"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ImageInfoError(Exception):
    """图像的颜色空间不支持或数据不符合预期"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ImageTransferError(Exception):
    """无法正常转换图像类型"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ImageFunctionError(Exception):
    """图像功能性错误"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


def import_image(path: str, style: uuid_t, is_cache: bool = False) -> ImStd:
    """
    从路径中导入图像 \n
    `style` 指定了导入方式 \n
    `is_cache` 指定了是否是从缓存中导入，期望类型由 `style` 指定，可以为 `auto__`  \n
    """
    ImStyle.check_in_(style, all_or_index=True)
    check_path_readable(path)
    if is_cache:
        image = load_data_from_pkl(path)
    else:
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
                else:
                    raise ImageStyleError(
                        f"ERROR: Image style {ImStyle.get_value_(style)} is not implemented an 'import' function!!!")
    im = ImStd(image, style)
    return im


class ImStd:
    """标准图像容器"""

    def __init__(self, image: im_all, style: uuid_t = ImStyle.auto__, color: uuid_t = ImColor.auto__,
                 check_style: bool = False, check_color: bool = False) -> None:
        """
        初始化，`is_check` 指定是否进行检查与判断；
        如果未显式指定，且 `style` 或 `color` 为 `_auto` 时使用懒惰思路，即当有转换需求的时候进行判断
        """
        self._image: im_all = image
        self._style: uuid_t = style
        self._color: uuid_t = color
        ImStyle.check_in_(self._style, all_or_index=True), ImColor.check_in_(self._color, all_or_index=True)
        self._is_check_style, self._is_check_color = False, False
        if check_style:
            self._judge_style()
        if check_color:
            self._judge_color()

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        """重设 `image` 时更新检查状态"""
        self._image = image
        self._is_check_style, self._is_check_color = False, False

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        """重设 `style` 时更新检查状态"""
        self._style = style
        self._is_check_style = False

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        """重设 `color` 时更新检查状态"""
        self._color = color
        self._is_check_color = False

    def _auto_style(self) -> None:
        """自动判断类型"""
        for uuid in ImStyle.iter_():
            if isinstance(self._image, ImStyle.get_value_(uuid)):
                self._style = uuid
                break
        else:
            raise ImageStyleError(f"ERROR: Image style {type(self._image)} is unknown!!!")

    def _judge_style(self) -> None:
        """检查图像类型"""
        self._is_check_style = True
        if self._style == ImStyle.auto__:
            self._auto_style()
        else:
            if not isinstance(self._image, ImStyle.get_value_(self._style)):
                raise ImageStyleError(
                    f"ERROR: Image style {type(self._image)} is not as given, {ImStyle.get_value_(self._style)}!!!")

    def _auto_color(self) -> None:
        """判断颜色类型，同时检查数据是否符合预期"""
        if self._style == ImStyle.im_file:
            if self._image.mode == 'L':
                self._color = ImColor.gray
            elif self._image.mode == 'RGB':
                self._color = ImColor.colored
            else:
                raise ImageInfoError(f"ERROR: The image, {ImStyle.get_value_(self._style)}, "
                                     f"has an unknown color format, f{self._image.mode}!!!")
        elif self._style == ImStyle.im_tensor:
            if len(self._image.shape) != 3:
                raise ImageInfoError(f"ERROR: The image, {ImStyle.get_value_(self._style)}, "
                                     f"has wrong shape, {self._image.shape}!!!")
            else:
                if self._image.shape[0] == 1:
                    self._color = ImColor.gray
                elif self._image.shape[0] == 3:
                    self._color = ImColor.colored
                else:
                    raise ImageInfoError(f"ERROR: The image, {ImStyle.get_value_(self._style)}, "
                                         f"has unknown numbers of color channel, {self._image.shape}!!!")
        elif self._style == ImStyle.im_ndarray or self._style == ImStyle.cv_ndarray:
            if len(self._image.shape) == 3:
                self._color = ImColor.colored
            elif len(self._image.shape) == 2:
                self._color = ImColor.gray
            else:
                raise ImageInfoError(f"ERROR: The image, {ImStyle.get_value_(self._style)}, "
                                     f"has unknown shapes, {self._image.shape}!!!")
        else:
            raise ImageStyleError(
                f"ERROR: Image style {type(self._image)} is not as given, {ImStyle.get_value_(self._style)}!!!")

    def _judge_color(self) -> None:
        """检查颜色类型"""
        self._is_check_color = True
        if self._color == ImColor.auto__:
            self._auto_color()
        else:
            color = self._color
            self._auto_color()
            if color != self._color:
                raise ImageInfoError(f"ERROR: The image's color is different from the auto-detection!!!")

    def _judge(self) -> None:
        """检查类型与颜色"""
        if not self._is_check_style:
            self._judge_style()
        if not self._is_check_color:
            self._judge_color()

    def _to_style(self, style: uuid_t) -> im_all | None:
        """`to` 函数的实际执行函数，若返回 `None` 则说明没有变化"""
        # 检查或判断
        ImStyle.check_in_(style)
        self._judge()
        # 排除同类转换
        if style == self._style:
            return
        # 转换为中间格式
        image = None
        if self._style == ImStyle.im_ndarray:
            pass
        elif self._style == ImStyle.cv_ndarray:
            if self._color == ImColor.colored:
                image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        elif self._style == ImStyle.im_file:
            image = np.array(self._image)
        elif self._style == ImStyle.im_tensor:
            image = self._image.to('cpu')
            if self._color == ImColor.gray:
                image.squeeze_(0)
            elif self._color == ImColor.colored:
                image = image.permute(1, 2, 0)
            else:
                raise ImageInfoError(f"ERROR: The image color format, {self._color}, is not implemented!!!")
            image = image.numpy()
        else:
            raise ImageStyleError(
                f"ERROR: Image style {ImStyle.get_value_(self._style)} is not implemented a 'to' function!!!")
        # 从中间格式转换为目标格式并进行必要的转换检查
        if style == ImStyle.im_ndarray:
            pass
        elif style == ImStyle.cv_ndarray:
            if self._color == ImColor.colored:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif style == ImStyle.im_file:
            # 检查数据类型，若不为uint8型的则发出警告
            add_msg(MsgLevel.Warning, f"The image dtype, {image.dtype}, is not {np.uint8}")
            if self._color == ImColor.colored:
                mode = 'RGB'
            elif self._color == ImColor.gray:
                mode = 'L'
            else:
                raise ImageInfoError(f"ERROR: The image color format, {self._color}, is not implemented!!!")
            image = Image.fromarray(image, mode=mode)
        elif style == ImStyle.im_tensor:
            image = torch.tensor(image, device='cpu')
            if self._color == ImColor.gray:
                image = np.expand_dims(image, axis=0)
            elif self._color == ImColor.colored:
                image = np.transpose(image, (2, 0, 1))
            else:
                raise ImageInfoError(f"ERROR: The image color format, {self._color}, is not implemented!!!")
        else:
            raise ImageStyleError(
                f"ERROR: Image style {ImStyle.get_value_(style)} is not implemented a 'to' function!!!")
        return image

    def to_style_(self, style: uuid_t) -> None:
        """
        格式转换为指定 `style` \n
        这里将 `im_ndarray` 作为中间格式用于转换 \n
        在因为格式原因无法进行转换时，会抛出异常 \n
        转换会就地进行 \n
        """
        image = self._to_style(style)
        if image is not None:
            self._image = image
            self._style = style

    def to_style(self, style: uuid_t, is_new: bool = False) -> ImStd:
        """
        格式转换 \n
        `is_new=True`，则总是返回一个全新的 `ImStd` 图像容器，无论目标类型与原有类型是否一致 \n
        `is_new=False`，则仅在目标类型与原有类型不一致时返回深拷贝结果，否则返回本身；
        当明确知道后续只有只读操作后可以这样做，会节省算力与内存 \n
        """
        image = self._to_style(style)
        if image is None:
            image = self._image
        if image is None and is_new is False:
            return self
        else:
            return ImStd(deepcopy(image), style, self._color)

    def _to_gray(self) -> im_all | None:
        """`to_gray` 的实质执行函数，如果本身就是灰度图则返回 `None`"""
        self._judge()
        if self._color == ImColor.gray:
            return
        elif self._color == ImColor.colored:
            if self._style == ImStyle.im_file:
                image = self._image.convert('L')
            elif self._style == ImStyle.im_ndarray:
                image = cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY)
            elif self._style == ImStyle.cv_ndarray:
                image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
            elif self._style == ImStyle.im_tensor:
                # 在此处手动转化时需要注意前后数据类型不变
                image = (self._image[0: 1, :, :] * 0.299 +
                         self._image[1: 2, :, :] * 0.587 +
                         self._image[2: 3, :, :] * 0.114).to(self._image.dtype)
            else:
                raise ImageStyleError(
                    f"ERROR: Image style {ImStyle.get_value_(self._style)} is not implemented a 'to_gray' function!!!")
        else:
            raise ImageInfoError(f"ERROR: The image color format, {self._color}, is not implemented!!!")
        return image

    def to_gray_(self) -> None:
        """将图像转为灰度图，并返回新的图像，本地转换"""
        image = self._to_gray()
        if image is not None:
            self._image = image
            self._color = ImColor.gray

    def to_gray(self, is_new: bool = False) -> ImStd:
        """
        格式转换 \n
        `is_new=True`，则总是返回一个全新的 `ImStd` 图像容器，无论原有颜色是否为 `gray` \n
        `is_new=False`，则仅在原有颜色不为 `gray` 时返回深拷贝结果，否则返回本身；
        当明确知道后续只有只读操作后可以这样做，会节省算力与内存 \n
        """
        image = self._to_gray()
        if image is None:
            image = self._image
        if image is None and is_new is False:
            return self
        else:
            return ImStd(deepcopy(image), self._style, ImColor.gray)

    def show_image(self, show: uuid_t) -> None:
        """显示图像，`show` 指定了图像显示的方式"""
        ImShow.check_in_(show)
        self._judge()
        if show == ImShow.inner:
            im = self.to_style(ImStyle.im_ndarray, is_new=False)
            if self._image.dtype != np.uint8:
                add_msg(MsgLevel.Warning, f"The image dtype, {self._image.dtype}, is not {np.uint8}")
            cmap = 'gray' if im.color == ImColor.gray else None
            plt.imshow(im.image, cmap=cmap)
            plt.show()
        elif show == ImShow.system:
            im = self.to_style(ImStyle.im_file, is_new=False)
            im.image.show()
        elif show == ImShow.windows:
            im = self.to_style(ImStyle.cv_ndarray, is_new=False)
            image = im.image
            if image.dtype != np.uint8:
                add_msg(MsgLevel.Warning, f"The image dtype, {image.dtype}, is not {np.uint8}")
            winname = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            cv2.imshow(winname, image)
            cv2.waitKey(0)
            cv2.destroyWindow(winname)
        else:
            raise ImageFunctionError(
                f"ERROR: Showing Type {ImShow.get_name_(show)} is not implemented!!!")
        return

    def save_image(self, path: str, is_cache: bool = False) -> None:
        """
        保存图像 \n
        `is_cache` 指定了是否使用数据的序列化缓存格式
        """
        if is_cache:
            save_data_to_pkl(self._image, path)
        else:
            ensure_path_writable(path, is_raise=True)
            im = self.to_style(ImStyle.im_file, is_new=False)
            im.image.save(path)


