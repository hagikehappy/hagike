"""
***图像导入库与格式转换库*** \n

**description** \n
与**图像处理相关的库**主要有2个，现在阐明其区别与联系： \n
1. `cv2`，图像格式为 `np.ndarray`，作为向量格式，主要用于计算机视觉中对于图像的高级处理算法 \n
2. `Image`，图像格式为 `ImageFile` 或 `Image`，作为文件格式，主要用于对图像的直接操作，如格式转换等；但可以转化为`np.ndarray` \n
而**图片**在各层级上存在如下区别与联系： \n
1. 屏幕显示：坐标原点在左上角，y轴向下，x轴向右 \n
2. `np.ndarray`： \n
    - `im_ndarray`：源于Image的导入后直接转换；彩图为三维，$[H, W, C]$，通道顺序为 `R, G, B` ；灰度图为二维，$[H, W]$； \n
    - `cv_ndarray`：源于cv2的导入，彩图的通道顺序为 `B, G, R` \n
3. `torch.Tensor`：卷积神经网络的期待输入为$[C, H, W]$，即使是灰度图也依然需要通道维度 \n
值得**注意**的是，python中的对整型的除法 `/` 默认会将原类型转换为小数格式，这对 `np.ndarray` 和 `torch.Tensor` 同样均成立； \n
如果希望整除，这需要使用 `//` 操作 \n
下面在各个图像操作函数内会具体说明操作差异。 \n

.. important::
    图像标准化容器 `ImStd` 设计的思想是，实现常量表中列出的最常用的配置； \n
    但对于非标准化的或待实现的配置，也允许使用，但要主动标记为 `else__` 来回避内置检查，但是不能调用该配置项相关的库函数，否则会报错； \n
    `auto__` 情况下在自动判断时会将不可识别项标记为 `else__` 并给出警告，
    考虑到自动判断是可能是不可靠的，最好主动标记为 `else__`，以避免不可预知的行为； \n
    对 `color` 和 `scale` 的判断依赖于 `style`，如果 `style` 项被标记为 `else__`，则 `color` 和 `scale` 也会被判断为 `else__`，
    但 `color` 与 `scale` 相互独立； \n

**term** \n
    im - 图像容器
    image - 图像文件
    style - 格式类型
    color - 颜色类型
    scale - 数据范围
    show - 显示方式

.. todo::
    1. 目前开放并未开放内置函数的自定义参数输入，而只能使用默认方式，缺少灵活拓展的功能 \n
    2. 目前ImStd容器集成了对各格式的具体处理，后续需要将格式特定部分分散到具体格式的封装中 \n
"""

from __future__ import annotations

import warnings

from .const import *
from .error import *
from datetime import datetime
from copy import deepcopy
from types import MappingProxyType
import cv2
import matplotlib.pyplot as plt


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
                    image = torch.tensor(image, device='cpu')
                    if len(image.shape) == 2:
                        image.unsqueeze_(0)
                else:
                    raise ImageStyleError(
                        f"ERROR: Image style {ImStyle.get_value_(style)} is not implemented an 'import' function!!!")
    im = ImStd(image, style)
    return im


class ImStd:
    """标准图像容器"""

    _np_int = (np.int64, np.int32, np.int16, np.int8,
               np.uint64, np.uint32, np.uint16, np.uint8, np.bool)
    """ `np` 中的整型集合"""
    _torch_int = (torch.int64, torch.int32, torch.int16, torch.int8,
                  torch.uint64, torch.uint32, torch.uint16, torch.uint8, torch.bool)
    """ `torch` 中的整型集合"""
    _np_float = (np.float64, np.float32, np.float16)
    """ `np` 中的浮点型集合"""
    _torch_float = (torch.float64, torch.float32, torch.float16)
    """ `torch` 中的浮点型集合"""
    _else = {ImTerm.style: ImStyle.else__, ImTerm.color: ImColor.else__, ImTerm.scale: ImScale.else__}
    """ 非标准化项在各 `term` 下对应的 `uuid` 值 """

    def __init__(self,
                 image: im_all,
                 style: uuid_t = ImStyle.auto__,
                 color: uuid_t = ImColor.auto__,
                 scale: uuid_t = ImScale.auto__,
                 check_style: bool = False,
                 check_color: bool = False,
                 check_scale: bool = False,
                 ) -> None:
        """
        初始化，`is_check` 指定是否进行检查与判断； \n
        如果未显式指定，且 `style` 或 `color` 或 `scale` 为 `_auto` 时使用懒惰思路，即当有转换需求的时候进行判断 \n
        """
        self._image: im_all = image
        self._style: uuid_t = style
        self._color: uuid_t = color
        self._scale: uuid_t = scale
        self._info: Dict = ImInfo.dict_()

        ImStyle.check_in_(self._style, all_or_index=True), ImColor.check_in_(self._color, all_or_index=True)
        self._is_check_style, self._is_check_color, self._is_check_scale = False, False, False
        if check_style:
            self._judge_style()
        if check_color:
            self._judge_color()
        if check_scale:
            self._judge_scale()

    @property
    def image(self) -> im_all | Any:
        """返回 `image` 本身"""
        return self._image

    @property
    def terms(self) -> Dict[uuid_t, Any]:
        """返回配置信息表"""
        return {ImTerm.style: self._style,
                ImTerm.color: self._color,
                ImTerm.scale: self._scale}

    @image.setter
    def image(self, image: Any) -> None:
        """重设 `image` 时更新检查状态，这里默认认为各配置项不变，若变则需要显式更改其它配置项，否则后续调用内置函数时自动检查会触发异常"""
        self._image = image
        self._is_check_style, self._is_check_color, self._is_check_scale = False, False, False

    @property
    def style(self) -> uuid_t:
        """给出 `style` 前进行检查"""
        self._judge_style()
        return self._style

    @style.setter
    def style(self, style: uuid_t) -> None:
        """重设 `style` 时更新检查状态"""
        self._style = style
        self._is_check_style = False

    @property
    def color(self) -> uuid_t:
        """给出 `color` 前进行检查"""
        self._judge_color()
        return self._color

    @color.setter
    def color(self, color) -> None:
        """重设 `color` 时更新检查状态"""
        self._color = color
        self._is_check_color = False

    @property
    def scale(self) -> uuid_t:
        """给出 `scale` 前进行检查"""
        self._judge_scale()
        return self._scale

    @scale.setter
    def scale(self, scale) -> None:
        """重设 `scale` 时更新检查状态，并重置 `vscale`"""
        self._scale = scale
        self._info[ImInfo.vscale] = None
        self._is_check_scale = False

    @property
    def info(self) -> MappingProxyType:
        """返回只读信息表"""
        return MappingProxyType(self._info)

    @property
    def others(self) -> Dict:
        """返回用户自定义信息表"""
        return self._info[ImInfo.others]

    @property
    def vscale(self) -> None | tuple:
        """返回自定义数据范围"""
        return self._info[ImInfo.vscale]

    @vscale.setter
    def vscale(self, vscale: vscale_t) -> None:
        """修改自定义数据范围，启用后 `scale` 自动变为 `else__`"""
        self.scale = ImScale.else__
        self._info[ImInfo.vscale] = vscale

    def _auto_style(self) -> None:
        """自动判断类型"""
        for uuid in ImStyle.iter_():
            if isinstance(self._image, ImStyle.get_value_(uuid)):
                self._style = uuid
                break
        else:
            self._style = ImStyle.else__
            warnings.warn(f"Image style {type(self._image)} is unknown!!!", ImageStyleWarning)

    def _judge_style(self) -> None:
        """检查图像类型"""
        self._is_check_style = True
        if self._style == ImStyle.auto__:
            self._auto_style()
        elif self._style == ImStyle.else__:
            # 由于 `color` 和 `scale` 依赖 `style`，故直接重设 `style`
            self._color = ImColor.else__
            self._scale = ImScale.else__
        else:
            if not isinstance(self._image, ImStyle.get_value_(self._style)):
                raise ImageStyleError(
                    f"Image style {type(self._image)} is not as given, {ImStyle.get_value_(self._style)}!!!")

    def _auto_color(self) -> None:
        """判断颜色类型，同时检查数据是否符合预期"""
        if self._style == ImStyle.im_file:
            if self._image.mode == 'L':
                self._color = ImColor.gray
            elif self._image.mode == 'RGB':
                self._color = ImColor.colored
            else:
                self._color = ImColor.else__
                warnings.warn(f"The image, {ImStyle.get_value_(self._style)}, "
                              f"has an unknown color format, f{self._image.mode}!!!", ImageInfoWarning)
        elif self._style == ImStyle.im_tensor:
            if len(self._image.shape) != 3:
                self._color = ImColor.else__
                warnings.warn(f"The image, {ImStyle.get_value_(self._style)}, "
                              f"has wrong shape, {self._image.shape}!!!", ImageInfoWarning)
            else:
                if self._image.shape[0] == 1:
                    self._color = ImColor.gray
                elif self._image.shape[0] == 3:
                    self._color = ImColor.colored
                else:
                    self._color = ImColor.else__
                    warnings.warn(f"The image, {ImStyle.get_value_(self._style)}, "
                                  f"has unknown numbers of color channel, {self._image.shape}!!!", ImageInfoWarning)
        elif self._style == ImStyle.im_ndarray or self._style == ImStyle.cv_ndarray:
            if len(self._image.shape) == 3:
                self._color = ImColor.colored
            elif len(self._image.shape) == 2:
                self._color = ImColor.gray
            else:
                self._color = ImColor.else__
                warnings.warn(f"The image, {ImStyle.get_value_(self._style)}, "
                              f"has unknown shapes, {self._image.shape}!!!", ImageInfoWarning)
        elif self._style == ImStyle.else__:
            self._color = ImColor.else__
        else:
            raise ImageStyleError(
                f"Image style, {ImStyle.get_value_(self._style)} is not implemented, a 'color'!!!")

    def _judge_color(self) -> None:
        """检查颜色类型"""
        self._is_check_color = True
        if self._color == ImColor.auto__:
            self._auto_color()
        elif self._color == ImColor.else__:
            pass
        else:
            color = self._color
            self._auto_color()
            if color != self._color:
                raise ImageInfoError(f"The image's color is different from the auto-detection!!!")

    def _auto_scale(self) -> None:
        """自动判断范围"""
        if self._style == ImStyle.im_file:
            self._scale = ImScale.uint8_t
        elif self._style == ImStyle.im_tensor or self._style == ImStyle.cv_ndarray or self._style == ImStyle.im_ndarray:
            if self._image.dtype in self._np_int:
                self._scale = ImScale.uint8_t
            elif self._image.dtype in self._np_float:
                self._scale = ImScale.float_n
            else:
                self._scale = ImScale.else__
                warnings.warn(f"The image dtype, {self._image.dtype}, "
                              f"is not in {self._np_int} or {self._np_float}", ImageInfoWarning)
        elif self._style == ImStyle.else__:
            self._scale = ImScale.else__
        else:
            raise ImageStyleError(
                f"Image style {ImStyle.get_value_(self._style)} is not implemented, a 'scale'!!!")

    def _judge_scale(self) -> None:
        """检查范围"""
        self._is_check_scale = True
        if self._scale == ImScale.auto__:
            self._auto_scale()
        elif self._scale == ImScale.else__:
            pass
        else:
            scale = self._scale
            self._auto_scale()
            if scale != self._scale:
                raise ImageInfoError(f"The image's scale is different from the auto-detection!!!")

    def _judge_all(self) -> None:
        """检查并刷新各项属性"""
        # `check_style` 需要在最前面，因为后面两项的检查依赖 `style`
        if not self._is_check_style:
            self._judge_style()
        if not self._is_check_color:
            self._judge_color()
        if not self._is_check_scale:
            self._judge_scale()

    def _judge_else(self, *terms) -> None:
        """排除非标准化情形，输入任意数量的检查项"""
        ImTerm.check_include_(terms, all_or_index=False)
        status = self.terms
        for term in terms:
            if status[term] == self._else[term]:
                raise ImageElseError(f"{ImTerm.get_name_(term)} is an `else__` type, not implemented!!!")

    def _to_style(self, style: uuid_t) -> im_all | None:
        """
        `to` 函数的实际执行函数，若返回 `None` 则说明没有变化； \n
        """
        # 检查或判断
        ImStyle.check_in_(style, all_or_index=False)
        self._judge_all()
        self._judge_else(ImTerm.style)
        # 排除同类转换
        if style == self._style:
            return
        # 转换为中间格式
        image = None
        if self._style == ImStyle.im_ndarray:
            image = self._image
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
                raise ImageInfoError(f"The image color format, {self._color}, is not implemented!!!")
            image = image.numpy()
        else:
            raise ImageStyleError(
                f"Image style {ImStyle.get_value_(self._style)} is not implemented a 'to' function!!!")
        # 从中间格式转换为目标格式并进行必要的转换检查
        if style == ImStyle.im_ndarray:
            pass
        elif style == ImStyle.cv_ndarray:
            if self._color == ImColor.colored:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif style == ImStyle.im_file:
            add_msg(MsgLevel.Warning, f"The image dtype, {image.dtype}, is not {np.uint8}")
            if self._color == ImColor.colored:
                mode = 'RGB'
            elif self._color == ImColor.gray:
                mode = 'L'
            else:
                raise ImageInfoError(f"The image color format, {self._color}, is not implemented!!!")
            image = Image.fromarray(image, mode=mode)
        elif style == ImStyle.im_tensor:
            image = torch.tensor(image, device='cpu')
            if self._color == ImColor.gray:
                image = np.expand_dims(image, axis=0)
            elif self._color == ImColor.colored:
                image = np.transpose(image, (2, 0, 1))
            else:
                raise ImageInfoError(f"The image color format, {self._color}, is not implemented!!!")
        else:
            raise ImageStyleError(
                f"Image style {ImStyle.get_value_(style)} is not implemented a 'to' function!!!")
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
        `is_new=False`，则仅在目标类型与原有类型不一致时返回深拷贝结果，否则返回本身； \n
        当明确知道后续只有只读操作后可以这样做，会节省算力与内存 \n
        要求 `style` 为标准化情形 \n
        """
        image = self._to_style(style)
        if image is None and is_new is False:
            return self
        else:
            im = deepcopy(self)
            if image is not None:
                im._image = image
            im._style = style
            return im

    def _to_gray(self) -> im_all | None:
        """
        `to_gray` 的实质执行函数，如果本身就是灰度图则返回 `None` \n
         """
        self._judge_all()
        self._judge_else(ImTerm.style, ImTerm.color)
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
                    f"Image style {ImStyle.get_value_(self._style)} is not implemented a 'to_gray' function!!!")
        else:
            raise ImageInfoError(f"The image color format, {self._color}, is not implemented!!!")
        return image

    def to_gray_(self) -> None:
        """
        将图像转为灰度图，并返回新的图像，本地转换 \n
        """
        image = self._to_gray()
        if image is not None:
            self._image = image
            self._color = ImColor.gray

    def to_gray(self, is_new: bool = False) -> ImStd:
        """
        格式转换 \n
        `is_new=True`，则总是返回一个全新的 `ImStd` 图像容器，无论原有颜色是否为 `gray` \n
        `is_new=False`，则仅在原有颜色不为 `gray` 时返回深拷贝结果，否则返回本身； \n
        当明确知道后续只有只读操作后可以这样做，会节省算力与内存 \n
        要求 `style` 和 `color` 为标准化情形，否则报错 \n
        """
        image = self._to_gray()
        if image is None and is_new is False:
            return self
        else:
            im = deepcopy(self)
            if image is not None:
                im._image = image
            im._color = ImColor.gray
            return im

    def _to_scale(self, scale: uuid_t, vscale: vscale_t | None = None) -> im_all | None:
        """
        `to_scale` 的实质执行函数，如果转换前后无变化则返回 `None` \n
        """
        ImScale.check_in_(scale, all_or_index=False)
        self._judge_all()
        self._judge_else(ImTerm.style)
        src_scale = self._info[ImInfo.vscale] if self._scale == ImScale.else__ else ImScale.get_value_(self._scale)
        if scale == ImScale.else__:
            dst_scale = vscale
            if vscale is None:
                raise ImageInfoError(f"The dst vscale, is not set when scale is `else__`!!!")
        else:
            dst_scale = ImScale.get_value_(scale)
            if vscale is not None:
                raise ImageInfoError(f"The dst vscale, {vscale}, is set "
                                     f"when scale, {dst_scale}, has been set in default!!!")
        if src_scale == dst_scale:
            return
        if self._style == ImStyle.im_file:
            image = self.to_style(ImStyle.im_ndarray).image
            image = np.clip(
                (dst_scale[0][1] - dst_scale[0][0]) / (src_scale[0][1] - src_scale[0][0])
                * (image - src_scale[0][0]) + dst_scale[0][0],
                0, 255).astype(np.uint8)
            image = self.to_style(ImStyle.im_file).image
        elif self._style == ImStyle.im_ndarray or self._style == ImStyle.cv_ndarray or self._style == ImStyle.im_tensor:
            image = ((dst_scale[0][1] - dst_scale[0][0]) / (src_scale[0][1] - src_scale[0][0])
                     * (self._image - src_scale[0][0]) + dst_scale[0][0])
            if self._style == ImStyle.im_ndarray or self._style == ImStyle.cv_ndarray:
                image = image.astype(dst_scale[1])
            else:
                # 将 `dtype` 类型由 `np.dtype` 映射为 `torch.dtype`
                if dst_scale[1] in self._np_int:
                    dtype = self._torch_int[self._np_int.index(dst_scale[1])]
                elif dst_scale[1] in self._np_float:
                    dtype = self._torch_float[self._np_float.index(dst_scale[1])]
                else:
                    raise ImageInfoError(
                        f"The dst dtype, {dst_scale[1]}, is not in {self._np_int} or {self._np_float}")
                image = image.type(dtype=dtype)
        else:
            raise ImageStyleError(
                f"Image style {ImStyle.get_value_(self._style)} is not implemented a 'to_scale' function!!!")
        return image

    def to_scale_(self, scale: uuid_t, vscale: vscale_t | None = None) -> None:
        """本地转换"""
        image = self._to_scale(scale, vscale)
        if image is not None:
            self._image = image
            self._scale = scale
            self._info[ImTerm.scale] = vscale

    def to_scale(self, scale: uuid_t, vscale: vscale_t | None = None, is_new: bool = False) -> ImStd:
        """
        范围转换，如果源范围是非标准化的，则要求 `info.vscale` 被设置 \n
        `is_new=True`，则总是返回一个全新的 `ImStd` 图像容器 \n
        `is_new=False`，则仅在原有范围不为 `scale` 或 `vscale` 时返回深拷贝结果，否则返回本身； \n
        要求 `style` 为标准化情形，否则报错； \n
        若 `style` 为 `im_file`，则无视类型说明，转换后的实际类型总是 `np.uint8`，但设置类型时依然设置为目标类型 \n
        .. important::
            `vscale` 只能在非标准情况下设置，如果标准情况下设置则会报错 \n
        """
        image = self._to_scale(scale, vscale)
        if image is None and is_new is False:
            return self
        else:
            im = deepcopy(self)
            if image is not None:
                im._image = image
            im._scale = scale
            im._info[ImTerm.scale] = vscale
            return im

    def show_image(self, show: uuid_t) -> None:
        """显示图像，`show` 指定了图像显示的方式"""
        ImShow.check_in_(show, all_or_index=False)
        self._judge_all()
        self._judge_else(ImTerm.style, ImTerm.color, ImTerm.scale)
        if show == ImShow.inner:
            im = self.to_style(ImStyle.im_ndarray, is_new=False)
            if self._image.dtype != np.uint8:
                warnings.warn(f"The image dtype, {self._image.dtype}, is not {np.uint8}", ImageFunctionWarning)
            if im.color == ImColor.gray:
                cmap = 'gray'
            elif im.color == ImColor.colored:
                cmap = None
            else:
                raise ImageInfoError(f"The image color format, {self._color}, is not implemented!!!")
            plt.imshow(im.image, cmap=cmap)
            plt.show()
        elif show == ImShow.system:
            im = self.to_style(ImStyle.im_file, is_new=False)
            im.image.show()
        elif show == ImShow.windows:
            im = self.to_style(ImStyle.cv_ndarray, is_new=False)
            image = im.image
            if image.dtype != np.uint8:
                warnings.warn(f"The image dtype, {self._image.dtype}, is not {np.uint8}", ImageFunctionWarning)
            winname = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            cv2.imshow(winname, image)
            cv2.waitKey(0)
            cv2.destroyWindow(winname)
        else:
            raise ImageFunctionError(
                f"Showing Type {ImShow.get_name_(show)} is not implemented!!!")
        return

    def save_image(self, path: str, is_cache: bool = False) -> None:
        """
        保存图像 \n
        `is_cache` 指定了是否使用数据的序列化缓存格式，缓存格式下不需要 `style` 是标准化的 \n
        """
        if is_cache:
            save_data_to_pkl(self._image, path)
        else:
            ensure_path_writable(path, is_raise=True)
            im = self.to_style(ImStyle.im_file, is_new=False)
            im.image.save(path)
