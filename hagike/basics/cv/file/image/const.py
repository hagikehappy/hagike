"""
***常量表***
"""


from hagike.utils import *
import numpy as np
import torch
from PIL import Image, ImageFile


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
    else__: uuid_t = None
    """非标准格式"""


@advanced_enum()
class ImScale(SuperEnum):
    """
    图像值范围 \n
    .. important::
        数据类型需要是 `np` 系列的，否则检查时会报错，内部会实现其它类型与 `np` 类型的转换！！！
    """
    uint8_t: uuid_t = ((0, 255), np.uint8)
    """
    标准RGB值 \n
    .. important::
        这里将所有的整数类型型都归为这一范围，
        且不会检查其范围是否真的在该区间内，但默认该区间为对应范围！！！
    """
    float_n: uuid_t = ((0.0, 1.0), np.float32)
    """
    归一化的像素值 \n
    .. important::
        这里将所有的浮点类型都归为这一范围，
        且不会检查其范围是否真的在该区间内，但默认该区间为对应范围！！！
    """
    auto__: uuid_t = None
    """自动判断"""
    else__: uuid_t = None
    """非标准范围"""


@advanced_enum()
class ImColor(SuperEnum):
    """颜色类型"""
    gray: uuid_t = None
    """单色"""
    colored: uuid_t = None
    """彩色"""
    freq: uuid_t = None
    """
    频域，限定单色或指定彩色某通道向此类型转换 \n
    频谱类型无视数值类型转换，容器中的 `scale` 指示的是频谱图来源的范围，而非频谱图本身的范围 \n
    容器中为实部虚部还是幅度相位，在附加信息表中会进行记录 \n
    """
    auto__: uuid_t = None
    """自动判断"""
    else__: uuid_t = None
    """非标准颜色"""


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


@advanced_enum()
class ImTerm(SuperEnum):
    """`ImStd` 中的项"""
    image__ = None
    style = None
    color = None
    scale = None
    info__ = None


@advanced_enum()
class ImInfo(SuperEnum):
    """
    图片的个性化信息，主要服务于非标准化情况，并提供用户个性化存储的容器 \n
    这里列出的仅为常用类型，实际使用时包括且不限于此，这里仅 others 中的项是用户有自由设置权限的，其它都需要通过接口设置 \n
    """
    vscale = None
    """范围，要求为有两项的元组"""
    ap_or_ri = None
    """频域情况下，图像为 `amplitude-phase` 还是 `real-imaginary`"""
    others = dict()
    """用户自定义项，是一个字典"""


@advanced_enum()
class ImOutput(SuperEnum):
    """
    显示图像时的选择性参数 \n
    """
    mag_or_phase = None
    """如果是频域，那显示幅频还是相频"""
    others = dict()
    """用户自定义项，是一个字典，会在下层调用时展开"""


vscale_t = Tuple[Tuple[int | float, int | float], np.dtype]
"""数据范围类型"""
