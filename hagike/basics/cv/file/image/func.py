"""
***对于图像容器的函数API***
"""


from .im import *


def import_image(path: str, style: uuid_t, is_cache: bool = False) -> ImStd:
    """
    从路径中导入图像 \n
    `style` 指定了导入方式 \n
    `is_cache` 指定了是否是从缓存中导入，期望类型由 `style` 指定，可以为 `auto__`  \n
    `scale`，如果从路径导入，一般为 `uint8`，若从 cache 导入则与存入情况一样 \n
    `freq`，只能从缓存中导入，因为一般图片无法有效保存频谱信息 \n
    如果以 `cv2.imread` 方式导入，默认指定 `cv2.IMREAD_ANYCOLOR`，否则灰度图也会有3个通道 \n
    """
    ImStyle.check_in_(style, all_or_index=True)
    check_path_readable(path)
    if is_cache:
        im: ImStd = load_data_from_pkl(path)
    else:
        if style == ImStyle.cv_ndarray:
            image = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
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

