from hagike.basics.cv.file.image.im import *


def test_basics_cv_file(path: str = 'tmp/raw.png'):
    """basics.cv.file测试用例"""
    im = import_image(path, ImStyle.im_file)
    im.to_style_(ImStyle.im_ndarray)
    im.to_gray_()
    im.to_scale_(ImScale.float_n)
    im.to_scale_(ImScale.uint8_t)
    im.show_image(ImShow.windows)








