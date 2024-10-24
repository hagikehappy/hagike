from hagike.basics.cv.file import *


def test_basics_cv_file(path: str = 'tmp/raw.png'):
    """basics.cv.file测试用例"""
    im = import_image(path, ImStyle.im_file)
    im.to_style_(ImStyle.im_tensor)
    im.to_gray_()
    # im.show_image(ImShow.windows)







