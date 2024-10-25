"""
***错误与警告***
"""


import warnings


class ImageStyleError(Exception):
    """图像类型与实际类型不符"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ImageStyleWarning(Warning):
    pass


class ImageInfoError(Exception):
    """图像的颜色空间不支持或数据不符合预期"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ImageInfoWarning(Warning):
    pass


class ImageElseError(Exception):
    """对非标准化情形使用内置函数"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ImageElseWarning(Warning):
    pass


class ImageFunctionError(Exception):
    """图像功能性错误"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ImageFunctionWarning(Warning):
    pass
