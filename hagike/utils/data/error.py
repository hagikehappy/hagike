"""
***异常处理***
"""


import warnings


class DataPackError(Exception):
    """数据包错误"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class DataPackWarning(Warning):
    """数据包警告"""
    pass
