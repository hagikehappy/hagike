"""
模型容器的异常处理
"""


import warnings


class ModelError(Exception):
    """模型异常"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ModelWarning(Warning):
    """模型警告"""
    pass

