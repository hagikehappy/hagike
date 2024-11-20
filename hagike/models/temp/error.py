"""
模型容器的异常处理
"""


import warnings


class ModuleModeError(Exception):
    """模块运行模式异常"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class ModuleModeWarning(Warning):
    """模块运行模式警告"""
    pass

