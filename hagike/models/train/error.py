"""
模型容器的异常处理
"""


import warnings


class TrainingError(Exception):
    """训练过程异常"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class TrainingWarning(Warning):
    """训练过程警告"""
    pass
