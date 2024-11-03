"""
**错误或警告**
"""


import warnings


class MEngineCallError(Exception):
    """调用类型未实现"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class MEngineCallWarning(Warning):
    pass
