"""
**错误与警告**
"""


import warnings


class MSysInputError(Exception):
    """系统输入类型未实现"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class MSysInputWarning(Warning):
    pass


class MSysResponseError(Exception):
    """系统响应错误"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class MSysResponseWarning(Warning):
    pass
