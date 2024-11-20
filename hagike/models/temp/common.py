"""
***常用 `module_node`***
"""


from module_node import *


class IdentityUnit(nn.Module):
    """恒等变换单元"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """直接返回输入"""
        return x


class IdentityModel(ModuleNode):
    """恒等变换模块"""
    def __init__(self):
        super().__init__(IdentityUnit())

