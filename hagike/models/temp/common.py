"""
***常用 `module_node`***
"""


from .module import nn, ModuleNode, ModuleTemp, ModuleKey, uuid_t, ModuleMode


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


class ModuleTemp_MaskHead(ModuleTemp):
    """根据 (Train, Val) / Predict 模式决定是否掩码头部"""
    def _to_mode(self, mode: uuid_t) -> None:
        """在由预测转换为评估或训练时掩码头部"""
        super()._to_mode(mode)
        if mode == ModuleMode.predict:
            self.to_mask(ModuleKey.head, False)
        else:
            self.to_mask(ModuleKey.head, True)
