"""
***判断器***
"""


import torch.nn as nn
from .const import uuid_t, CriterionInfo
from ..temp import ModuleNode
from .error import TrainingError
from typing import Dict, Any


class CriterionTemp:
    """用于计算损失函数的容器"""

    def __init__(self, info: Dict[uuid_t, Any] | None = None) -> None:
        """创建评估器容器，需要先初始化才能使用"""
        self._info = CriterionInfo.dict_(info)
        self._crt = None
        self._is_init = False

    def init(self, model: ModuleNode | None = None) -> None:
        """初始化损失函数容器，可根据 `model` 动态生成评估器，但基准实现中不使用"""
        if self._is_init:
            return
        self._is_init = True
        self._crt = self._info[CriterionInfo.crt_type](**self._info[CriterionInfo.para])

    @property
    def crt(self) -> nn.Module:
        """返回评估器"""
        if self._is_init:
            return self._crt
        else:
            raise TrainingError(f"visiting crt when criterion uninitialized!!!")

    def __call__(self, *args, **kwargs):
        """重定向自身调用至 `self._crt`"""
        return self.crt(*args, **kwargs)

    def __getattr__(self, name: str):
        """重定向属性或方法至 `self._crt`"""
        return getattr(self.crt, name)

