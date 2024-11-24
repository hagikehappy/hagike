"""
***优化器容器***
"""


import torch.optim as optim
from .const import uuid_t, OptimInfo
from ..temp import ModuleNode
from .error import TrainingError
from typing import Dict, Any


class OptimTemp:
    """优化器容器"""

    def __init__(self, info: Dict[uuid_t, Any] | None = None) -> None:
        """创建优化器容器，容器需要先外部初始化才能使用"""
        self._info = OptimInfo.dict_(info)
        self._op = None
        self._is_init = False

    def init(self, model: ModuleNode) -> None:
        """初始化优化器"""
        if self._is_init:
            return
        self._is_init = True
        self._op = self._info[OptimInfo.op_type](
            model.parameters(), lr=self._info[OptimInfo.lr],
            **self._info[OptimInfo.para]
        )

    @property
    def op(self) -> optim.Optimizer:
        """返回优化器"""
        if self._is_init:
            return self._op
        else:
            raise TrainingError(f"visiting op when optimizer uninitialized!!!")

    def __getattr__(self, name: str):
        """重定向属性或方法至 `self._op`"""
        return getattr(self.op, name)
