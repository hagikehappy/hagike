"""
模型容器中最基本的构成，实现下层库与容器的统一接口 \n
"""

import torch
import torch.nn as nn
from torchsummary import summary
from typing import Mapping, Any, Sequence
from hagike.utils import *
from .const import ModuleInfo, uuid_t


class ModuleNode(nn.Module):
    """
    模型最小组成部分的模板，若希望有自定义实现，可用于被继承 \n
    同时也作为更高级的模型结构的基类 \n
    """

    def __init__(self, model: nn.Module, info: Dict[uuid_t, Any] | None = None):
        """
        初始化
        :param model 指定模型
        :param info 指定初始参数描述，若未输入描述则默认为固定值
        """
        super(ModuleNode, self).__init__()
        self._model = model
        self._info = ModuleInfo.dict_(info)

    @property
    def model(self) -> nn.Module: return self._model
    @property
    def device(self) -> str: return self._info[ModuleInfo.device]
    @property
    def dtype(self) -> torch.dtype: return self._info[ModuleInfo.dtype]
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self._model(x)

    def load_weights(self, weights_src: str | Any, is_path: bool = False) -> None:
        """根据is_path，选择从路径或从内存中加载指定部分的模块参数"""
        if is_path:
            state_dict = torch.load(weights_src, map_location=self.device)
        else:
            state_dict = weights_src
        self._model.load_state_dict(state_dict)

    def save_weights(self, path: str | None = None) -> Any:
        """根据path，选择加载指定部分的模块参数到路径或从内存中"""
        state_dict = self._model.state_dict()
        if path is not None:
            torch.save(state_dict, path)
        return state_dict

    def print_summary(self, input_size=(3, 224, 224)) -> None:
        """
        打印模型的情况，输入尺寸不包括batch，进行模型测试时的参数与当前参数一致 \n
        可用于检查模型可用性 \n
        """
        summary(self._model, input_size, device=self.device)

    def to(self,
           device: str | None = None,
           dtype: torch.dtype | None = None,
           is_train: bool | None = None) -> None:
        """转换模型类型，原位替换"""
        if device is not None:
            self._info[ModuleInfo.device] = device
            self._model = self._model.to(device=device)
        if dtype is not None:
            self._info[ModuleInfo.dtype] = dtype
            self._model = self._model.to(dtype=dtype)
        if is_train is not None:
            self._info[ModuleInfo.is_train] = is_train
            if is_train:
                self.train()
            else:
                self.eval()
