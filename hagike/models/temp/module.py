"""
模块的父类模板
"""


import torch
import torch.nn as nn
from torchsummary import summary
from typing import Mapping, Any, Sequence
from ...utils import *


class _ModuleModeError(Exception):
    """模块运行模式异常"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class IdentityModule(nn.Module):
    """恒等变换模块，用于占位"""
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x):
        """直接返回输入"""
        return x


@advanced_enum()
class ModuleKey(SuperEnum):
    """模块构成，未指定部分默认为None"""
    _sequence = (
        'pre', 'tail', 'bone', 'head', 'final'
    )
    all__ = None     # 如果启用了'_all'，则是忽略其它选项，并将该模块作为一个整体处理
    pre = None      # 预处理，将数据从原始格式转换为张量格式
    tail = None     # 尾部，将数据规范化，如批量归一化、嵌入等，以便于骨干网处理
    bone = None     # 骨干网，进行特征提取等操作
    head = None     # 头部，根据需求构造输出层格式
    final = None    # 激活层，获取最终输出


class ModuleTemp(nn.Module):
    """模块的通用模板父类"""

    def __init__(self, module_dict: Mapping[uuid_t, nn.Module] | None = None) -> None:
        """
        创建结构时的初始化
        """
        super(ModuleTemp, self).__init__()
        self._is_all: None | bool
        self._modules: List[nn.Module] | None
        self._model: Sequence[nn.Module] | None
        self._init(module_dict)

    def _init(self, module_dict: Mapping[uuid_t, nn.Module] | None = None) -> None:
        """
        在创建结构和刷新结构时调用； \n
        根据输入初始化各个模块组件，若字典为None则恒等变换，若组件为None则不会执行； \n
        模式在初始化创建后就不可改变； \n
        """
        # 恒等变换模式
        if module_dict is None:
            self._model = IdentityModule()
        else:
            self._is_all = True if ModuleKey.all__ in module_dict.keys() else False
            # 单组件模式
            if self._is_all:
                uuid = ModuleKey.all__
                self._model = module_dict[uuid]
            # 多组件模式
            else:
                self._modules: List[nn.Module] = ModuleKey.list_(module_dict, is_default=True)
                self._model = nn.Sequential()
                for module in self._modules:
                    if module is not None:
                        self._model.append(module)

    def refresh(self, module_dict: Mapping[uuid_t, nn.Module] | None = None) -> None:
        """刷新结构"""
        self._init(module_dict)

    def update(self, module_dict: Mapping[uuid_t, nn.Module] | None = None) -> None:
        """更新模型结构，如果更新后的模式与原模式不一致则报错"""
        # 检查更新后的模式与更新前是否一致
        is_all = None
        if module_dict is not None:
            is_all = True if ModuleKey.all__ in module_dict.keys() else False
        if is_all != self._is_all:
            raise _ModuleModeError(
                f"When updating module, You update it({self._is_all}) in a different way({is_all})!!!")
        # 进行更新
        if self._is_all is None:
            pass
        else:
            if self._is_all is True:
                self._model = module_dict[ModuleKey.all__]
            else:
                for uuid, value in module_dict.items():
                    self._modules[ModuleKey.get_index_(uuid)] = value
                    self._model = nn.Sequential()
                    for module in self._modules:
                        if module is not None:
                            self._model.append(module)

    def forward(self, x):
        """前向传播，若model为空的Sequential则会报错"""
        return self._model(x)

    def load_weights(self, module: uuid_t, weights_src: str | Any, is_path: bool) -> None:
        """根据is_path，选择从路径或从内存中加载指定部分的模块参数"""
        if is_path:
            state_dict = torch.load(weights_src, map_location=torch.device('cpu'))
        else:
            state_dict = weights_src
        if module == ModuleKey.all__:
            self._model.load_state_dict(state_dict)
        else:
            self._module[ModuleKey.get_index_(module)].load_state_dict(state_dict)

    def save_weights(self, module: uuid_t, path: str | None = None) -> Any:
        """根据path，选择加载指定部分的模块参数到路径或从内存中"""
        if module == ModuleKey.all__:
            state_dict = self._model.state_dict()
        else:
            state_dict = self._modules[ModuleKey.get_index_(module)].state_dict()
        if path is not None:
            torch.save(state_dict, path)
        return state_dict

    def print_summary(self, input_size=(3, 224, 224)) -> None:
        """打印模型的情况，输入尺寸不包括batch，进行模型测试时的参数与当前参数一致"""
        para = self.check_para(is_print=False)
        summary(self._model, input_size, device=para['device'])

    def trans_para(self, device: str | None = None,
                   dtype: torch.dtype | None = None,
                   is_train: bool | None = None) -> None:
        """转换模型类型"""
        if device is not None:
            self._model = self._model.to(device=device)
        if dtype is not None:
            self._model = self._model.to(dtype=dtype)
        if is_train is not None:
            if is_train:
                self.train()
            else:
                self.eval()

    def check_para(self, is_print: bool = True) -> dict:
        """返回当前模型参数"""
        para = dict()
        prop = next(self.parameters())
        para['device'] = 'cuda' if prop.is_cuda else 'cpu'
        para['dtype'] = prop.dtype
        if is_print:
            print(f"Model Property: {para}")
        return para


