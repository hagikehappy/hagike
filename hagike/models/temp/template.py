"""
:description
    模型的父类模板
:term
    module - 模块，仅包含ModuleKey中的固定术语的单一执行流
    model - 模型，由若干Module组成，各Module间呈现树形执行流
"""


import torch
import torch.nn as nn
from torchsummary import summary
from typing import Mapping, Any, Sequence
from ...utils import *


@advanced_enum()
class ModuleKey(SuperEnum):
    _sequence_ = (
        'all', 'pre', 'tail', 'bone', 'head', 'final'
    )
    all = None
    pre = None      # 预处理，将数据从原始格式转换为张量格式
    tail = None     # 尾部，将数据规范化，如批量归一化、嵌入等，以便于骨干网处理
    bone = None     # 骨干网，进行特征提取等操作
    head = None     # 头部，根据需求构造输出层格式
    final = None    # 激活层，获取最终输出


class Module_Temp(nn.Module):
    """模块的通用模板父类"""

    def __init__(self, model_dict: Mapping[uuid_t, nn.Module | Sequence[nn.Module]] | None = None) -> None:
        """
        模块是
        这里的模型模板使用树形结构去定义，这样可以处理多分支情况
        要求：输入的会
        """
        super(Module_Temp, self).__init__()

        if model_dict is None:
            self.model = None
        elif 'all' in self.module:
            self.model = model_dict['all']
        else:
            for key in model_dict:
                if key not in self._module_key:
                    add_msg(MsgLevel.Error.value, f"Key {key} Not In Module Key")
            self.model = nn.Sequential()
            index = 0
            for key in self._module_key:
                if key in model_dict:
                    self.str2val[key] = index
                    self.model.append(model_dict[key])
                    index += 1

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def load_weights(self, module: str, weights_src: str | Any, is_path: bool) -> None:
        """根据is_path，选择从路径或从内存中加载模块参数"""
        if is_path:
            state_dict = torch.load(weights_src, map_location=torch.device('cpu'))
        else:
            state_dict = weights_src
        if module == 'all':
            self.load_state_dict(state_dict)
        else:
            self.model[self.str2val[module]].load_state_dict(state_dict)

    def save_weights(self, weight_path: str) -> None:
        """ Saves weights to a .pt or .pth file """
        torch.save(self.state_dict(), weight_path)

    def print_summary(self, input_size=(3, 224, 224)) -> None:
        """打印模型的情况，输入尺寸不包括batch，进行模型测试时的参数与当前参数一致"""
        para = self.check_para(is_print=False)
        summary(self, input_size, device=para['device'])

    def trans_para(self, device: str | None = None,
                   dtype: torch.dtype | None = None,
                   is_train: bool | None = None) -> None:
        """转换模型类型"""
        if device is not None:
            self.to(device=device)
        if dtype is not None:
            self.to(dtype=dtype)
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

