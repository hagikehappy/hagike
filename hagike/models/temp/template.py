"""模型的父类模板"""


import torch
import torch.nn as nn
from torchsummary import summary
from typing import Mapping, Any
from ...utils import *


class Model_Temp(nn.Module):
    """模型的通用模板父类"""
    _module_key = ('pre', 'tail', 'bone', 'head', 'final')

    def __init__(self, model_dict: Mapping[str, nn.Module] | None = None) -> None:
        """dict的key来自于('all', 'pre', 'tail', 'bone', 'head', 'final')， 如果有all则忽略其余部分"""
        super(Model_Temp, self).__init__()
        self.str2val = dict()

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

