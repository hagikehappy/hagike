"""
模型容器中最基本的构成，实现下层库与容器的统一接口
"""


import torch.nn as nn


class UnitTemp(nn.Module):
    """模型最小组成部分的模板"""

    def __init__(self, model):
        """初始化"""
        super(UnitTemp, self).__init__()
        self._model = model

    @property
    def model(self):
        return self._model



