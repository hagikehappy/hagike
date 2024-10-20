"""
模型的父亲类模板
"""


from .module import ModuleTemp
import torch
import torch.nn as nn
from torchsummary import summary
from typing import Mapping, Any, Sequence
from ...utils import *


@dataclass
class ModelNode(nn.Module):
    """
    模型的DAG节点； \n
    输入会被打包为元组，由节点内模块进行拆解并处理； \n
    输出会自动分散到各个子节点 \n
    """

    def __init__(self, uuid: uuid_t, module: ModuleTemp, ):
        super(ModelNode, self).__init__()
        self.uuid = uuid
        self.module = module
        # TODO
        # self.inputs
        # self.outputs



class ModelTemp(nn.Module):
    """模型的通用模板父类"""

    def __init__(self):
        """
        根据输入初始化各个模块组件，模块递归构建模型
        """
        super(ModelTemp, self).__init__()




