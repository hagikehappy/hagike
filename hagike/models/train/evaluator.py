"""
***评估器***
"""


import torch
import torch.nn as nn
from .const import uuid_t, EvaluatorInfo
from typing import Dict, Any, Tuple
from ..temp import ModuleNode, ModuleMode
from .error import TrainingError
from .dataloader import DataLoaderTemp
from .criterion import CriterionTemp


class EffectorTemp:
    """效果生成器，这里的基准类实现了分类网络的准确率评估"""
    def __init__(self):
        """初始化"""
        self._activator = nn.Softmax(dim=-1)

    def set(self):
        """生成初始 `effect`"""
        return 0.0

    def gen(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        单次获取 `effect` \n
        :param outputs: (batch_size, num_classes), 独热码, 0.0 ~ 1.0, float32
        :param labels: (batch_size, num_classes), 独热码, 1.0, float32
        """
        # torch..max 返回 (值, 索引)
        _, predicted = torch.max(self._activator(outputs), 1)
        _, labeled = torch.max(labels, 1)
        return (predicted == labeled).sum()

    def add(self, a, b):
        """效果累积函数"""
        return a + b

    def avg(self, effect, num):
        """平均化效果"""
        return effect / num


class EvaluatorTemp:
    """评估器"""

    def __init__(self, info: Dict[uuid_t, Any] | None = None):
        """创建评估器容器"""
        self._info = EvaluatorInfo.dict_(info)
        self._effector: EffectorTemp = self._info[EvaluatorInfo.effector_type](**self._info[EvaluatorInfo.para])
        self._model = None
        self._dataloader = None
        self._criterion = None
        self._is_init = False

    def init(self, model: ModuleNode, dataloader: DataLoaderTemp, criterion: CriterionTemp):
        """初始化，配置模型与测试数据集"""
        if self._is_init:
            return
        self._is_init = True
        self._model, self._dataloader, self._criterion = model, dataloader, criterion

    def evaluate(self) -> Tuple[float, Any]:
        """评估过程封装"""
        if self._is_init:
            return self._evaluate()
        else:
            raise TrainingError(f"Evaluate when container uninitialized!!!")

    def _evaluate(self) -> Tuple[float, Any]:
        self._model.to(mode=ModuleMode.val)
        val_loss, sample_num = 0.0, 0
        val_effect = self._effector.set()
        with torch.no_grad():
            for i, data in enumerate(self._dataloader.val):
                inputs, labels = data
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                effect = self._effector.gen(outputs, labels)
                val_loss += loss * inputs.size(0)
                sample_num += inputs.size(0)
                val_effect = self._effector.add(val_effect, effect)
            val_loss /= sample_num
            val_effect = self._effector.avg(val_effect, sample_num)
        return val_loss, val_effect
