"""
***训练器***
"""
import os.path

import torch.nn as nn
import torch
from typing import Dict, Any
from .const import uuid_t, TrainerInfo
from ..temp import ModuleNode, ModuleMode
from .monitor import TrainerMonitor
from .optimizer import OptimTemp
from .criterion import CriterionTemp
from .evaluator import EvaluatorTemp
from .dataloader import DataLoaderTemp


class TrainerTemp:
    """训练器模板"""

    def __init__(self, model: ModuleNode,
                 optim: OptimTemp,
                 criterion: CriterionTemp,
                 dataloader: DataLoaderTemp,
                 monitor: TrainerMonitor,
                 evaluator: EvaluatorTemp,
                 info: Dict[uuid_t, Any] | None = None):
        """
        初始化容器，基准训练器本身不负责具体实现，仅负责调度各个子容器并协调配置 \n

        :param model: 模型 \n
        :param optim: 优化器 \n
        :param criterion: 损失函数计算 \n
        :param dataloader: 数据管理器 \n
        :param monitor: 监视器与日志记录器 \n
        :param evaluator: 评估器 \n
        :param info: 配置信息 \n

        ..todo::
            定义训练截止条件
        """
        # 检查信息表
        info = TrainerInfo.dict_(info)
        # 容器赋值
        self._model = model
        self._dataloader, self._optim, self._criterion = dataloader, optim, criterion
        self._monitor, self._evaluator = monitor, evaluator
        self._info = info
        # 交叉配置各部分
        self._model.to(self._info[TrainerInfo.device])
        self._monitor.init(self._info)
        self._optim.init(self._model)
        self._criterion.init(self._model)
        self._dataloader.init(self._model.info)
        # 最后配置评估器
        self._evaluator.init(self._model, self._dataloader, self._criterion)

    def train(self):
        """按照配置方式进行训练"""
        self._monitor.start()
        max_effect = 0.0
        now_path, old_path = None, None
        for epoch in range(self._info[TrainerInfo.max_epochs]):
            self._monitor.new_epoch(epoch)
            train_loss, sample_num = 0.0, 0
            self._model.to(mode=ModuleMode.train)
            for i, data in enumerate(self._dataloader.train):
                inputs, labels = data
                self._optim.zero_grad()
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optim.step()
                train_loss += loss * inputs.size(0)
                sample_num += inputs.size(0)
            train_loss /= sample_num
            val_loss, val_effect = self._evaluator.evaluate()
            self._monitor.record_effect(epoch, train_loss, val_loss, val_effect)
            # todo：进行可配置的参数保存，这里直接实现保存最大值
            if val_effect > max_effect:
                max_effect = val_effect
                now_path = os.path.join(self._monitor.logdir, f"{epoch}: {max_effect}")
                self._model.save_weights(now_path)
                if old_path is not None:
                    os.remove(old_path)
                old_path = now_path
        self._monitor.end()

    def end(self):
        """
        结束训练器并销毁资源占用 \n
        .. todo::
            未实现
        """
