"""
***监视器***
"""


from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from .const import uuid_t, MonitorInfo, TrainerInfo
from typing import Dict, Any
from hagike.log import LoggerTemp, LogTemp, LoggerConf
import os
from tqdm import tqdm
from .error import TrainingError


class TrainerMonitor:
    """
    ***模型监视器***
    """

    def __init__(self, info: Dict[uuid_t, Any] | None = None) -> None:
        """
        按照参数配置监视器，默认监视器实现如下功能： \n
        1. 提供各种可能用到的统一接口，提供最基本的接口规范 \n
        2. 提供一个最基本的实现，通过日志方式提供记录与分析功能，控制终端输出与显示 \n
        3. 接入外部监控器，如 `Tensorboard` \n
        注意：在创建监视器后必须在外部进行init才能正常工作！！！ \n

        :param info: `MonitorInfo` 监视器参数 \n
        """
        self._info = MonitorInfo.dict_(info)
        # 创建日志子目录
        now_date = datetime.now()
        formatted_date = now_date.strftime("%Y-%m-%d-%H-%M-%S")
        self._info[MonitorInfo.logdir] = os.path.join(self._info[MonitorInfo.logdir], formatted_date)
        # todo: 根据TrainerMonitor的需求配置LoggerTemp
        self._logger = LoggerTemp(is_init=False, conf={
            LoggerConf.logdir: self._info[MonitorInfo.logdir]
        })
        self._logger.init()
        # 这里默认使用 `Tensorboard` 作为外部监视器
        os.makedirs(self._info[MonitorInfo.logdir], exist_ok=True)
        self._writer = SummaryWriter(self._info[MonitorInfo.logdir])
        self._is_init = False
        self._train_info: Dict[uuid_t, Any] | None = None
        self._pbar: tqdm | None = None

    def init(self, train_info: Dict[uuid_t, Any]) -> None:
        """接入训练参数，初始化监视器"""
        if self._is_init:
            return
        self._is_init = True
        self._train_info = train_info
        # 用子进程打开
        if self._info[MonitorInfo.autostart]:
            import subprocess
            process = subprocess.Popen(
                [f'konsole --hold -e tensorboard --logdir={self._info[MonitorInfo.logdir]}'],
                shell=True, close_fds=True)
        # todo: 在日志中保存训练参数，并在日志文件夹中对训练参数进行备份

    @property
    def writer(self) -> SummaryWriter:
        if self._is_init:
            return self._writer
        else:
            raise TrainingError(f"visiting writer when train monitor uninitialized!!!")

    @property
    def logger(self) -> LoggerTemp:
        if self._is_init:
            return self._logger
        else:
            raise TrainingError(f"visiting logger when train monitor uninitialized!!!")

    @property
    def pbar(self) -> tqdm:
        if self._is_init:
            return self._pbar
        else:
            raise TrainingError(f"visiting pbar when trainer not started!!!")

    @property
    def logdir(self) -> str:
        return self._info[MonitorInfo.logdir]

    def start(self):
        """开始训练"""
        self._logger.add_log(LogTemp(msg="Begin Training Process"))
        self._pbar = tqdm(total=self._train_info[TrainerInfo.max_epochs])

    def new_epoch(self, epoch: int):
        """记录新一轮训练"""
        self.pbar.update()
        self.logger.add_log(LogTemp(msg=f"Begin Training Epoch {epoch}"), is_print=False)

    def record_effect(self, epoch: int, train_loss: float, val_loss: float, effect: float):
        """记录训练效果"""
        self.logger.add_log(
            LogTemp(msg=f"Epoch {epoch}: train - {train_loss}; val - {val_loss}; effect - {effect}"), is_print=False)
        # todo: 通过配置方法定义名称
        self.writer.add_scalar('Train Loss', train_loss, epoch)
        self.writer.add_scalar('Val Loss', val_loss, epoch)
        self.writer.add_scalar('Effect', effect, epoch)

    def end(self):
        """关闭监视器"""
        self.pbar.close()
        self.logger.add_log(LogTemp(msg="End Training Process"))

    def destroy(self):
        """销毁监视器"""
        self.writer.close()
        self.logger.end()
