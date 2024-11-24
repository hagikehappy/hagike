"""
***日志容器*** \n
这里的日志器希望实现以下功能： \n
1. 根据日志系统配置控制终端的消息输出 \n
2. 记录日志内容 \n
3. 提供日志统计与分析功能 \n
4. 提供保存与读取日志的方法 \n
5. 可以接入其它日志库，提供转换方法 \n
6. 可以进行日志合并等高级操作 \n
7. 可以提供跨进程的访问机制（可以作为子类拓展实现） \n
8. 定义基准消息格式 \n
9. 提供全局默认日志器 \n
.. todo::
    未全部完成
"""

from typing import Dict, Any
from .const import uuid_t, LoggerConf, LoggerStatus
from dataclasses import dataclass
from loguru import logger


@dataclass
class LogTemp:
    """日志项"""
    # todo: 实现对如pbar等回滚的日志项支持
    uuid: int = None
    """日志标识符"""
    time: int | float = None
    """时间"""
    level: uuid_t = None
    """级别"""
    src: str = None
    """来源"""
    event_main: uuid_t = None
    """主事件号"""
    event_sub: uuid_t = None
    """次事件号"""
    msg: str = None
    """消息"""
    else__: Any = None
    """其它附加要素"""


class LoggerTemp:
    """
    日志管理器模板 \n
    设计思路与实现目标： \n
    1. 为日志提供全局配置方法，使得可以控制日志记录与输出行为 \n
    2. 提供格式化的日志保存方法 \n
    3. 消息是日志的一种子形式，可以被包含在日志项范围内 \n
    """

    def __init__(self, conf: Dict[uuid_t, Any] | None = None, is_init: bool = True) -> None:
        """根据配置初始化日志模板"""
        self._conf = LoggerConf.dict_(conf)
        self._record = list()
        self._uuid = 0
        self._init_time = 0
        self._status = LoggerStatus.inited
        # todo: 基于配置的进一步初始化
        if is_init:
            self.init()

    def init(self) -> None:
        """启动日志管理器"""
        if self._status != LoggerStatus.inited:
            return
        # todo: 配置时间，写入日志头等
        self._status = LoggerStatus.running

    def end(self) -> None:
        """结束日志管理器"""
        # todo: 释放资源，写入日志尾等
        self._status = LoggerStatus.ended

    @property
    def status(self): return self._status

    def add_log(self, log: LogTemp, is_print: bool = True):
        """添加日志项"""
        # todo: 实现功能、过滤等
        if is_print:
            self.print_log(log)

    def print_msg(self, msg: str):
        """配置并打印消息，但不进行日志记录"""
        # todo: 实现可配置打印
        logger.info(msg)

    def print_log(self, log: LogTemp):
        """打印日志项"""
        # todo: 实现可配置打印
        print(log.msg)


# 全局日志记录器
logger_g = LoggerTemp()
