"""
***日志常量表***
"""


from hagike.utils.enum import uuid_t, advanced_enum, SuperEnum


@advanced_enum()
class LoggerConf(SuperEnum):
    """日志管理器配置表"""
    logdir = None
    """日志输入文件夹，若配置为 `None` 则不进行文件输出"""
    event_list = dict()
    """提供事件号到事件列表的映射"""
    others = dict()
    """其它项"""


@advanced_enum()
class LoggerStatus(SuperEnum):
    """日志管理器状态表"""
    inited = None
    """已创建"""
    running = None
    """正在运行"""
    stopped = None
    """已暂停"""
    ended = None
    """已结束"""
    destroyed = None
    """已销毁"""
