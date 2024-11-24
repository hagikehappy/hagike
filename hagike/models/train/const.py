"""
模型容器的常量表
"""

from hagike.utils.enum import uuid_t, SuperEnum, advanced_enum


@advanced_enum()
class TrainerInfo(SuperEnum):
    """训练参数表"""
    max_epochs = 100
    """最大轮数"""
    device = 'cuda'
    """训练设备"""
    others = dict()
    """其它"""


@advanced_enum()
class EvaluatorInfo(SuperEnum):
    """评估器参数表"""
    effector_type = None
    """效果生成器类型"""
    para = dict()
    """Dict[str, Any], 效果参数"""
    others = dict()
    """其它"""


@advanced_enum()
class MonitorInfo(SuperEnum):
    """监视器参数表"""
    logdir = None
    """监视器日志的存储位置"""
    autostart = True
    """是否自动启动外置监视进程"""
    others = dict()
    """其它"""


@advanced_enum()
class DatasetInfo(SuperEnum):
    """数据集参数表"""
    train_rate = None
    """训练集数量占比"""
    para = dict()
    """Dict[str, Any], Dataset类一次封装的配置参数（外部创建参数）"""
    sub_para = dict()
    """Dict[str | uuid_t, Any], Dataset类二次封装的配置参数（内部container配置参数）"""
    others = dict()
    """其它"""


@advanced_enum()
class DataLoaderInfo(SuperEnum):
    """数据加载器参数表"""
    para = dict()
    """Dict[str, Any], 如果定义了para，则默认训练部分与评估部分参数配置一致"""
    train_para = dict()
    """训练部分DataLoader的参数表"""
    val_para = dict()
    """评估部分DataLoader的参数表"""
    others = dict()
    """其它"""


@advanced_enum()
class OptimInfo(SuperEnum):
    """训练参数表"""
    lr = 0.001
    """学习率"""
    op_type = None
    """优化器类型"""
    para = dict()
    """Dict[str, Any], 优化器参数"""
    others = dict()
    """其它"""


@advanced_enum()
class CriterionInfo(SuperEnum):
    """损失函数参数表"""
    crt_type = None
    """损失函数类型"""
    para = dict()
    """Dict[str, Any], 损失函数参数"""
    others = dict()
    """其它"""
