"""
模型容器的常量表
"""


from hagike.utils.enum import *
import torch


@advanced_enum()
class ModuleKey(SuperEnum):
    """模块构成"""
    _sequence = (
        'pre', 'tail', 'bone', 'head', 'final'
    )
    all__ = None    # 指代整个模型
    pre = None      # 预处理，将数据从原始格式转换为张量格式
    tail = None     # 尾部，将数据规范化，如批量归一化、嵌入等，以便于骨干网处理
    bone = None     # 骨干网，进行特征提取等操作
    head = None     # 头部，根据需求构造输出层格式
    final = None    # 激活层，获取最终输出


@advanced_enum()
class ModuleMode(SuperEnum):
    """模块运行模式"""
    train = None
    """训练模式"""
    val = None
    """评估模式"""
    predict = None
    """预测模式"""
    else__ = None
    """其它模式"""


@advanced_enum()
class ModuleInfo(SuperEnum):
    """模块信息表"""
    dtype = torch.float32
    """模块数据类型"""
    device = 'cpu'
    """模块所在设备"""
    mode = ModuleMode.predict
    """训练模式 / 评估模式 / 预测模式"""
    others = dict()
    """其它信息"""

