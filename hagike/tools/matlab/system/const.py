"""
常量表
"""


from hagike.utils.enum import SuperEnum, advanced_enum, uuid_t


@advanced_enum()
class MSysInput(SuperEnum):
    """输入类型"""
    impulse = None
    """冲激响应"""
    step = None
    """阶跃响应"""
    ramp = None
    """斜坡响应"""
    else__ = None
    """非标准类型"""
