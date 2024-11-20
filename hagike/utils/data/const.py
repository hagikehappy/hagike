"""
***常量表***
"""


from hagike.utils.enum import *


@advanced_enum()
class DataSrc(SuperEnum):
    """数据源类型"""
    mem = None
    """内存类"""
    file = None
    """pkl文件类"""
    compressed = None
    """压缩类"""
    else__ = None
    """其它类"""


@advanced_enum()
class DataForm(SuperEnum):
    """数据格式类型"""
    direct = None
    """直接可用"""
    mem_cached = None
    """内存缓存"""
    else__ = None
    """其它类"""


@advanced_enum()
class DataCompress(SuperEnum):
    """
    数据压缩类型 \n
    """
    else__ = None
    """其它类"""


@advanced_enum()
class DataInfo(SuperEnum):
    """
    数据包自定义数据类型
    """
    load_func = None
    """自定义加载函数"""
    load_check = None
    """自定义检查函数"""
    form_func = None
    """自定义格式转换函数"""
    mem_cacher = None
    """内存缓存器，如果希望从内存缓存读取数据，则需要指定"""
    compressed_type = None
    """压缩类型"""
    compressed_func = None
    """压缩函数"""
    others = dict()
    """用户自定义项，是一个字典"""
