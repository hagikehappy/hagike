"""
死代码检测工具

用于检测Python项目中的未使用代码，包括：
1. 未使用的导入
2. 未使用的函数和类
3. 完全未被引用的文件
"""

from .detector import *
from .analyzer import *
from .cli import *