"""
**常量表**
"""

from hagike.utils.enum import *


@advanced_enum()
class MCall(SuperEnum):
    """调用引擎的类型"""
    func = None
    """调用函数或创建实例或函数化封装的特殊关键字或调用函数句柄"""
    obj_value = None
    """对象成员属性"""
    obj_func = None
    """对象成员函数"""
    print = None
    """打印属性"""
    operator = None
    """运算符"""


# 将算术运算符转为函数，包括：
# 基本算术运算符（Basic）
#    加法（Addition）：+
#    减法（Subtraction）：-
#    乘法（Multiplication）：*
#    除法（Division）：/
#    左除（Left Division）：\（左除，用于矩阵 Ax = B，x = B \ A）
#    幂运算（Power）：^
# 点运算符（Dot）
#    点乘（Element-wise Multiplication）：.*
#    点除（Element-wise Division）：./
#    点幂（Element-wise Power）：.^
# 矩阵运算符（Matrix）
#    共轭转置（Conjugate Transpose）：'
#    非共轭转置（Non-conjugate Transpose）：.'
# 逻辑运算符（Logic）
#    与（AND）：&
#    或（OR）：|
#    非（NOT）：~
#    短路与（SAND）：&&
#    短路或（SOR）：||
#    大于（Bigger）：>
#    大于等于（Bigger Equal）：>=
#    小于（Smaller）: <
#    小于等于（Smaller Equal）: <=
#    等于（Equal）：==
#    不等于（Unequal）：~=
@advanced_enum()
class MOperator(SuperEnum):
    """操作符的类型，类别名字与 `m_operator.m` 中一致"""

    class basic(SuperEnum):
        add = '+'
        sub = '-'
        mul = '*'
        div = '/'
        ldiv = '\\'
        pow = '^'

    class dot(SuperEnum):
        elmul = '.*'
        eldiv = './'
        elpow = '.^'

    class matrix(SuperEnum):
        conj_transpose = "'"
        nonconj_transpose = ".'"

    class logic(SuperEnum):
        and_m = '&'
        or_m = '|'
        not_m = '~'
        and_short = '&&'
        or_short = '||'
        gt = '>'
        ge = '>='
        lt = '<'
        le = '<='
        eq = '=='
        ne = '~='
