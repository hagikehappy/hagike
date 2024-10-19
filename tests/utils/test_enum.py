from hagike.utils import *


def test_utils_enum():
    """utils.enum的测试用例"""

    # 举例
    @advanced_enum()
    class EnumExample(SuperEnum):
        """一个使用创建枚举类型的例子"""
        _sequence_ = (
            'z', 'b', 'SubExample1'
        )
        _a = 0
        z = 2
        b = 3

        class SubExample1(SuperEnum):
            _value_ = 4
            a = 5
            b = 6

        class SubExample2(SuperEnum):
            _sequence_ = (
                'c', 'd', 'SubSubExample'
            )
            _hide_ = True
            c = 7
            d = 8

            class SubSubExample(SuperEnum):
                e = 9
                f = 10

    # 测试
    EnumExample.SubExample2.print(is_value=True)
    EnumExample.tree(is_value=True)
    print(EnumExample.dict())
    for i in EnumExample.iter():
        print(i, end=', ')

