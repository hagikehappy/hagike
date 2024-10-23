from hagike.utils.enum import *


def test_utils_enum():
    """utils.enum的测试用例"""

    # 举例
    @advanced_enum()
    class EnumExample(SuperEnum):
        """一个使用创建枚举类型的例子"""
        _sequence = (
            'z', 'b', 'SubExample1'
        )
        a__ = 0
        z = 2
        b = 3

        class SubExample1(SuperEnum):
            _value = 4
            a = 5
            b = 6

        class SubExample2(SuperEnum):
            _sequence = (
                'c', 'd', 'SubSubExample'
            )
            _hide = True
            c = 7
            d = 8

            class SubSubExample(SuperEnum):
                e = 9
                f = 10

    # 测试
    EnumExample.SubExample2.print_(is_value=True)
    EnumExample.tree_(is_value=True)
    print(EnumExample.dict_())
    for i in EnumExample.iter_():
        print(i, end=', ')
    print()
    print(EnumExample.list_({EnumExample.b: 5, EnumExample.z: 6}))

