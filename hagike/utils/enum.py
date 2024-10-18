"""
高级的Enum类，对于枚举成员，要求：
1. 命名不能包含enum_key_word中的内容，否则初始化类定义时会报错；命名不能以'_'和'__'开头，否则会被忽略
2. 值如果是继承了SuperEnum的枚举类型，则会递归导入，需要确保此处枚举类的归属是唯一的，否则uuid会被多次修改；
    否则这需要以实例导入
3. 如果未启用顺序访问索引，无法保证Enum的书写顺序就是index顺序，顺序是由Python机制决定的（默认是按名称顺序）
4. 对于group本身的赋值需要写在成员_value_中，否则会被视为None，访问时依然通过value
5. 如果在某子类下启用顺序访问索引，则需要赋值成员_sequence_: Tuple[str]，其中需要按顺序列出所有成员名称；
    如果未列全或有不存在的成员名称则在初始化时会报错
6. 根类的值是不被包含在uuid及其映射中的，根类本身的值会被忽略
"""


from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Sequence
import collections.abc
from copy import deepcopy


# 重定义Enum中的标识符类型
uuid_t = int
index_t = int
# SuperEnum类型的关键字
enum_key_word = ('uuid', 'index', 'name', 'value', 'length', 'index2uuid', 'uuid2val')
"""
:param uuid: Enum根类下的枚举成员唯一标识符；根类本身的uuid表示整个根类的常量数量
:param index: Enum子类本身在Enum父类中的枚举成员索引，根类的index总为0
:param name: Enum子类本身的名称
:param value: Enum子类本身的值，根类的value总为None
:param length: Enum子类的成员数量，不包括子类本身
:param index2uuid: Enum子类的索引到唯一标识符的映射，不包括子类本身
:param uuid2val: Enum子类下的唯一标识符到值的映射，不包括子类本身
"""


class EnumOccupiedError(Exception):
    """枚举类关键字占用异常"""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class EnumSequenceError(Exception):
    """枚举类顺序访问索引异常"""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


@dataclass
class EnumPack:
    """常量数据包"""
    uuid: uuid_t = None
    index: index_t = None
    name: str = None
    value: Any = None

    def print(self, is_value: bool = False) -> None:
        """打印值"""
        print(f"{self.name}({self.index}) -> {self.uuid}", end='')
        if is_value:
            print(f": {self.value}")
        else:
            print()

    @staticmethod
    def pack(cls) -> Any:
        """打包枚举类本身的值"""
        cls._pack_ = EnumPack(uuid=cls.uuid, index=cls.index, name=cls.name, value=cls.value)
        return cls._pack_


class SuperEnum:
    """Enum类的父类"""
    uuid: uuid_t
    index: index_t
    name: str
    value: Any
    length: int
    index2uuid: List[uuid_t]
    uuid2val: Dict[uuid_t, EnumPack]
    # 隐藏属性
    _value_: Any
    _sequence_: Tuple[str]
    _pack_: EnumPack   # 存储子类本身的信息，是信息的打包形式
    _blank_ = 4     # 打印时的单位空格长度

    @classmethod
    def get(cls, uuid: uuid_t) -> EnumPack:
        """返回深拷贝赋值"""
        val_n = cls.uuid2val[uuid]
        # 如果是子类则打包
        if isinstance(val_n, type) and issubclass(val_n, SuperEnum):
            val_n = val_n._pack_
        return deepcopy(val_n)

    @classmethod
    def default(cls, enum_dict: Dict[uuid_t, Any] = None) -> Dict[uuid_t, Any]:
        """填补Enum类中不在常量表部分的默认赋值"""
        if enum_dict is None:
            enum_dict = dict()
        key_enum = cls.uuid2val.keys()
        key_dict = enum_dict.keys()
        for key in key_enum:
            if key not in key_dict:
                enum_dict[key] = cls.get(key)
        return enum_dict

    @classmethod
    def print(cls, is_value: bool = False) -> None:
        """打印枚举类单级信息"""
        print()
        cls._pack_.print(is_value)
        blank_str = '' + ' ' * cls._blank_
        for uuid in cls.index2uuid:
            val = cls.uuid2val[uuid]
            print(blank_str, end='')
            val.print(is_value)
        print()

    @classmethod
    def tree(cls, is_value: bool = False):
        """以树形结构递归打印该枚举类信息"""

        def regress_enum(cls_n: Any, blank_n: int) -> None:
            """递归打印"""
            blank_str = '' + ' ' * blank_n
            for uuid_n in cls_n.index2uuid:
                val_n = cls_n.uuid2val[uuid_n]
                print(blank_str, end='')
                val_n.print(is_value)
                # 如果是子类则递归
                val_n = getattr(cls_n, val_n.name)
                if isinstance(val_n, type) and issubclass(val_n, SuperEnum):
                    regress_enum(val_n, blank_n + cls._blank_)

        # 递归入口
        print()
        cls._pack_.print(is_value)
        regress_enum(cls, cls._blank_)
        print()


def advanced_enum():
    """
    该函数作为常量表的装饰器，自动建立映射，子类与子成员均视为常量，封装为常量类型，仅用于顶级Enum，配置：
    uuid(全局，唯一标识符，可用于全局遍历)；
    index(局部，当前group下的标识符，可用于局部遍历)；
    value(常量对应的拓展值，如描述类字符串，这由具体需求决定；
        默认情况下，父级group不会继承子级group的标志，若有需要则需在父级重新定义子级标志)
    length: 常量数量
    """
    def decorator(cls):
        """装饰器，进行常量封装"""

        def check_key(keys: Sequence) -> None:
            """检查是否存在关键字冲突"""
            for word in enum_key_word:
                if word in keys:
                    raise EnumOccupiedError(f"ERROR: {word} in enum occupied, change a Name!!!")

        def regress_enum(uuid_n: uuid_t, cls_n: Any) -> uuid_t:
            """逐目录递归赋值uuid常量表，不会赋值顶级enum组"""
            index2uuid_n: List[Any] = list()
            uuid2val_n: Dict[uuid_t, Any] = dict()
            index_n = 0
            all_attrs_n = dir(cls_n)
            all_attrs_n.reverse()   # 默认值在前，子类在后

            # 判断是否启用局部顺序映射表，如果启用则判断是否合法（是否恰好一致）并调换顺序
            is_seq, seq_n, seq_len = False, None, None
            if hasattr(cls_n, '_sequence_'):
                is_seq, seq_n = True, getattr(cls_n, '_sequence_')
                if not isinstance(seq_n, collections.abc.Sequence):
                    raise EnumSequenceError(f"ERROR: seq_n is not a sequence type but a {type(seq_n)}!!!")
                # 检查以确保_sequence_中没有关键字冲突
                check_key(seq_n)
                # 初始化seq结构
                seq_len = len(seq_n)
                index2uuid_n = [None for _ in range(seq_len)]

            for attr_n in all_attrs_n:
                # 排除内置属性和父类方法，'_value_'在父级中设置，不在本级设置
                if attr_n.startswith('__') or attr_n.startswith('_') or hasattr(cls_n.__base__, attr_n):
                    continue
                else:
                    # 重置子类标志位
                    is_sub = False
                    # 检查以确保本级enum没有关键字冲突
                    check_key(cls_n.__dict__)
                    # 赋值枚举类型
                    val_n, val_g = getattr(cls_n, attr_n), None
                    # 递归并处理子类或子类实例
                    if isinstance(val_n, type) and issubclass(val_n, SuperEnum):
                        is_sub = True
                        # 先递归
                        uuid_n = regress_enum(uuid_n, val_n)
                        # 赋值子级group属性
                        if hasattr(val_n, '_value_'):
                            val_g = getattr(val_n, '_value_')
                        val_n.uuid = uuid_n
                        val_n.name = attr_n
                        val_n.value = val_g
                    # 处理一般枚举成员
                    else:
                        pack_n = EnumPack(uuid=uuid_n, name=attr_n, value=val_n)
                        val_n = pack_n
                        setattr(cls_n, attr_n, val_n)
                    # 赋值索引值，如果启用了顺序索引则填入对应位置，否则挂到最后
                    if is_seq:
                        try:
                            index = seq_n.index(attr_n)
                            index2uuid_n[index] = uuid_n
                        except ValueError:
                            raise EnumSequenceError(f"ERROR: '{attr_n}' is not in _sequence_!!!")
                    else:
                        index = index_n
                        index2uuid_n.append(uuid_n)
                    val_n.index = index
                    # 如果为子类，设置子类本身的包值
                    if is_sub:
                        val_n = EnumPack.pack(val_n)
                    # 刷新计数器
                    index_n += 1
                    uuid2val_n[uuid_n] = val_n
                    uuid_n += 1
            # 如果启用了顺序索引，则检查_sequence_是否全部被包含
            if is_seq:
                if index_n != seq_len:
                    raise EnumSequenceError(f"ERROR: index_n({index_n}) != _sequence_({seq_len})!!!")
            # 赋值本级group属性
            cls_n.index2uuid = index2uuid_n
            cls_n.length = index_n
            cls_n.uuid2val = uuid2val_n
            return uuid_n

        # 递归入口
        uuid = 0
        uuid = regress_enum(uuid, cls)
        # 赋值根目录本身的属性，本身仅用于占位，无实际意义
        cls.uuid = uuid
        cls.index = 0
        cls.name = cls.__name__
        cls.value = None
        EnumPack.pack(cls)
        return cls

    return decorator


if __name__ == '__main__':
    print()

    # 举例
    @advanced_enum()
    class EnumExample(SuperEnum):
        """一个使用创建枚举类型的例子"""
        _sequence_ = (
            'z', 'b', 'SubExample1', 'SubExample2'
        )
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
            c = 7
            d = 8

            class SubSubExample(SuperEnum):
                e = 9
                f = 10

    # 测试
    EnumExample.print(is_value=True)
    EnumExample.tree(is_value=True)
    default_dict = EnumExample.default()
    print(default_dict)
