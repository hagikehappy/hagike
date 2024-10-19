"""
高级Enum类 \n
term: \n
    SuperEnum - 高级枚举类的父类模板 \n
    advanced_enum - 根级的装饰器，用于自动生成枚举类的各项配置 \n
notice: \n
    对于枚举成员： \n
    1. 本身的值在定义时是value，在访问时是uuid \n
    2. index不应该被外部访问，而仅作为迭代器的索引值 \n
    3. 命名不能以'__'开头，否则会被忽略 \n
    4. 值如果是继承了SuperEnum的枚举类型，则会递归导入，需要确保此处枚举类的归属是唯一的，否则uuid会被多次修改； \n
        否则这需要以实例导入 \n
    5. 如果未启用顺序访问索引，无法保证Enum的书写顺序就是index顺序，顺序是由Python机制决定的（默认是按名称顺序） \n
    6. '_xxx_'方式命名的枚举量为内置属性，不可用常规枚举量占用，且只能配置其中的配置字部分，否则会报错 \n
    7. 对于'_xxx'方式命名的枚举量，这里会将其作为隐藏枚举量，不会被包含在index列表中，也不会分配index值 \n
    8. 根类的值是会被包含在uuid及其映射中的，但由于其没有父类，因而不会被包含在任何index列表中，其index本身总为0 \n
"""


from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Sequence, Set, Iterator
from copy import deepcopy


# 重定义Enum中的标识符类型
uuid_t = int
index_t = int
# SuperEnum类型配置字，在父类SuperEnum中定义的值为默认值
enum_conf_word = ('_value_', '_sequence_', '_hide_', '_blank_')
"""
:param _value_:  \n
    对于group本身的赋值需要写在成员_value_中，否则会被视为None，访问时依然通过value \n
:param _sequence_:  \n
    如果在某子类下启用顺序访问索引，则需要赋值成员_sequence_: Tuple[str]； \n
    其中需要按顺序列出所有成员名称；如果未列全或有不存在的成员名称则在初始化时会报错 \n
:param _hide_: \n
    是否将类本身的值作为隐藏枚举值 \n
:param _blank_:  \n
    打印时的单位空格长度 \n
"""
# SuperEnum类型隐藏字
enum_hide_word = ('_uuid_', '_pack_', '_length_',
                  '_index2uuid_', '_uuid2pack_', '_uuid2sub_',
                  '_uuid_all_', '_uuid_hide_', '_uuid_sub_')
"""
:param _uuid_: \n
    子类本身的唯一标识符 \n
:param _pack_: \n
    存储子类本身的信息，是信息的打包形式 \n
:param _length_: \n
    子类的非隐藏成员数量，不包括子类本身 \n
:param _index2uuid_:  \n
    子类的非隐藏成员索引到唯一标识符的映射，不包括子类本身 \n
:param _uuid2pack_:  \n
    子类下的唯一标识符到数据包的映射，不包括子类本身 \n
:param _uuid2sub_: \n
    子类下的唯一标识符到孙类的映射 \n
:param _uuid_all_: \n
    子类下所有唯一标识符的列表，其中非隐藏部分在前，隐藏部分在后 \n
:param _uuid_hide_: \n
    子类下所有隐式枚举成员的集合 \n
:param _uuid_sub_: \n
    子类下所有孙类枚举成员的集合 \n
"""


class _EnumOccupiedError(Exception):
    """枚举类关键字占用异常"""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class _EnumSequenceError(Exception):
    """枚举类顺序访问索引异常"""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class _EnumUuidError(Exception):
    """枚举类的uuid不存在"""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class _EnumTypeError(Exception):
    """枚举类的配置项的类型不正确"""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


@dataclass
class _EnumPack:
    """常量数据包"""
    uuid: uuid_t = None
    index: index_t = None
    name: str = None
    value: Any = None

    def print(self, is_value: bool = False) -> None:
        """打印值，is_value表示是否打印值"""
        print(f"{self.name}({self.index}) -> {self.uuid}", end='')
        if is_value:
            print(f": {self.value}", end='')
        print()


class SuperEnum:
    """Enum类的父类"""
    # 配置属性
    _value_: Any | None = None
    _sequence_: Tuple[str] = None
    _hide_: bool = False
    _blank_: int = 4
    # 隐藏属性
    _uuid_: uuid_t
    _pack_: _EnumPack
    _length_: int
    _index2uuid_: List[uuid_t]
    _uuid2pack_: Dict[uuid_t, _EnumPack]
    _uuid2sub_: Dict[uuid_t, Any]
    _uuid_all_: List[uuid_t]
    _uuid_hide_: Set[uuid_t]
    _uuid_sub_: Set[uuid_t]

    @classmethod
    def get_uuid(cls):
        """获得类本身的uuid"""
        return cls._uuid_

    @classmethod
    def get_name(cls, uuid: uuid_t):
        """获得枚举量的名称"""
        return cls._uuid2pack_[uuid].name

    @classmethod
    def get_index(cls, uuid: uuid_t):
        """获得枚举量的名称"""
        return cls._uuid2pack_[uuid].index

    @classmethod
    def get_value(cls, src: index_t | uuid_t, index_or_uuid: bool = False) -> Any:
        """返回深拷贝赋值，index_or_uuid决定根据index还是uuid返回值"""
        uuid = cls._index2uuid_[src] if index_or_uuid else src
        pack_n = cls._uuid2pack_[uuid]
        return deepcopy(pack_n.value)

    @classmethod
    def dict(cls, enum_dict: Dict[uuid_t, Any] = None, is_force: bool = True) -> Dict[uuid_t, Any]:
        """
        填补Enum类中不在常量表部分的默认赋值，生成dict； \n
        如果选中is_force则会检查enum_dict中的key是否都在enum类的非隐藏部分中，若不满足则会报错； \n
        此项检查用于排除不正常输入的字典项，如隐藏的enum成员。 \n
        """
        if enum_dict is None:
            enum_dict = dict()
        uuid_dict = enum_dict.keys()
        for index in range(cls._length_):
            uuid = cls._index2uuid_[index]
            if uuid not in uuid_dict:
                enum_dict[uuid] = cls.get_value(uuid, False)
        if is_force:
            if cls._length_ != len(uuid_dict):
                raise _EnumUuidError(f"ERROR: dict(len={len(uuid_dict)}) is not included in enum(len={cls._length_})!!!")
        return enum_dict

    @classmethod
    def iter(cls) -> Iterator[int]:
        """返回index2uuid迭代器"""
        return iter(cls._index2uuid_)

    @classmethod
    def print_pack(cls, uuid: uuid_t, blank: int = 0, is_value: bool = False) -> None:
        """打印单个枚举量的信息"""
        blank_str = '' + ' ' * blank
        pack = cls._uuid2pack_[uuid]
        print(blank_str, end='')
        pack.print(is_value)

    @classmethod
    def print(cls, is_value: bool = False) -> None:
        """打印枚举类单级信息"""
        print()
        cls._pack_.print(is_value)
        # 优先顺序打印非隐藏值
        for uuid in cls._uuid_all_:
            cls.print_pack(uuid, cls._blank_, is_value)
        print()

    @classmethod
    def tree(cls, is_value: bool = False):
        """以树形结构递归打印该枚举类信息"""

        def regress_enum(cls_n: Any, blank_n: int) -> None:
            """递归列举"""
            # 优先顺序打印非隐藏值
            for uuid_n in cls_n._uuid_all_:
                cls_n.print_pack(uuid_n, blank_n, is_value)
                if uuid_n in cls_n._uuid_sub_:
                    regress_enum(cls_n._uuid2sub_[uuid_n], blank_n + cls_n._blank_)

        # 递归入口
        print()
        cls._pack_.print(is_value)
        regress_enum(cls, cls._blank_)
        print()


def advanced_enum():
    """
    该函数作为常量表的装饰器，自动建立映射，子类与子成员均视为常量，封装为常量类型，仅用于顶级Enum。
    """
    def decorator(cls):
        """装饰器，进行常量封装"""

        def check_key(keys: Sequence, all_or_hide: bool = True) -> None:
            """检查是否存在关键字冲突，all_or_hide指定全部检查还是仅检查隐藏属性"""
            for word in keys:
                if word in enum_hide_word:
                    raise _EnumOccupiedError(f"ERROR: {word} in enum_hide_word, change a Name!!!")
            if all_or_hide:
                for word in keys:
                    if word in enum_conf_word:
                        raise _EnumOccupiedError(f"ERROR: {word} in enum_conf_word, change a Name!!!")

        def check_conf(cls_n: Any) -> None:
            """检查枚举类的配置项的类型与值是否正确"""
            if not isinstance(cls_n._hide_, bool):
                raise _EnumTypeError(f"ERROR: _hide_ typeof {type(cls_n._hide_)} but not bool!!!")
            if isinstance(cls_n._blank_, int):
                if cls_n._blank_ < 0:
                    raise _EnumTypeError(f"ERROR: _blank_({cls_n._blank_}) < 0!!!")
            else:
                raise _EnumTypeError(f"ERROR: _blank_ typeof {type(cls_n._blank_)} but not int!!!")

        def regress_enum(uuid_n: uuid_t, cls_n: Any) -> uuid_t:
            """逐目录递归赋值uuid常量表，不会赋值顶级enum组"""
            uuid2pack_n: Dict[uuid_t, _EnumPack] = dict()
            index2uuid_n: List[uuid_t | None] = list()
            uuid_hide_n: Set[uuid_t] = set()
            uuid2sub_n: Dict[uuid_t, Any] = dict()
            index_n = 0
            all_attrs_n = dir(cls_n)
            all_attrs_n.reverse()   # 默认普通枚举量在前，子类在后

            # 检查是否存在关键字占用
            check_key(all_attrs_n, all_or_hide=False)
            # 检查配置字是否合法
            check_conf(cls_n)

            # 判断是否启用局部顺序映射表，如果启用则判断是否合法（是否恰好一致）并调换顺序
            seq_n, is_seq, seq_len = cls_n._sequence_, False, None
            if seq_n is not None:
                is_seq, seq_len = True, len(seq_n)
                # 检查以确保_sequence_中没有关键字冲突
                check_key(seq_n, all_or_hide=True)
                index2uuid_n = [None for _ in range(seq_len)]

            for attr_n in all_attrs_n:
                # 排除魔法属性和父类方法，'_value_'在父级中设置，不在本级设置
                if attr_n.startswith('__') or hasattr(cls_n.__base__, attr_n):
                    continue
                # 排除内置属性，并确保不存在自定义的内置属性
                elif attr_n.startswith('_') and attr_n.endswith('_'):
                    if attr_n not in enum_conf_word:
                        raise _EnumOccupiedError(f"ERROR: {attr_n} not in enum_conf_word, change a Name!!!")
                else:
                    # 重置标志位
                    is_hide, is_sub = False, False
                    # 赋值枚举类型
                    val_n = getattr(cls_n, attr_n)
                    # 判断类型是否为子类
                    if isinstance(val_n, type) and issubclass(val_n, SuperEnum):
                        is_sub = True
                    # 递归并处理子类
                    if is_sub:
                        # 先递归
                        uuid_n = regress_enum(uuid_n, val_n)
                        # 赋值子级group属性
                        uuid2sub_n[uuid_n] = val_n
                        pack_n = _EnumPack(uuid=uuid_n, name=attr_n, value=val_n._value_)
                        val_n._uuid_, val_n._pack_ = uuid_n, pack_n
                    # 处理一般枚举成员
                    else:
                        pack_n = _EnumPack(uuid=uuid_n, name=attr_n, value=val_n)
                        setattr(cls_n, attr_n, uuid_n)
                    # 检查是否为隐藏枚举量并处理
                    if is_sub:
                        if val_n._hide_:
                            is_hide = True
                    else:
                        if attr_n.startswith('_'):
                            is_hide = True
                    if is_hide:
                        uuid_hide_n.add(uuid_n)
                    # 如果为隐藏属性，则跳过配置index过程
                    if not is_hide:
                        # 赋值索引值，如果启用了顺序索引则填入对应位置，否则挂到最后
                        if is_seq:
                            try:
                                index = seq_n.index(attr_n)
                                index2uuid_n[index] = uuid_n
                            except ValueError:
                                raise _EnumSequenceError(f"ERROR: '{attr_n}' is not in _sequence_!!!")
                        else:
                            index = index_n
                            index2uuid_n.append(uuid_n)
                        pack_n.index = index
                        # 刷新计数器
                        index_n += 1
                    else:
                        pack_n.index = None
                    uuid2pack_n[uuid_n] = pack_n
                    uuid_n += 1
            # 如果启用了顺序索引，则检查_sequence_是否全部被包含
            if is_seq:
                if index_n != seq_len:
                    raise _EnumSequenceError(f"ERROR: index_n({index_n}) != _sequence_({seq_len})!!!")
            # 赋值本级group属性
            cls_n._index2uuid_ = index2uuid_n
            cls_n._length_ = index_n
            cls_n._uuid2pack_ = uuid2pack_n
            cls_n._uuid_hide_ = uuid_hide_n
            cls_n._uuid2sub_ = uuid2sub_n
            cls_n._uuid_all_ = index2uuid_n + list(uuid_hide_n)
            cls_n._uuid_sub_ = set(uuid2sub_n.keys())

            return uuid_n

        # 递归入口
        uuid = 0
        uuid = regress_enum(uuid, cls)
        # 赋值根目录本身的属性，本身一般仅用于占位，无实际意义
        cls._uuid_ = uuid
        cls._pack_ = _EnumPack(uuid=uuid, index=(None if cls._hide_ else 0), name=cls.__name__, value=cls._value_)
        return cls

    return decorator

